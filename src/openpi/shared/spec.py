from functools import partial
import importlib
from typing import Any, TypedDict


class ModuleSpec(TypedDict):
    """函数或类的 JSON 可序列化表示，可附带传给它的一些默认 args 和 kwargs。

    这适合在配置文件中指定某个类或函数，同时保持可序列化，并允许通过 ml_collections
    从命令行覆盖。

    用法：

        # Preferred way to create a spec:
        >>> from src.model.components.transformer import Transformer
        >>> spec = ModuleSpec.create(Transformer, num_layers=3)
        # Same as above using the fully qualified import string:
        >>> spec = ModuleSpec.create("src.model.components.transformer:Transformer", num_layers=3)

        # Usage:
        >>> ModuleSpec.instantiate(spec) == partial(Transformer, num_layers=3)
        # can pass additional kwargs at instantiation time
        >>> transformer = ModuleSpec.instantiate(spec, num_heads=8)

    注意：ModuleSpec 只是一个带强类型标注的字典别名，不是真正的类。因此从代码角度看，
    它就是一个普通字典。

    module (str): callable 所在的模块
    name (str): callable 在模块中的名称
    args (tuple): 传给 callable 的 args
    kwargs (dict): 传给 callable 的 kwargs
    """

    module: str
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @staticmethod
    def create(callable_or_full_name: str | callable, *args, **kwargs) -> "ModuleSpec":  # type: ignore
        """从 callable 或导入字符串创建 module spec。

        Args:
            callable_or_full_name (str or object): 对象本身，或完整限定导入字符串
                （例如 "src.model.components.transformer:Transformer"）。
        args (tuple, optional): 实例化时传给 callable。
        kwargs (dict, optional): 实例化时传给 callable。
        """
        if isinstance(callable_or_full_name, str):
            assert callable_or_full_name.count(":") == 1, (
                "If passing in a string, it must be a fully qualified import string "
                "(e.g. 'src.model.components.transformer:Transformer')"
            )
            module, name = callable_or_full_name.split(":")
        else:
            module, name = _infer_full_name(callable_or_full_name)

        return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)

    @staticmethod
    def instantiate(spec: "ModuleSpec"):  # type: ignore
        if set(spec.keys()) != {"module", "name", "args", "kwargs"}:
            raise ValueError(
                f"Expected ModuleSpec, but got {spec}. "
                "ModuleSpec must have keys 'module', 'name', 'args', and 'kwargs'."
            )
        cls = _import_from_string(spec["module"], spec["name"])
        return partial(cls, *spec["args"], **spec["kwargs"])

    @staticmethod
    def to_string(spec: "ModuleSpec"):  # type: ignore
        return (
            f"{spec['module']}:{spec['name']}"
            f"({', '.join(spec['args'])}"
            f"{', ' if spec['args'] and spec['kwargs'] else ''}"
            f"{', '.join(f'{k}={v}' for k, v in spec['kwargs'].items())})"
        )


def _infer_full_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__name__"):
        return o.__module__, o.__name__
    raise ValueError(
        f"Could not infer identifier for {o}. "
        "Please pass in a fully qualified import string instead "
        "e.g. 'src.model.components.transformer:Transformer'"
    )


def _import_from_string(module_string: str, name: str):
    try:
        module = importlib.import_module(module_string)
        return getattr(module, name)
    except Exception as e:
        raise ValueError(f"Could not import {module_string}:{name}") from e
