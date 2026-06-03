from collections.abc import Callable
import dataclasses
import functools
import inspect
import re
from typing import Any, ParamSpec, TypeVar

import flax.nnx as nnx
import jax

P = ParamSpec("P")
R = TypeVar("R")


def module_jit(meth: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]:
    """用于 JIT 编译 `nnx.Module` 方法的高阶函数，并在过程中冻结模块状态。

    为什么不用 `nnx.jit`？出于某些原因，直接把 `nnx.jit` 用在 `nnx.Module` 方法上时，
    无论是 bound 还是 unbound 方法，都会占用远超必要的内存。推测这和它必须跟踪模块突变有关。
    此外，相比标准 `jax.jit`，`nnx.jit` 本身也有额外开销，因为每次调用都要遍历 NNX 模块图。
    详情见 https://github.com/google/flax/discussions/4224。

    `module_jit` 是一个替代方案，通过冻结模块状态来避开这些问题。`module_jit` 返回的函数行为与
    原方法完全一致，只是模块状态会固定为调用 `module_jit` 时的状态。`meth` 内部仍允许修改模块，
    但这些修改会在方法调用完成后被丢弃。
    """
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError("module_jit must only be used on bound methods of nnx.Modules.")

    graphdef, state = nnx.split(meth.__self__)

    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper


@dataclasses.dataclass(frozen=True)
class PathRegex:
    """使用正则表达式匹配路径的 NNX Filter。

    默认会用 `/` 分隔符拼接路径；可以通过设置 `sep` 参数覆盖。
    """

    pattern: str | re.Pattern
    sep: str = "/"

    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join(str(x) for x in path)
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """将函数应用到 state 中匹配 filter 的叶子节点。"""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)
