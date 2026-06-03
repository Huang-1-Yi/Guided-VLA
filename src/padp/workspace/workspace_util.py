from typing import List


def _get_nested_attr(obj, name: str):
    value = obj
    for part in name.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def freeze_encoder_modules(policy) -> List[str]:
    """Freeze common encoder modules when a policy exposes them."""
    candidate_names = [
        "obs_encoder",
        "enc",
        "encoder",
        "nets",
        "model.nets",
        "model.model.backbones",
        "model.model.backbone",
    ]

    frozen = []
    for name in candidate_names:
        module = _get_nested_attr(policy, name)
        if module is None:
            continue
        if hasattr(module, "eval"):
            module.eval()
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
            frozen.append(name)
    return frozen
