import dataclasses
import enum
import logging
import socket

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """支持的环境。"""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    CALVIN = "calvin"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """从已训练的 checkpoint 加载 policy。"""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """使用指定环境的默认 policy。"""


@dataclasses.dataclass
class Args:
    """serve_policy 脚本的参数。"""

    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None

    port: int = 8000
    record: bool = False

    max_cuda_mem_fraction: float = 0.9
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.CALVIN: Checkpoint(
        config="pi05_calvin",
        dir="checkpoints/pi05_calvin/calvin_test/29999",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """为指定环境创建默认 policy。"""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据给定参数创建 policy。"""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


class PolicyServerApp:
    def __init__(self, args: Args):
        self.args = args
        self.base_policy = create_policy(args)
        self.policy = self.base_policy
        if args.record:
            self.policy = _policy.PolicyRecorder(self.base_policy, "policy_records")

    def serve_forever(self) -> None:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

        server = websocket_policy_server.WebsocketPolicyServer(
            policy=self.policy,
            host="0.0.0.0",
            port=self.args.port,
            metadata=self.base_policy.metadata,
        )
        server.serve_forever()


def _maybe_configure_cuda(args: Args) -> None:
    if not torch.cuda.is_available():
        return

    try:
        frac = max(0.0, min(1.0, args.max_cuda_mem_fraction))
        for dev_idx in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
        logging.info("Set CUDA memory fraction to %.2f for %d device(s)", frac, torch.cuda.device_count())
    except Exception as e:  # pragma: no cover - defensive
        logging.warning("Failed to set CUDA memory fraction: %s", e)


def main(args: Args) -> None:
    _maybe_configure_cuda(args)
    PolicyServerApp(args).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
