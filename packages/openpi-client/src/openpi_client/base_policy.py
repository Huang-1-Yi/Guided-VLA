import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """根据观测推理动作。"""

    def reset(self) -> None:
        """将 policy 重置到初始状态。"""
        pass
