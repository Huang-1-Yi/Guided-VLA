import abc


class Agent(abc.ABC):
    """Agent 是具备决策能力的实体。

    Agent 接收关于世界状态的观测，并返回相应要执行的动作。
    """

    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        """向 agent 查询下一个动作。"""

    @abc.abstractmethod
    def reset(self) -> None:
        """将 agent 重置到初始状态。"""
