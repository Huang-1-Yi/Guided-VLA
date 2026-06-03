import abc


class Environment(abc.ABC):
    """Environment 表示机器人及其所处环境。

    Environment 的主要契约是：可以查询其状态观测，也可以向其施加动作以改变状态。
    """

    @abc.abstractmethod
    def reset(self) -> None:
        """将环境重置到初始状态。

        每个 episode 开始前会调用一次。
        """

    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        """允许环境发出 episode 已结束的信号。

        每一步之后都会调用。如果 episode 已结束（无论成功或失败）应返回 `True`，
        否则返回 `False`。
        """

    @abc.abstractmethod
    def get_observation(self) -> dict:
        """查询环境当前状态。"""

    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        """在环境中执行一个动作。"""
