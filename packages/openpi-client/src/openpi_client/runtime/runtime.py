import logging
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class Runtime:
    """协调系统关键组件交互的核心模块。"""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0

    def run(self) -> None:
        """持续运行 runtime 循环，直到调用 stop() 或环境结束。"""
        for _ in range(self._num_episodes):
            self._run_episode()

        # 最后再 reset 一次，这对真实环境很重要，可以让机器人回到 home position。
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """在新线程中运行 runtime 循环。"""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """标记 episode 结束。"""
        self._in_episode = False

    def _run_episode(self) -> None:
        """运行单个 episode。"""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # 通过 sleep 维持期望帧率。
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """runtime 循环中的单步。"""
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
