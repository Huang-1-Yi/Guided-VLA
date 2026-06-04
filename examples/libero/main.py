import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
from typing import Optional, List, Dict, Any

import datetime as dt
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    selected_task_ids: Optional[List[int]] = None
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # Results JSON output
    #################################################################################################################
    # Where to write the rolling results JSON. Configure via:
    #   --args.results_json_path path/to/results.json
    results_json_path: str = "data/libero/results.json"


def _atomic_write_json(obj: Dict[str, Any], path: pathlib.Path) -> None:
    """Write JSON atomically: write to tmp then replace to avoid partial files."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (important for atomic rename)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace: works across platforms
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            # First time: ensure directory exists then rename
            tmp_path.rename(path)
    except Exception as e:
        # Cleanup temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


def _init_results_file(args: Args, selected: List[int]) -> pathlib.Path:
    """Create or update the results file header (meta + empty buckets)."""
    path = pathlib.Path(args.results_json_path)
    now_iso = dt.datetime.now().isoformat()
    data: Dict[str, Any]
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("meta", {})
    meta = data["meta"]
    # Keep original created_at if present, otherwise set it.
    meta.setdefault("created_at", now_iso)
    # Always update these
    meta.update(
        {
            "updated_at": now_iso,
            "task_suite_name": args.task_suite_name,
            "selected_task_ids": selected,
            "host": args.host,
            "port": args.port,
            "resize_size": args.resize_size,
            "replan_steps": args.replan_steps,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "video_out_path": str(args.video_out_path),
        }
    )

    data.setdefault("success", [])
    data.setdefault("failure", [])
    data.setdefault(
        "running_counts",
        {"total_episodes": 0, "total_successes": 0, "success_rate": 0.0},
    )
    _atomic_write_json(data, path)
    return path


def _record_episode_result(
    args: Args,
    *,
    task_id: int,
    task_description: str,
    episode_index: int,
    steps_taken: int,
    success: bool,
    video_path: pathlib.Path,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one episode result and persist to disk immediately."""
    path = pathlib.Path(args.results_json_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {
            "meta": {},
            "success": [],
            "failure": [],
            "running_counts": {
                "total_episodes": 0,
                "total_successes": 0,
                "success_rate": 0.0,
            },
        }

    # Create episode record
    bucket = "success" if success else "failure"

    record: Dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode_index": int(episode_index),
        "steps_taken": int(steps_taken),
        "video": str(video_path),
    }
    if error:
        record["error"] = str(error)
    if extra:
        record["extra"] = extra

    data.setdefault(bucket, []).append(record)

    rc = data.setdefault(
        "running_counts",
        {"total_episodes": 0, "total_successes": 0, "success_rate": 0.0},
    )
    rc["total_episodes"] = int(rc.get("total_episodes", 0)) + 1
    if success:
        rc["total_successes"] = int(rc.get("total_successes", 0)) + 1
    total = max(1, rc["total_episodes"])
    rc["success_rate"] = float(rc["total_successes"]) / float(total)

    # touch meta.updated_at
    data.setdefault("meta", {})
    data["meta"]["updated_at"] = dt.datetime.now().isoformat()

    _atomic_write_json(data, path)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Generate timestamp for unique video filenames
    run_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    selected_task_ids = args.selected_task_ids if args.selected_task_ids is not None else list(range(num_tasks_in_suite))
    for task_id in selected_task_ids:
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(f"Task id {task_id} is out of range for {args.task_suite_name} with {num_tasks_in_suite} tasks.")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Initialize results file with all task IDs
    _init_results_file(args, selected_task_ids)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(selected_task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{args.task_suite_name}_task{task_id:03d}_{task_segment}_ep{episode_idx:02d}_{run_timestamp}_{suffix}.mp4"
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Record episode result to JSON
            _record_episode_result(
                args,
                task_id=task_id,
                task_description=task_description,
                episode_index=episode_idx,
                steps_taken=t,
                success=bool(done),
                video_path=video_path,
                error=None,
                extra={"max_steps": max_steps, "num_steps_wait": args.num_steps_wait},
            )

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
