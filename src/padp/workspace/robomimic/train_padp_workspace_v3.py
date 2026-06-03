if __name__ == "__main__":
    import sys
    import os
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
import json
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import time
from termcolor import cprint
import tqdm
import numpy as np
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from padp.env_runner.base_image_runner import BaseImageRunner

from padp.workspace.base_workspace import BaseWorkspace
from padp.dataset.base_dataset import BaseImageDataset
from padp.common.checkpoint_util import TopKCheckpointManager
from padp.common.json_logger import JsonLogger
from padp.common.pytorch_util import dict_apply, optimizer_to
from padp.model.diffusion.ema_model import EMAModel
from padp.model.common.lr_scheduler import get_scheduler
from padp.workspace.workspace_util import freeze_encoder_modules

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'lr_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0
        self.lr_step = 0        # 策略新增，区分预热和非预热阶段的步数

        # 新增：加载预训练模型（其他配置都不变）
        if cfg.training.get('pretrained_model_path', None) is not None:
            pretrained_path = pathlib.Path(cfg.training.pretrained_model_path)
            if pretrained_path.is_file():
                print()
                print(f"正在从 pretrained model 加载权重: {pretrained_path}")
                ckpt = torch.load(pretrained_path, map_location='cpu')
                # 支持两种格式：完整 workspace 格式 或 纯 state_dict 格式
                if 'state_dicts' in ckpt and 'model' in ckpt['state_dicts']:
                    self.model.load_state_dict(ckpt['state_dicts']['model'])
                    if self.ema_model is not None and 'ema_model' in ckpt['state_dicts']:
                        self.ema_model.load_state_dict(ckpt['state_dicts']['ema_model'])
                else:
                    self.model.load_state_dict(ckpt)
                print("Pretrained model 加载成功。")
                print()
            else:
                print(f"Pretrained model 路径不存在: {pretrained_path}，将从零开始训练。")

    def run_eval_2(self, checkpoint: str, output_dir: str, device: str = None,
                   split: str = 'test', batch_size: int = None, max_batches: int = None):
        """
        Eval mode 2: iterate dataset (test/val) and compute per-position MSE across whole dataset.
        Writes per_pos_mse.npy and per_pos_mse.json into output_dir.
        Parameters:
            checkpoint: path to checkpoint file (will be loaded if provided)
            output_dir: directory to save results
            device: device string like 'cuda:0' or 'cpu'
            split: 'test' or 'val'
            batch_size: override val_dataloader batch size if provided
            max_batches: if set, only process this many batches (useful for smoke-test)
        """
        # prepare output dir
        os.makedirs(output_dir, exist_ok=True)

        # load checkpoint payload if provided
        if checkpoint is not None:
            print(f"Loading checkpoint from {checkpoint}")
            payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
            try:
                # try to load model weights etc into this workspace
                self.load_payload(payload)
            except Exception as e:
                print(f"Warning: load_payload failed: {e}")

        cfg = self.cfg

        # dataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert hasattr(dataset, 'get_validation_dataset')
        if (split == 'test') and hasattr(dataset, 'get_test_dataset'):
            eval_dataset = dataset.get_test_dataset()
        else:
            eval_dataset = dataset.get_validation_dataset()

        # dataloader config
        val_loader_cfg = dict(cfg.val_dataloader)
        if batch_size is not None:
            val_loader_cfg['batch_size'] = int(batch_size)
        eval_dataloader = DataLoader(eval_dataset, **val_loader_cfg)

        # set normalizer to model(s)
        normalizer = dataset.get_normalizer()
        try:
            self.model.set_normalizer(normalizer)
            if self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)
        except Exception:
            pass

        # pick policy (use ema if configured)
        policy = self.model
        if cfg.training.use_ema and (self.ema_model is not None):
            policy = self.ema_model

        # device
        if device is None:
            device = cfg.training.device if 'training' in cfg and 'device' in cfg.training else 'cpu'
        device = torch.device(device)
        policy.to(device)
        policy.eval()

        H = int(getattr(policy, 'horizon', getattr(self, 'horizon', None) or 0))
        if H == 0:
            # try cfg
            H = int(cfg.task.horizon) if 'task' in cfg and 'horizon' in cfg.task else None
        assert H is not None and H > 0, f"Cannot determine horizon H (got {H})"

        import numpy as _np
        from padp.common.pytorch_util import dict_apply as _dict_apply

        sum_sqerr = _np.zeros(H, dtype=_np.float64)
        sum_gt_abs = _np.zeros(H, dtype=_np.float64)
        sum_gt_l2 = _np.zeros(H, dtype=_np.float64)
        sum_gt_power = _np.zeros(H, dtype=_np.float64)
        total_count = 0
        skipped_batches = 0
        eval_start_time = time.time()
        progress_every = 500
        total_samples_hint = len(eval_dataset) if hasattr(eval_dataset, '__len__') else None

        # track initialization (warmup) counts and timing
        init_count = 0
        total_init_time = 0.0
        reset_reason_counts = {
            'first_sample': 0,
            'episode_change': 0,
            'non_contiguous_step': 0,
            'explicit_step_zero': 0,
            'metadata_missing_fallback': 0,
        }
        prev_episode_id = None
        prev_local_pos = None

        # PADP eval_2 专用：优先直接按 episode 顺序从 replay_buffer 读取，
        # 不依赖数据集内部已存 mapping 的全局顺序。
        use_direct_episode_loading = all([
            hasattr(eval_dataset, 'replay_buffer'),
            hasattr(eval_dataset, 'episode_map'),
            hasattr(eval_dataset, 'rgb_keys'),
            hasattr(eval_dataset, 'lowdim_keys'),
            hasattr(eval_dataset, 'n_obs_steps'),
        ])
        direct_episode_ids = _np.array([], dtype=_np.int64)
        if use_direct_episode_loading:
            if hasattr(eval_dataset, 'train_episode_ids') and (eval_dataset.train_episode_ids is not None):
                direct_episode_ids = _np.array(eval_dataset.train_episode_ids, dtype=_np.int64)
            elif hasattr(eval_dataset, 'train_mask'):
                direct_episode_ids = _np.nonzero(eval_dataset.train_mask)[0].astype(_np.int64)
            else:
                # fallback: use all episodes
                direct_episode_ids = _np.arange(eval_dataset.replay_buffer.n_episodes, dtype=_np.int64)

            if direct_episode_ids.size == 0:
                use_direct_episode_loading = False
            else:
                direct_episode_ids = _np.sort(direct_episode_ids)
                try:
                    total_samples_hint = int(sum(int(eval_dataset.episode_map[int(e), 5]) for e in direct_episode_ids))
                except Exception:
                    pass

        with torch.no_grad():
            if use_direct_episode_loading:
                stop_eval = False
                S = int(getattr(eval_dataset, 'n_obs_steps', 1))
                rb = eval_dataset.replay_buffer
                ep_map = _np.asarray(eval_dataset.episode_map)
                for ep_idx, ep_id in enumerate(direct_episode_ids):
                    if stop_eval:
                        break

                    # 每个 episode 开始时重置一次推理状态
                    t0 = time.time()
                    try:
                        policy.reset()
                    except Exception:
                        pass
                    t1 = time.time()
                    init_count += 1
                    total_init_time += (t1 - t0)
                    if total_count == 0:
                        reset_reason_counts['first_sample'] += 1
                    else:
                        reset_reason_counts['episode_change'] += 1

                    info = ep_map[int(ep_id)]
                    buf_start_1b = int(info[3])
                    win_num = int(info[5])

                    prev_episode_id = int(ep_id)
                    prev_local_pos = None

                    for local_pos in range(win_num):
                        if (max_batches is not None) and (total_count >= int(max_batches)):
                            stop_eval = True
                            break

                        abs_start = (buf_start_1b - 1) + local_pos

                        # build one sample directly from replay buffer
                        obs_np = {}
                        for key in eval_dataset.rgb_keys:
                            seq = rb[key][abs_start: abs_start + S]
                            seq = _np.moveaxis(seq, -1, 1).astype(_np.float32) / 255.0
                            obs_np[key] = seq[None, ...]
                        for key in eval_dataset.lowdim_keys:
                            seq = rb[key][abs_start: abs_start + S].astype(_np.float32)
                            obs_np[key] = seq[None, ...]

                        gt_window = rb['action'][abs_start: abs_start + H].astype(_np.float32)
                        if gt_window.shape[0] < H:
                            skipped_batches += 1
                            continue

                        sample_obs = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in obs_np.items()}

                        out = policy.predict_action(sample_obs)
                        if 'action_pred' not in out:
                            skipped_batches += 1
                            continue

                        pred_full = out['action_pred'].detach().cpu().numpy()      # [1,H,D]
                        gt_sample = gt_window[None, :H, :]                          # [1,H,D]

                        per_pos_sqerr = ((pred_full - gt_sample) ** 2).mean(axis=-1)
                        per_pos_gt_abs = _np.abs(gt_sample).mean(axis=-1)
                        per_pos_gt_l2 = _np.linalg.norm(gt_sample, axis=-1)
                        per_pos_gt_power = (gt_sample ** 2).mean(axis=-1)

                        sum_sqerr += per_pos_sqerr.sum(axis=0)
                        sum_gt_abs += per_pos_gt_abs.sum(axis=0)
                        sum_gt_l2 += per_pos_gt_l2.sum(axis=0)
                        sum_gt_power += per_pos_gt_power.sum(axis=0)
                        total_count += 1
                        prev_local_pos = local_pos

                        if (total_count % progress_every) == 0:
                            elapsed = max(time.time() - eval_start_time, 1e-9)
                            sps = total_count / elapsed
                            if total_samples_hint is not None and total_samples_hint > 0:
                                remain = max(total_samples_hint - total_count, 0)
                                eta_sec = remain / max(sps, 1e-9)
                                print(f"[eval_2] processed={total_count}/{total_samples_hint}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s, eta={eta_sec:.1f}s")
                            else:
                                print(f"[eval_2] processed={total_count}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s")
            else:
                for batch_idx, batch in enumerate(eval_dataloader):
                    if (max_batches is not None) and (batch_idx >= int(max_batches)):
                        break

                    # handle window_info -> extract episode_id/local_pos for natural sequential stepping
                    episode_id_arr = None
                    local_pos_arr = None
                    if 'window_info' in batch:
                        window_info = batch.pop('window_info')
                        try:
                            wi_np = (window_info.numpy() if hasattr(window_info, 'numpy') else _np.array(window_info))
                            # supported formats:
                            # 1) SequenceSamplerRT window_info: [*, *, 5], col0=episode_id, col4=local_wid
                            # 2) PADP pos_info:            [*, *, 5], col3=episode_id, col2=episode_new_pos_id
                            if wi_np.ndim == 3 and wi_np.shape[-1] >= 5:
                                # detect PADP format by warmup flag at col0 (mostly 0/1) and 1-based buffer id at col4
                                col0 = wi_np[:, 0, 0]
                                col4 = wi_np[:, 0, 4]
                                is_padp_like = _np.all(_np.isin(col0, [0, 1])) and _np.all(col4 >= 1)
                                if is_padp_like:
                                    episode_id_arr = wi_np[:, 0, 3].astype(_np.int64)
                                    local_pos_arr = wi_np[:, 0, 2].astype(_np.int64)
                                else:
                                    episode_id_arr = wi_np[:, 0, 0].astype(_np.int64)
                                    local_pos_arr = wi_np[:, 0, 4].astype(_np.int64)
                            elif wi_np.ndim == 2 and wi_np.shape[-1] >= 5:
                                col0 = wi_np[:, 0]
                                col4 = wi_np[:, 4]
                                is_padp_like = _np.all(_np.isin(col0, [0, 1])) and _np.all(col4 >= 1)
                                if is_padp_like:
                                    episode_id_arr = wi_np[:, 3].astype(_np.int64)
                                    local_pos_arr = wi_np[:, 2].astype(_np.int64)
                                else:
                                    episode_id_arr = wi_np[:, 0].astype(_np.int64)
                                    local_pos_arr = wi_np[:, 4].astype(_np.int64)
                        except Exception:
                            episode_id_arr = None
                            local_pos_arr = None

                    # prepare obs; keep gt on cpu to avoid extra transfers
                    if 'obs' not in batch or 'action' not in batch:
                        print('Batch missing obs or action keys, skipping')
                        skipped_batches += 1
                        continue

                    obs_dict = batch['obs']
                    gt_actions = batch['action']

                    # ensure gt has enough horizon
                    if gt_actions.shape[1] < H:
                        print(f"Skipping batch {batch_idx} because gt length {gt_actions.shape[1]} < H {H}")
                        skipped_batches += 1
                        continue

                    # move obs to device
                    obs_t = _dict_apply(obs_dict, lambda x: x.to(device, non_blocking=True))

                    # keep gt on cpu for numpy ops
                    gt_np = (gt_actions.numpy() if hasattr(gt_actions, 'numpy') else _np.array(gt_actions))[:, :H, :]

                    B = gt_np.shape[0]

                    # iterate samples individually to respect per-episode initialization
                    for i in range(B):
                        # determine whether to reset for natural episode stepping
                        cur_episode_id = None
                        cur_local_pos = None
                        if episode_id_arr is not None:
                            try:
                                cur_episode_id = int(episode_id_arr[i])
                            except Exception:
                                cur_episode_id = None
                        if local_pos_arr is not None:
                            try:
                                cur_local_pos = int(local_pos_arr[i])
                            except Exception:
                                cur_local_pos = None

                        need_reset = False
                        reset_reason = None
                        if total_count == 0:
                            need_reset = True
                            reset_reason = 'first_sample'
                        elif cur_episode_id is not None:
                            # episode boundary reset
                            if (prev_episode_id is None) or (cur_episode_id != prev_episode_id):
                                need_reset = True
                                reset_reason = 'episode_change'
                            # same episode but non-contiguous local step
                            elif (cur_local_pos is not None) and (prev_local_pos is not None):
                                if cur_local_pos == 0:
                                    need_reset = True
                                    reset_reason = 'explicit_step_zero'
                                elif cur_local_pos != (prev_local_pos + 1):
                                    need_reset = True
                                    reset_reason = 'non_contiguous_step'
                        elif cur_local_pos is not None:
                            # fallback when only local step is available
                            if cur_local_pos == 0:
                                need_reset = True
                                reset_reason = 'explicit_step_zero'
                            elif (prev_local_pos is not None) and (cur_local_pos != (prev_local_pos + 1)):
                                need_reset = True
                                reset_reason = 'non_contiguous_step'
                        else:
                            # no metadata: conservative fallback
                            need_reset = True
                            reset_reason = 'metadata_missing_fallback'

                        if need_reset:
                            t0 = time.time()
                            try:
                                policy.reset()
                            except Exception:
                                # best-effort: if policy has no reset, ignore
                                pass
                            t1 = time.time()
                            init_count += 1
                            total_init_time += (t1 - t0)
                            if reset_reason in reset_reason_counts:
                                reset_reason_counts[reset_reason] += 1

                        prev_episode_id = cur_episode_id if cur_episode_id is not None else prev_episode_id
                        prev_local_pos = cur_local_pos if cur_local_pos is not None else prev_local_pos

                        # build per-sample obs (keep batch dim = 1)
                        sample_obs = {}
                        for k, v in obs_t.items():
                            try:
                                sample_obs[k] = v[i:i+1]
                            except Exception:
                                sample_obs[k] = v[i:i+1]

                        out = policy.predict_action(sample_obs)
                        if 'action_pred' not in out:
                            print('policy.predict_action did not return action_pred for sample, skipping')
                            skipped_batches += 1
                            continue

                        pred_full = out['action_pred'].detach().cpu().numpy()  # [1,H,D]
                        gt_sample = gt_np[i:i+1, :H, :]

                        # compute per-sample per-position squared error (mean over action dim)
                        per_pos_sqerr = ((pred_full - gt_sample) ** 2).mean(axis=-1)  # [1,H]
                        # 统计真实动作每个位置的幅度（便于分析 MSE 是否受目标尺度影响）
                        per_pos_gt_abs = _np.abs(gt_sample).mean(axis=-1)             # [1,H]
                        per_pos_gt_l2 = _np.linalg.norm(gt_sample, axis=-1)           # [1,H]
                        # 归一化分母：每个位置真实动作的均方能量 E[a^2]
                        per_pos_gt_power = (gt_sample ** 2).mean(axis=-1)             # [1,H]

                        # accumulate
                        sum_sqerr += per_pos_sqerr.sum(axis=0)
                        sum_gt_abs += per_pos_gt_abs.sum(axis=0)
                        sum_gt_l2 += per_pos_gt_l2.sum(axis=0)
                        sum_gt_power += per_pos_gt_power.sum(axis=0)
                        total_count += 1

                        if (total_count % progress_every) == 0:
                            elapsed = max(time.time() - eval_start_time, 1e-9)
                            sps = total_count / elapsed
                            if total_samples_hint is not None and total_samples_hint > 0:
                                remain = max(total_samples_hint - total_count, 0)
                                eta_sec = remain / max(sps, 1e-9)
                                print(f"[eval_2] processed={total_count}/{total_samples_hint}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s, eta={eta_sec:.1f}s")
                            else:
                                print(f"[eval_2] processed={total_count}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s")

        if total_count == 0:
            print('No samples were processed. Exiting without writing results.')
            return {'per_pos_mse': None, 'count': 0}

        mean_mse_per_pos = (sum_sqerr / float(total_count)).astype(float)
        mean_gt_abs_per_pos = (sum_gt_abs / float(total_count)).astype(float)
        mean_gt_l2_per_pos = (sum_gt_l2 / float(total_count)).astype(float)
        mean_gt_power_per_pos = (sum_gt_power / float(total_count)).astype(float)
        nmse_eps = 1e-12
        per_pos_nmse = (mean_mse_per_pos / (mean_gt_power_per_pos + nmse_eps)).astype(float)

        # save results
        out_np_path = os.path.join(output_dir, 'per_pos_mse.npy')
        out_json_path = os.path.join(output_dir, 'per_pos_mse.json')
        _np.save(out_np_path, mean_mse_per_pos)
        # 保持原有格式，但添加位置索引信息
        result_dict = {
            'per_pos_mse': mean_mse_per_pos.tolist(),  # 保持原字段
            'per_pos_gt_abs': mean_gt_abs_per_pos.tolist(),
            'per_pos_gt_l2': mean_gt_l2_per_pos.tolist(),
            'per_pos_gt_power': mean_gt_power_per_pos.tolist(),
            'per_pos_nmse': per_pos_nmse.tolist(),
            'nmse_eps': nmse_eps,
            'count': int(total_count),
            'horizon': int(H),
            'time_steps': list(range(int(H)))  # 新增位置索引
        }

        json.dump(result_dict, open(out_json_path, 'w'), indent=2)

        # write eval log (initialization counts and timing)
        eval_log = {
            'init_count': int(init_count),
            'total_init_time': float(total_init_time),
            'avg_init_time': float(total_init_time / init_count) if init_count > 0 else 0.0,
            'reset_reason_counts': reset_reason_counts,
            'processed_samples': int(total_count),
            'skipped_batches': int(skipped_batches),
            'progress_every': int(progress_every),
            'eval_elapsed_sec': float(time.time() - eval_start_time),
            'eval_samples_per_sec': float(total_count / max(time.time() - eval_start_time, 1e-9)),
            'total_samples_hint': int(total_samples_hint) if total_samples_hint is not None else None
        }
        eval_log_path = os.path.join(output_dir, 'eval_2_log.json')
        json.dump(eval_log, open(eval_log_path, 'w'), indent=2)

        print(f"Wrote per-position MSE to {out_json_path} (n={int(total_count)})")
        return {
            'per_pos_mse': mean_mse_per_pos.tolist(),
            'per_pos_gt_abs': mean_gt_abs_per_pos.tolist(),
            'per_pos_gt_l2': mean_gt_l2_per_pos.tolist(),
            'per_pos_gt_power': mean_gt_power_per_pos.tolist(),
            'per_pos_nmse': per_pos_nmse.tolist(),
            'count': int(total_count)
        }

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print()
                print(f"正在从 checkpoint 恢复训练: {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                print()

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler（按 dataloader 的 batch 步进：每个 batch 只更新一次 LR）
        # num_training_steps 以 batch 为单位计算
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            # 使用 batch 级 lr_step 对齐 last_epoch
            last_epoch=self.lr_step-1    # last_epoch由self.global_step-1改为self.lr_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner: 'BaseImageRunner' = None  # deferred – needs padp.env_runner

        # 引入了 termcolor 库来在控制台中输出带颜色的日志信息，使得日志更加易于阅读
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        print("开始设置wandb，并Logging to wandb")
        # 创建 logging 配置的副本
        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **logging_config,  # 使用修改后的配置
        )
        wandb.config.update({
                "output_dir": self.output_dir,
        })

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # 创建独立的日志文件路径
        train_log_path = os.path.join(self.output_dir, 'train_logs.json.txt')
        val_log_path = os.path.join(self.output_dir, 'val_logs.json.txt')
        test_log_path = os.path.join(self.output_dir, 'test_logs.json.txt')
        with JsonLogger(train_log_path) as train_logger, \
             JsonLogger(val_log_path) as val_logger, \
             JsonLogger(test_log_path) as test_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = dict()
                # ========= train for this epoch ==========
                if getattr(cfg.training, 'freeze_encoder', False):
                    freeze_encoder_modules(self.model)

                # 在 GPU 上累加训练 loss，避免每个 batch 调用 batch_loss.item()
                # 触发 CUDA 同步；epoch 结束后只同步一次，数学均值不变。
                train_loss_sum = torch.zeros((), device=device)
                train_loss_count = 0

                # 部分1：每个 epoch 开始时随机指定一个 batch_idx 用于采样监控（如果 train_sampling_batch 还未设置）
                # # 在进入 dataloader 前，为当前 epoch 随机指定一个抽查的 batch_idx
                # self._target_sample_idx = random.randint(0, max(0, len(train_dataloader) - 1))

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # 先提取 window_info 等元数据，再将剩余的数据搬到 GPU
                        if 'window_info' in batch:
                            window_info = batch.pop('window_info')# 保留在 CPU，如需使用可单独读取，避免将 window_info 等元数据搬到 GPU
                            warm_up_flags = window_info[:, 0, 0]  # [B]
                        
                        # 搬到 GPU
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch    # 用于采样监控训练进度
                        
                        # 部分2：计算损失并反向传播；每 cfg.training.gradient_accumulate_every 步更新一次优化器和 lr_scheduler
                        # if batch_idx == getattr(self, '_target_sample_idx', -1) or train_sampling_batch is None:
                        #     # detach() 防止某些情况下计算图残留导致显存泄漏
                        #     train_sampling_batch = dict_apply(batch, lambda x: x.detach().clone())
                        
                        # 计算每个样本的损失
                        raw_loss = self.model.compute_loss(batch) # 计算每个样本的损失，返回 [B] 张量

                        # 计算整个批次的平均损失
                        batch_loss = raw_loss.mean()  # 标量
                        
                        loss = batch_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.lr_step += 1
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        # detach 只切断日志累加器的计算图，不改变前向、反向和优化器更新。
                        train_loss_sum += batch_loss.detach()
                        train_loss_count += 1
                        # 这里的 step_log 只含 Python 标量元信息，不读取 GPU tensor，
                        # 所以 wandb 逐步记录不会触发 GPU-CPU 同步。
                        step_log = {
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # train_loss 故意不在每 step 记录；否则需要 .item() 并阻塞 GPU。
                            # 最后一个 batch 的日志会和 validation / rollout / epoch 平均 loss 合并。
                            wandb_run.log(step_log, step=self.global_step)
                            step_log = {
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                        self.global_step += 1  # ✅ 移动到外层

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # 唯一一次 GPU-CPU 同步：得到本 epoch 的平均训练 loss 用于日志和 checkpoint。
                mean_loss = (train_loss_sum / train_loss_count).item()
                step_log['train_loss'] = mean_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()


# run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    # Lazy instantiation to avoid importing env_runner at module load time
                    if env_runner is None:
                        from padp.env_runner.base_image_runner import BaseImageRunner
                        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
                    if hasattr(env_runner, 'set_epoch'):
                        env_runner.set_epoch(self.epoch)
                    runner_log = env_runner.run(policy)
                    
                    # 新增：提取测试分数并记录到测试日志
                    test_score = runner_log.get('test/mean_score', None)  # 注意键名是 'test/mean_score'
                    if test_score is not None:
                        test_log_entry = {
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'test_mean_score': test_score
                        }
                        test_logger.log(test_log_entry)
                        print(f"记录测试分数: {test_score} (epoch {self.epoch})")

                    # log all
                    step_log.update(runner_log)
                    print("Rollout done.")


                """
                # ==========================================================================
                # [排障探针说明]：Validation (验证集开环评估)
                # - 默认状态：【已注释屏蔽】(追求极致训练速度，释放 100% 算力)
                # - 核心逻辑：当前 Checkpoint 保存已完全绑定仿真 Rollout 成功率 (test_mean_score)，
                #   因此纯数学上的验证集 Loss 已无参与选拔的必要。
                  
                # - 🚨 何时需要取消注释 (开启调试)：
                #   当你跑完实验，发现 Rollout 成功率一直是 0% 时。你需要知道是哪里崩了：
                #   1. 若取消注释后，发现 `val_loss` 很低 / 正常下降 -> 说明模型拟合能力没问题，
                #      成功率为 0 是因为仿真环境的 Covariate Shift (OOD) 或物理引擎配置出错。
                #   2. 若取消注释后，发现 `val_loss` 极高或 NaN -> 说明模型超参崩了，网络根本没有
                #      学会拟合人类动作，不要去查物理引擎的 Bug，先调学习率或网络结构。
                # ==========================================================================

                # run validation（仅统计平均验证损失，不做反归一化作图与逐位置MSE）
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        total_non_warmup_loss = torch.zeros((), device=device)
                        total_non_warmup_samples = 0
                        # 新增：按时间步统计 MSE（对 action 维求均值，再对样本求和）
                        # MSE 累加器保持在 GPU 上，避免每个 validation batch 都 .to('cpu') 阻塞。
                        # 只在整个验证循环结束后统一同步一次用于日志，不改变 MSE 计算公式。
                        val_sqerr_sum_per_t = None
                        val_mse_total_samples = 0
                        mean_mse_per_t = None
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                warm_up_flags = None
                                if 'window_info' in batch:
                                    window_info = batch.pop('window_info')# 保留在 CPU，如需使用可单独读取，避免将 window_info 等元数据搬到 GPU
                                    # 提取每个序列的第一个时间步的 warm_up_flag
                                    warm_up_flags = window_info[:, 0, 0]  # [B]

                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                realtime_losses = self.model.compute_loss(batch)  # [B]

                                # 逐时间步 MSE: (pred[:, t, :] - gt[:, t, :])^2
                                # 这里使用当前 eval policy（EMA 或在线模型）进行动作预测
                                try:
                                    # policy.reset()
                                    pred_result = policy.predict_action(batch['obs'])
                                    if 'action_pred' in pred_result:
                                        pred_action = pred_result['action_pred']  # [B,H,A]
                                        gt_action = batch['action']               # [B,H,A]

                                        cur_h = min(pred_action.shape[1], gt_action.shape[1])
                                        if cur_h > 0:
                                            sqerr = (pred_action[:, :cur_h, :] - gt_action[:, :cur_h, :]) ** 2
                                            # [B,H]，对 action 维取均值
                                            mse_per_t = sqerr.mean(dim=-1)
                                            # [H]，对 batch 维求和。保持在 GPU 上累加，避免逐 batch CPU 同步。
                                            mse_per_t_sum = mse_per_t.sum(dim=0).detach().to(dtype=torch.float64)

                                            if val_sqerr_sum_per_t is None:
                                                # 第一次按实际 horizon 初始化，device 与 mse_per_t_sum 保持一致。
                                                val_sqerr_sum_per_t = mse_per_t_sum.clone()
                                            else:
                                                # 防御性处理：若 horizon 不一致，按最小长度对齐
                                                if val_sqerr_sum_per_t.shape[0] != mse_per_t_sum.shape[0]:
                                                    min_h = min(val_sqerr_sum_per_t.shape[0], mse_per_t_sum.shape[0])
                                                    val_sqerr_sum_per_t = val_sqerr_sum_per_t[:min_h]
                                                    mse_per_t_sum = mse_per_t_sum[:min_h]
                                                val_sqerr_sum_per_t += mse_per_t_sum

                                            val_mse_total_samples += int(pred_action.shape[0])
                                except Exception as e:
                                    print(f"Warning: failed to compute per-timestep val MSE at batch {batch_idx}: {e}")
                                
                                # 只记录非预热样本的损失，预热期间的损失会被自动滤除
                                # 计算非预热样本的损失
                                if warm_up_flags is not None:
                                    non_warmup_mask = (warm_up_flags == 0)
                                    non_warmup_losses = realtime_losses[non_warmup_mask]
                                else:
                                    non_warmup_losses = realtime_losses
                                
                                # 累加损失和样本数量
                                if len(non_warmup_losses) > 0:
                                    total_non_warmup_loss += non_warmup_losses.sum().detach()
                                    total_non_warmup_samples += len(non_warmup_losses)

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if total_non_warmup_samples > 0:
                            val_loss = (total_non_warmup_loss / total_non_warmup_samples).item()
                        else:
                            val_loss = 0.0
                            print("Warning: No non-warmup samples found in validation.")
                        
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss
                        # 记录逐时间步 MSE 到 WandB（通过 step_log 统一上报）
                        if (val_sqerr_sum_per_t is not None) and (val_mse_total_samples > 0):
                            # 唯一一次 MSE 指标同步：验证结束后再搬到 CPU，供 numpy / wandb 日志使用。
                            mean_mse_per_t = (val_sqerr_sum_per_t / float(val_mse_total_samples)).cpu().numpy()
                            for t_idx, t_mse in enumerate(mean_mse_per_t.tolist()):
                                step_log[f'val_mse_t{t_idx}'] = float(t_mse)

                        # 创建专门的验证日志条目
                        val_log_entry = {
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'val_loss': val_loss
                        }
                        if mean_mse_per_t is not None:
                            val_log_entry['val_mse_per_t'] = mean_mse_per_t.tolist()
                        val_logger.log(val_log_entry)  # 记录到验证日志
                print("Validation done.")

                
                # ==========================================================================
                # [排障探针说明]：Train Sampling (训练集闭环去噪监控)
                # - 默认状态：【已注释屏蔽】(追求极致训练速度)
                # - 核心逻辑：`train_loss` 监控的是“预测加噪误差”，而此模块是让模型在熟悉的
                #   训练集上执行几十步完整的“闭环去噪生成”，监控最终动作的 action_mse。
                  
                # - 🚨 何时需要取消注释 (开启调试)：
                #   当你发现 `val_loss` 正常下降，但 Rollout 成功率依旧极差时。
                #   开启此探针，让模型在最熟悉的训练集数据上走一遍完整的去噪推理：
                #   1. 若 `train_action_mse_error` 很低 -> 扩散去噪过程本身没问题，纯粹是泛化失败。
                #   2. 若 `train_action_mse_error` 很高 -> 扩散模型的生成过程坏了（比如 PADP 的
                #      噪声调度表错误、滑动窗口预热机制失效等）。
                     
                # ⚠️ 警告：取消本模块注释时，务必同步取消内部 `policy.reset()` 的注释！
                # ==========================================================================


                # 在每个 epoch 结束时进行额外的采样、误差计算
                # 对训练集采样的第一个batch，使用当前 policy 进行动作预测，并与真实动作（gt_action）计算均方误差（MSE），将误差记录到日志（step_log['train_action_mse_error']）。
                # 监控模型在训练集上的动作预测效果。
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        policy.reset() # 🚨 注意：如果模型有 reset 机制，务必在采样前调用，否则可能导致采样状态异常，误差过大。
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                        # torch.cuda.empty_cache()
                    """
                
                # checkpoint save
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                train_logger.log(step_log)  # 记录完整的step_log到训练日志

                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

