from typing import Optional, Dict
import os
import json
import time

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            # write train loss metadata next to logs.json.txt (parent of checkpoints dir)
            try:
                self._write_train_loss_file(ckpt_path, data)
            except Exception:
                pass
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)

            # update train loss metadata file
            try:
                self._write_train_loss_file(ckpt_path, data)
            except Exception:
                pass

            return ckpt_path

    def _write_train_loss_file(self, ckpt_path: str, data: Dict[str, float]):
        """
        Write or update a JSON file `train_loss.json.txt` in the parent
        directory of the checkpoints dir (this is the same directory as
        `logs.json.txt` used by the workspaces).

        The file contains a list of entries; each entry is a dict with at least
        `filename` and the keys from `data` sanitized to plain Python types.
        """
        # parent of checkpoints dir, typically the workspace output dir
        parent_dir = os.path.dirname(os.path.abspath(self.save_dir))
        if parent_dir == '':
            parent_dir = '.'

        # Prefer the directory that contains logs.json.txt (exact same dir as logs)
        def _find_logs_dir(start_dir: str) -> Optional[str]:
            cur = os.path.abspath(start_dir)
            prev = None
            # walk upwards until filesystem root
            while cur != prev:
                candidate = os.path.join(cur, 'logs.json.txt')
                if os.path.exists(candidate):
                    return cur
                prev = cur
                cur = os.path.dirname(cur)
            return None

        logs_dir = _find_logs_dir(parent_dir)
        target_dir = logs_dir if logs_dir is not None else parent_dir
        os.makedirs(target_dir, exist_ok=True)

        out_path = os.path.join(target_dir, 'train_loss.json.txt')

        # load existing entries if any
        entries = []
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            except Exception:
                # if file is corrupted or not json, overwrite
                entries = []

        # sanitize data values to JSON-serializable types
        sanitized = {}
        for k, v in data.items():
            try:
                # try native conversion
                if v is None:
                    sanitized[k] = None
                elif isinstance(v, (int, float, str, bool)):
                    sanitized[k] = v
                else:
                    # numpy types etc.
                    try:
                        sanitized[k] = float(v)
                        # convert to int when possible
                        if abs(sanitized[k] - int(sanitized[k])) < 1e-12:
                            sanitized[k] = int(sanitized[k])
                    except Exception:
                        sanitized[k] = str(v)
            except Exception:
                sanitized[k] = str(v)

        entry = {'filename': os.path.basename(ckpt_path), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
        entry.update(sanitized)

        # replace existing entry with same filename or append
        replaced = False
        for i, e in enumerate(entries):
            if isinstance(e, dict) and e.get('filename') == entry['filename']:
                entries[i] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)

        # atomic write
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, out_path)
