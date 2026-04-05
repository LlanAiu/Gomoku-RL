# builtin
import csv
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

# external
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# internal


class Logger:
	_instance = None
	_lock = threading.Lock()

	def __new__(cls):
		with cls._lock:
			if cls._instance is None:
				cls._instance = super().__new__(cls)
		return cls._instance

	@classmethod
	def get_instance(cls, save_dir: str = "logs"):
		inst = cls()
		if not hasattr(inst, "_initialized") or not inst._initialized:
			inst._init(save_dir=save_dir)
		return inst

	def _init(self, save_dir: str = "logs"):
		self._save_dir = os.path.abspath(save_dir)
		os.makedirs(self._save_dir, exist_ok=True)
  
		self._data: Dict[str, List] = defaultdict(list)
  
		self._global_step = 0
		self._data_lock = threading.Lock()
		self._initialized = True

	def log_scalar(self, name: str, value: float,
				   *, step: Optional[int] = None,
				   episode: Optional[int] = None,
				   timestep: Optional[int] = None) -> None:
		with self._data_lock:
			if step is None:
				step = self._global_step
				self._global_step += 1
			self._data[name].append((None if episode is None else int(episode),
									  None if timestep is None else int(timestep),
									  int(step),
									  float(value)))

	def log_dict(self, metrics: Dict[str, float], *, step: Optional[int] = None,
				 episode: Optional[int] = None, timestep: Optional[int] = None) -> None:
		with self._data_lock:
			if step is None:
				step = self._global_step
				self._global_step += 1
			for k, v in metrics.items():
				self._data[k].append((None if episode is None else int(episode),
									  None if timestep is None else int(timestep),
									  int(step),
									  float(v)))

	def save_csv(self) -> List[str]:
		files = []
		with self._data_lock:
			for metric, rows in self._data.items():
				safe_name = metric.replace("/", "_")
				path = os.path.join(self._save_dir, f"{safe_name}.csv")
				with open(path, "w", newline="", encoding="utf-8") as f:
					writer = csv.writer(f)
					writer.writerow(["episode", "timestep", "step", "value"])
					for ep, ts, step, val in rows:
						writer.writerow(["" if ep is None else ep,
										 "" if ts is None else ts,
										 step,
										 val])
				files.append(path)
		return files

	def _get_series(self, metric: str, episode_filter: Optional[int] = None):
		with self._data_lock:
			rows = self._data.get(metric, [])
			if not rows:
				return np.array([]), np.array([])

			if episode_filter is not None:
				rows = [r for r in rows if r[0] == episode_filter]
			if not rows:
				return np.array([]), np.array([])
			_, _, steps, values = zip(*rows)
   	
			x = np.array(steps)
   
			return x, np.array(values)

	def plot(self, metrics: Optional[List[str]] = None, rolling_window: int = 1, episode: Optional[int] = None) -> List[str]:
		with self._data_lock:
			available = list(self._data.keys())
		if metrics is None:
			metrics = available
		timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		out_files = []

		for m in metrics:
			steps, vals = self._get_series(m, episode_filter=episode)
			if steps.size == 0:
				continue
			plt.figure(figsize=(8, 4))
			if rolling_window > 1 and vals.size >= rolling_window:
				window = np.ones(rolling_window) / rolling_window
				smooth = np.convolve(vals, window, mode="valid")
				steps_smooth = steps[rolling_window - 1:]
				plt.plot(steps_smooth, smooth, label=f"{m} (smoothed)")
				plt.plot(steps, vals, alpha=0.25, label=f"{m} (raw)")
			else:
				plt.plot(steps, vals, label=m)
			plt.xlabel("step")
			plt.ylabel(m)
			plt.title(m)
			plt.legend()
			indiv_path = os.path.join(self._save_dir, f"{m.replace('/', '_')}_{timestamp}.png")
			plt.tight_layout()
			plt.savefig(indiv_path)
			plt.close()
			out_files.append(indiv_path)

		return out_files

	def clear(self) -> None:
		with self._data_lock:
			self._data.clear()
			self._global_step = 0

__all__ = ["Logger"]
