"""Lightweight training logger with CSV export and plotting.

Usage:
  from modules.log.logger import Logger
  logger = Logger.get_instance(save_dir="./logs")
  logger.log_scalar("episode_reward", 1.23)
  logger.log_dict({"loss": 0.5, "accuracy": 0.8})
  logger.plot()  # saves plots to the save_dir

Install: pip install numpy matplotlib
"""

import csv
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Logger:

	_instance = None
	_lock = threading.Lock()

	def __new__(cls, *args, **kwargs):
		with cls._lock:
			if cls._instance is None:
				cls._instance = super().__new__(cls)
		return cls._instance

	@classmethod
	def get_instance(cls, save_dir: str = "logs", resume: bool = False):
		inst = cls()
		if not hasattr(inst, "_initialized") or not inst._initialized:
			inst._init(save_dir=save_dir, resume=resume)
		return inst

	def _init(self, save_dir: str = "logs", resume: bool = False):
		self.save_dir = os.path.abspath(save_dir)
		os.makedirs(self.save_dir, exist_ok=True)
		# data: metric -> list of (episode, timestep, step, value)
		# episode and timestep may be None; step is an internal global counter
		self._data: Dict[str, List] = defaultdict(list)
		# simple global step counter (increments on each log_scalar call without step)
		self._global_step = 0
		self._data_lock = threading.Lock()
		self._initialized = True

	def log_scalar(self, name: str, value: float,
				   *, step: Optional[int] = None,
				   episode: Optional[int] = None,
				   timestep: Optional[int] = None) -> None:
		"""Log a single scalar value under `name`.

		Prefer supplying `episode` and `timestep` for per-episode observations.
		If `step` is provided it will be used as the internal step value; if not
		provided an internal monotonically increasing step is used.
		"""
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
		"""Log multiple scalars at once. All receive the same `step`, `episode`,
		and `timestep` if provided."""
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
		"""Write one CSV file per metric. Returns list of written file paths.

		CSV columns: episode, timestep, step, value
		"""
		files = []
		with self._data_lock:
			for metric, rows in self._data.items():
				safe_name = metric.replace("/", "_")
				path = os.path.join(self.save_dir, f"{safe_name}.csv")
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
		"""Return (x, y) arrays for plotting.

		If any row has a `timestep` value, x will be the `timestep` value.
		Otherwise x will be the internal `step` value. If `episode_filter` is
		provided only rows matching that episode are returned.
		"""
		with self._data_lock:
			rows = self._data.get(metric, [])
			if not rows:
				return np.array([]), np.array([])
			# filter by episode if requested
			if episode_filter is not None:
				rows = [r for r in rows if r[0] == episode_filter]
			if not rows:
				return np.array([]), np.array([])
			episodes, timesteps, steps, values = zip(*rows)
			# choose x: use timestep when any timestep is not None
			use_timestep = any(ts is not None for ts in timesteps)
			if use_timestep:
				x = np.array([0 if ts is None else ts for ts in timesteps])
			else:
				x = np.array(steps)
			return x, np.array(values)

	def plot(self, metrics: Optional[List[str]] = None, rolling_window: int = 1,
			 save_name_prefix: str = "training", episode: Optional[int] = None) -> List[str]:
		"""Create and save plots. Returns list of saved plot file paths.

		- `metrics`: list of metric names to include. If None, all metrics are used.
		- `rolling_window`: smoothing window (1 = no smoothing).
		"""
		with self._data_lock:
			available = list(self._data.keys())
		if metrics is None:
			metrics = available
		timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		out_files = []

		# Combined plot
		plt.figure(figsize=(10, 6))
		plotted = 0
		for m in metrics:
			steps, vals = self._get_series(m, episode_filter=episode)
			if steps.size == 0:
				continue
			if rolling_window > 1 and vals.size >= rolling_window:
				window = np.ones(rolling_window) / rolling_window
				smooth = np.convolve(vals, window, mode="valid")
				# align steps for smoothed values
				steps_smooth = steps[rolling_window - 1:]
				plt.plot(steps_smooth, smooth, label=f"{m} (smoothed)")
			else:
				plt.plot(steps, vals, label=m)
			plotted += 1
		if plotted > 0:
			plt.xlabel("step")
			plt.ylabel("value")
			plt.title("Training metrics")
			plt.legend()
			combined_path = os.path.join(self.save_dir, f"{save_name_prefix}_combined_{timestamp}.png")
			plt.tight_layout()
			plt.savefig(combined_path)
			plt.close()
			out_files.append(combined_path)

		# Individual plots
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
			indiv_path = os.path.join(self.save_dir, f"{m.replace('/', '_')}_{timestamp}.png")
			plt.tight_layout()
			plt.savefig(indiv_path)
			plt.close()
			out_files.append(indiv_path)

		return out_files

	def clear(self) -> None:
		"""Clear all recorded data (keeps save_dir)."""
		with self._data_lock:
			self._data.clear()
			self._global_step = 0

	def flush(self) -> None:
		"""Alias to save CSVs to disk."""
		self.save_csv()


__all__ = ["Logger"]


