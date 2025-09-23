"""
CustomSUTrafficEnv: Minimal Gym environment for SUMO traffic light control.
Implements RL interface for PPO training. Extend for more signals or richer state.
"""

import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from src.utils.config import CONFIG
from src.utils.logger import get_logger

class CustomSUTrafficEnv(gym.Env):
	"""
	Minimal SUMO traffic light RL environment.
	Action: Discrete(2) (0=NS-green/EW-red, 1=EW-green/NS-red)
	Observation: Lane vehicle counts (normalized)
	Reward: Negative sum of lane waiting times (minimize congestion)
	"""
	metadata = {"render.modes": ["human"]}

	def __init__(self, nogui=True, use_cv=False):
		super().__init__()
		self.logger = get_logger()
		self.nogui = nogui
		self.use_cv = use_cv
		self.sumo_cfg = CONFIG.get("sumo_cfg_file", "data/net/simple_net.sumocfg")
		self.sim_steps = CONFIG.get("simulation_steps", 100)
		self.current_step = 0
		self.tl_ids = CONFIG.get("traffic_light_ids", ["center"])  # list of traffic light ids
		self.lane_ids = CONFIG.get("lane_ids", ["north2center_0", "south2center_0", "east2center_0", "west2center_0"])  # all incoming lanes
		self._start_sumo()

		# Action: one phase per traffic light (0=NS-green/EW-red, 1=EW-green/NS-red)
		self.action_space = spaces.MultiDiscrete([2] * len(self.tl_ids))
		# Observation: vehicle count per lane (normalized to [0,1]), plus CV count if enabled
		obs_dim = len(self.lane_ids) + (1 if self.use_cv else 0)
		self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

	def _start_sumo(self):
		"""Start SUMO (headless or GUI) and connect via traci."""
		if traci.isLoaded():
			traci.close()
		sumo_binary = "sumo"
		if not self.nogui:
			from shutil import which
			if which("sumo-gui"):
				sumo_binary = "sumo-gui"
		sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--step-length", "1"]
		traci.start(sumo_cmd)
		self.current_step = 0

	def reset(self, *, seed=None, options=None):
		"""Reload SUMO and return initial observation (lane vehicle counts)."""
		if traci.isLoaded():
			traci.close()
		self._start_sumo()
		self.current_step = 0
		obs = self._get_obs()
		info = {}
		return obs, info

	def _get_obs(self):
		"""Get normalized vehicle count for each incoming lane, plus CV count if enabled."""
		obs = []
		for lane in self.lane_ids:
			count = traci.lane.getLastStepVehicleNumber(lane)
			if isinstance(count, tuple):
				count = count[0]
			count = float(count)
			obs.append(min(count / 10.0, 1.0))
		# Optionally add live vehicle count from CV
		if self.use_cv:
			cv_count = self._get_cv_vehicle_count()
			obs.append(min(cv_count / 20.0, 1.0))  # Normalize by max 20 vehicles
		return np.array(obs, dtype=np.float32)

	def _get_cv_vehicle_count(self):
		"""Read latest vehicle count from logs/live_vehicle_counts.txt (YOLOv8 output)."""
		log_path = os.path.join("logs", "live_vehicle_counts.txt")
		if not os.path.exists(log_path):
			return 0.0
		try:
			with open(log_path, "r") as f:
				last_line = f.readlines()[-1]
			parts = last_line.strip().split(",")
			if len(parts) >= 2:
				return float(parts[1])
		except Exception:
			return 0.0
		return 0.0

	def step(self, action):
		"""
		Set traffic light phases for all intersections, advance SUMO, compute reward.
		Enhanced reward: aggregate metrics across all intersections.
		"""
		# Set phase for each traffic light
		if not isinstance(action, (list, np.ndarray)):
			action = [action] * len(self.tl_ids)
		for idx, tl in enumerate(self.tl_ids):
			traci.trafficlight.setPhase(tl, int(action[idx]))
		traci.simulationStep()
		self.current_step += 1

		# Aggregate metrics for all lanes
		total_waiting = 0.0
		total_queue = 0.0
		for lane in self.lane_ids:
			waiting = traci.lane.getWaitingTime(lane)
			queue = traci.lane.getLastStepHaltingNumber(lane)
			if isinstance(waiting, tuple):
				waiting = waiting[0]
			if isinstance(queue, tuple):
				queue = queue[0]
			total_waiting += float(waiting)
			total_queue += float(queue)
		arrived = len(traci.simulation.getArrivedIDList())

		# Refined reward function (example: penalize congestion, prioritize throughput)
		reward = -0.7 * total_waiting - 0.2 * total_queue + 0.1 * arrived

		info = {
			"step": self.current_step,
			"waiting_time": total_waiting,
			"queue_length": total_queue,
			"arrived": arrived,
			"reward": reward
		}

		obs = self._get_obs()
		terminated = self.current_step >= self.sim_steps
		truncated = False
		return obs, reward, terminated, truncated, info

	def close(self):
		"""Close SUMO connection."""
		if traci.isLoaded():
			traci.close()

	# Optionally add render() or seed() if needed for Gym compatibility
