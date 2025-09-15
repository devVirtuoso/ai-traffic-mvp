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

	def __init__(self, nogui=True):
		super().__init__()
		self.logger = get_logger()
		self.nogui = nogui
		self.sumo_cfg = CONFIG.get("sumo_cfg_file", "data/net/simple_net.sumocfg")
		self.sim_steps = CONFIG.get("simulation_steps", 100)
		self.current_step = 0
		self.tl_id = CONFIG.get("traffic_light_id", "center")  # traffic light id in net
		self.lane_ids = CONFIG.get("lane_ids", ["north2center_0", "south2center_0", "east2center_0", "west2center_0"])  # incoming lanes
		self._start_sumo()

		# Action: 0 or 1 (2 phases)
		self.action_space = spaces.Discrete(2)
		# Observation: vehicle count per lane (normalized to [0,1])
		self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.lane_ids),), dtype=np.float32)

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
		"""Get normalized vehicle count for each incoming lane."""
		obs = []
		for lane in self.lane_ids:
			count = traci.lane.getLastStepVehicleNumber(lane)
			# Normalize by a reasonable max (e.g., 10 vehicles)
			obs.append(min(count / 10.0, 1.0))
		return np.array(obs, dtype=np.float32)

	def step(self, action):
		"""
		Set traffic light phase, advance SUMO, compute reward.
		Reward = -sum of lane waiting times (minimize congestion)
		"""
		# Set phase: 0=NS-green/EW-red, 1=EW-green/NS-red
		traci.trafficlight.setPhase(self.tl_id, int(action))
		traci.simulationStep()
		self.current_step += 1

		# Compute reward: negative sum of waiting times on all incoming lanes
		reward = 0.0
		for lane in self.lane_ids:
			reward -= traci.lane.getWaitingTime(lane)

		obs = self._get_obs()
		terminated = self.current_step >= self.sim_steps
		truncated = False
		info = {}
		return obs, reward, terminated, truncated, info

	def close(self):
		"""Close SUMO connection."""
		if traci.isLoaded():
			traci.close()

	# Optionally add render() or seed() if needed for Gym compatibility
