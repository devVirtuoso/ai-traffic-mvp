"""
Train a reinforcement learning agent (PPO) for traffic signal control using SUMO.
Customize parameters in src/utils/config.py.
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.utils.config import CONFIG
from src.utils.logger import get_logger
from src.sim.env import CustomSUTrafficEnv

def main():
	parser = argparse.ArgumentParser(description="Train PPO agent for SUMO traffic light control.")
	parser.add_argument('--nogui', action='store_true', help='Run SUMO in headless mode (no GUI)')
	args = parser.parse_args()

	logger = get_logger()
	model_path = CONFIG.get("model_save_path", "models/ppo_sumo.zip")
	log_dir = CONFIG.get("log_dir", "logs/")
	os.makedirs(log_dir, exist_ok=True)
	train_timesteps = CONFIG.get("train_timesteps", 50000)

	# Instantiate environment
	def make_env():
		return CustomSUTrafficEnv(nogui=args.nogui)
	env = DummyVecEnv([make_env])

	# Check for existing model
	if os.path.exists(model_path):
		logger.warning(f"Model already exists at {model_path}.")
		resp = input("Overwrite (o), Resume (r), or Abort (a)? [o/r/a]: ").strip().lower()
		if resp == 'a':
			logger.info("Aborting training.")
			return
		elif resp == 'r':
			logger.info("Resuming training from saved model.")
			model = PPO.load(model_path, env=env)
		else:
			logger.info("Overwriting previous model.")
			model = PPO('MlpPolicy', env, verbose=1)
	else:
		model = PPO('MlpPolicy', env, verbose=1)

	# Log rewards to file
	reward_log_path = os.path.join(log_dir, "episode_rewards.txt")
	episode_rewards = []

	try:
		obs = env.reset()
		for step in range(train_timesteps):
			action, _ = model.predict(obs, deterministic=True)
			obs, reward, done, info = env.step(action)
			episode_rewards.append(reward[0])
			if done[0]:
				obs = env.reset()
		# Save model after training
		model.save(model_path)
		logger.info(f"Model saved to {model_path}")
	except KeyboardInterrupt:
		logger.warning("Training interrupted by user. Saving model...")
		model.save(model_path)
		logger.info(f"Model saved to {model_path}")
		return

	# Write rewards to file
	with open(reward_log_path, "w") as f:
		for r in episode_rewards:
			f.write(f"{r}\n")
	logger.info(f"Episode rewards logged to {reward_log_path}")

	# Print mean reward
	mean_reward = np.mean(episode_rewards)
	logger.info(f"Mean reward over training: {mean_reward:.2f}")

	# Evaluate agent for 100 steps
	logger.info("Evaluating trained agent...")
	import traci
	eval_env = CustomSUTrafficEnv(nogui=True)
	obs, _ = eval_env.reset()
	eval_rewards = []
	waiting_times = []
	vehicle_count = 0
	total_travel_time = 0.0
	for _ in range(100):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = eval_env.step(action)
		eval_rewards.append(reward)
		# Collect average waiting time for reporting
		waiting = 0.0
		for lane in eval_env.lane_ids:
			waiting += traci.lane.getWaitingTime(lane)
		waiting_times.append(waiting / len(eval_env.lane_ids))
		# Count vehicles and travel time (approximate)
		vehicle_count += len(traci.simulation.getArrivedIDList())
		for veh_id in traci.simulation.getArrivedIDList():
			# This is a proxy; in a real setup, track entry/exit times
			total_travel_time += 1.0  # Placeholder: 1 step per vehicle
		if terminated or truncated:
			obs, _ = eval_env.reset()
	eval_env.close()
	logger.info(f"Evaluation mean reward: {np.mean(eval_rewards):.2f}")
	logger.info(f"Evaluation average waiting time: {np.mean(waiting_times):.2f} s")

	# Save RL evaluation results for dashboard (app.py)
	def save_rl_results_json():
		"""
		Save RL evaluation results to a JSON file for dashboard consumption (app.py).
		"""
		import json
		from datetime import datetime
		output = {
			"mean_reward": float(np.mean(eval_rewards)),
			"avg_waiting_time_per_step": float(np.mean(waiting_times)),
			"total_vehicles_completed": int(vehicle_count),
			"total_travel_time": float(total_travel_time),
			"training_timesteps": int(train_timesteps),
			"timestamp": datetime.now().isoformat()
		}
		log_dir = CONFIG.get("log_dir", "logs")
		os.makedirs(log_dir, exist_ok=True)
		out_path = os.path.join(log_dir, "rl_results.json")
		try:
			with open(out_path, "w") as f:
				json.dump(output, f, indent=2)
			logger.info(f"Saved RL evaluation results to {out_path}")
		except Exception as e:
			logger.warning(f"Could not save RL results to {out_path}: {e}")

	save_rl_results_json()

	logger.info("Training and evaluation complete.")

if __name__ == "__main__":
	main()
