"""
Multi-Agent PPO Training for Traffic Signal Control
Each intersection is controlled by a separate PPO agent. The environment supports multi-discrete actions.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.utils.config import CONFIG
from src.utils.logger import get_logger
from src.sim.env import CustomSUTrafficEnv

# Number of agents = number of traffic lights
NUM_AGENTS = len(CONFIG.get("traffic_light_ids", ["center"]))
TRAIN_TIMESTEPS = CONFIG.get("train_timesteps", 50000)
MODEL_DIR = CONFIG.get("model_save_path", "models/")
LOG_DIR = CONFIG.get("log_dir", "logs/")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = get_logger()

class MultiAgentEnvWrapper:
    """
    Wraps CustomSUTrafficEnv for multi-agent training.
    Each agent gets its own observation and reward, but acts in the shared environment.
    """
    def __init__(self):
        self.env = CustomSUTrafficEnv(nogui=True)
        self.num_agents = NUM_AGENTS

    def reset(self):
        obs, info = self.env.reset()
        # Split observation for each agent (if needed)
        return obs, info

    def step(self, actions):
        # actions: list of actions, one per agent
        obs, reward, terminated, truncated, info = self.env.step(actions)
        # Split reward for each agent (if needed)
        if isinstance(reward, (list, tuple, np.ndarray)):
            rewards = reward
        else:
            rewards = [reward] * self.num_agents
        return obs, rewards, terminated, truncated, info

    def close(self):
        self.env.close()

# Create one PPO agent per intersection
agents = [PPO('MlpPolicy', make_vec_env(lambda: CustomSUTrafficEnv(nogui=True), n_envs=1), verbose=0) for _ in range(NUM_AGENTS)]

# Training loop
for step in range(TRAIN_TIMESTEPS):
    # Reset environment and get initial obs
    env_wrapper = MultiAgentEnvWrapper()
    obs, info = env_wrapper.reset()
    done = False
    while not done:
        actions = []
        for i, agent in enumerate(agents):
            # Each agent selects its action
            action, _ = agent.predict(obs, deterministic=True)
            actions.append(action[0] if isinstance(action, (list, np.ndarray)) else action)
        # Step environment with joint actions
        obs, rewards, terminated, truncated, info = env_wrapper.step(actions)
        # Each agent learns from its own reward
        for i, agent in enumerate(agents):
            agent.learn(total_timesteps=1)
        done = terminated or truncated
    env_wrapper.close()
    if (step + 1) % 1000 == 0:
        logger.info(f"MA-PPO training step {step+1}/{TRAIN_TIMESTEPS}")
        # Save models
        for i, agent in enumerate(agents):
            agent.save(os.path.join(MODEL_DIR, f"ppo_agent_{i}.zip"))

logger.info("Multi-Agent PPO training complete.")
