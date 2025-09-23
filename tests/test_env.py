"""
Unit tests for CustomSUTrafficEnv (RL environment)
"""
import pytest
from src.sim.env import CustomSUTrafficEnv

def test_env_reset():
    env = CustomSUTrafficEnv(nogui=True)
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, (list, tuple, type(env.observation_space.sample())))
    env.close()

def test_env_step():
    env = CustomSUTrafficEnv(nogui=True)
    obs, info = env.reset()
    action = env.action_space.sample()
    step_result = env.step(action)
    assert len(step_result) == 5  # obs, reward, terminated, truncated, info
    env.close()
