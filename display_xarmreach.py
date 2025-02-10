import time
import os
import gym
import Gym_xarm
# Import the RL model
from stable_baselines3 import DDPG,PPO

save_dir = "/home/xarm/Documents/Gym_env/"
os.makedirs(save_dir, exist_ok=True)

config = {
    'GUI': True, 
    'reward_type':'dense_diff',
    'Sim2Real': False
}

env = gym.make('Gym_xarm/XarmReach-v0', config = config, render_mode="rgb_array") 
loaded_model = PPO.load(f"{save_dir}/saved_XarmReach_best")
# Check that the prediction is the same after loading (for the same observation)

# Start a new episode
obs = env.reset()
episode_reward = 0
for i in range(1000):
    # What action to take in state `obs`?
    # _states are only useful when using LSTM policies
    # `deterministic` is to use deterministic actions
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    # episode_reward += rewards
    if dones or info.get("is_success", False):
        # print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        # episode_reward = 0.0
        print("Success?", info.get("is_success", False))
        # break
        time.sleep(0.5)
        obs = env.reset()
    
env.close()