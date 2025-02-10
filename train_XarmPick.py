import time
import os
import gym
import Gym_xarm_pick
from stable_baselines3 import DDPG, PPO

save_dir = "/home/xarm/Documents/Gym_env/"
os.makedirs(save_dir, exist_ok=True)

config = {
    'GUI': True, 
    'reward_type':'dense_diff',
    'Sim2Real': False
}

env = gym.make('Gym_xarm_pick/XarmPick-v0', config = config, render_mode="rgb_array") 

# Train an agent using PPO
# model = PPO("MultiInputPolicy", "Gym_xarm/XarmReach-v0").learn(total_timesteps=30000)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

model.save(f"{save_dir}/saved_XarmPick")

# sample an observation from the environment
obs = model.env.observation_space.sample()

# Check prediction before saving
print("pre saved", model.predict(obs, deterministic=True))
