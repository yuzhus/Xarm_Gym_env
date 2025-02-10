import gymnasium as gym
import gym_examples
import time
from stable_baselines3 import PPO

env = gym.make("gym_examples/GridWorld-v0",render_mode="rgb_array")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=30_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    time.sleep(1./10)
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()