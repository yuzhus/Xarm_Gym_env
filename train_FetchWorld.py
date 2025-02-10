import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("FetchReachDense-v2",render_mode="rgb_array")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=30_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
