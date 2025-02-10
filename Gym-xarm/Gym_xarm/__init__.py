from gym.envs.registration import register

register(
     id="Gym_xarm/XarmReach-v0",
     entry_point="Gym_xarm.envs:XarmReachEnv",
     max_episode_steps=50,
)

register(
     id="Gym_xarm/XarmReachSac-v0",
     entry_point="Gym_xarm.envs:XarmReachSacEnv",
     max_episode_steps=50,
)
