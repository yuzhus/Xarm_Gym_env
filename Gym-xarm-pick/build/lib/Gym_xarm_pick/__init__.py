from gym.envs.registration import register
#We are registering the env to the gym 0.21, not gymnasium. So, the version of stablebaseline3 should be 1.7.0
register(
     id="Gym_xarm_pick/XarmPick-v0",
     entry_point="Gym_xarm_pick.envs:XarmPickEnv",
     max_episode_steps=50,
)
