import Gym_xarm
import gym
import pybullet,gym

config = {
    'GUI': True,
    'num_obj': 2, 
    'same_side_rate': 0.5, 
    'goal_shape': 'any', 
    'use_stand': False, 
    'reward_type':'dense'
}

env = gym.make('Gym_xarm/XarmReach-v0', config = config) 
env.render("human")
env.reset()