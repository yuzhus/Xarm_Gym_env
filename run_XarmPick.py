import Gym_xarm_pick
import gym
import pybulletgym

config = {
    'GUI': True,
    'num_obj': 2, 
    'same_side_rate': 0.5, 
    'goal_shape': 'any', 
    'use_stand': False, 
    'reward_type':'dense'
}

env = gym.make('Gym_xarm_pick/XarmPick-v0', config = config) 
env.render("human")
env.reset()