import gym_examples
import gymnasium 

from gymnasium.wrappers import FlattenObservation

env = gymnasium.make('gym_examples/GridWorld-v0')
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset()) 
