#!/usr/bin/env python3
import rospy
import time
import os
import gym
import Gym_xarm
import numpy as np
# Import the RL model
from stable_baselines3 import DDPG,PPO
from xarm_msgs.srv import *

n_steps = 20
dt = 1./240 * n_steps
max_vel = 0.5

def Move_Xarm(config, save_dir):

	# initialize the simulation
	env = gym.make('Gym_xarm/XarmReach-v0', config = config, render_mode="rgb_array") 
	loaded_model = PPO.load(f"{save_dir}/saved_XarmReach_best")

	# Start a new episode
	obs = env.reset()
	for i in range(35):

		# predict the action from the trained model
		action, _states = loaded_model.predict(obs, deterministic=True)
		delta_x = action[0]*max_vel*dt
		delta_y = -action[1]*max_vel*dt
		delta_z = -action[2]*max_vel*dt
		real_action = [delta_x*1000, delta_y*1000, delta_z*1000, 0, 0, 0]
		real_action = np.clip(real_action, -1000, +1000)

		# run in simulation
		obs, rewards, dones, info = env.step(action)
		env.render()

		
		try:
			move_line = rospy.ServiceProxy("/xarm/move_line_tool",Move)
			move_line(pose=real_action, mvvelo=500, mvacc=2000, mvtime=0, mvradii=0)
			time.sleep(0.5)
			print("xarm is moving, action is", real_action)
		except rospy.ServiceException as e:
			rospy.logwarn("Service failed: " + str(e))



		if dones or info.get("is_success", False):
			print("Success?", info.get("is_success", False))
			break
			# obs = env.reset()
	env.close()


if __name__ == '__main__':
	
	save_dir = "/home/ros-tf/Gym_env"
	os.makedirs(save_dir, exist_ok=True)
	config = {
		'GUI': True, 
		'reward_type':'dense_diff'
	}

	# ROS
	rospy.init_node("implement_trained_model")
	rospy.wait_for_service("/xarm/move_line_tool")

	rospy.set_param('/xarm/wait_for_finish', True) # return after motion service finish
	motion_en = rospy.ServiceProxy('/xarm/motion_ctrl', SetAxis)
	set_mode = rospy.ServiceProxy('/xarm/set_mode', SetInt16)
	set_state = rospy.ServiceProxy('/xarm/set_state', SetInt16)
	get_position = rospy.ServiceProxy('/xarm/get_position_rpy', GetFloat32List)
	home = rospy.ServiceProxy('/xarm/go_home', Move)

	# initialize the Xarm
	try:
		motion_en(8,1)
		set_mode(0)
		set_state(0)
		req = MoveRequest() # MoveRequest for go_home
		req.mvvelo = 0.4 # rad/s
		req.mvacc = 30 # rad/s^2
		req.mvtime = 0
		home(req)

	except rospy.ServiceException as e:
		print("go_home, service call failed: %s"%e)
		exit(-1)

	rospy.set_param('/xarm/wait_for_finish', False) 
	# *CRITICAL for move_lineb and move_jointb, for successful blending


	if Move_Xarm(config, save_dir) == 0:
		print("execution of Xarm finished!")


	rospy.set_param('/xarm/wait_for_finish', True) 
	# After sending all blending motion commands, you can set wait_for_finish back to true if needed.



	





		










	