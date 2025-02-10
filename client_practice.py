#!/usr/bin/env python3
import sys
import rospy
import time
import os
import gym
import Gym_xarm
import numpy as np
import math
from xarm_msgs.srv import *

if __name__ == '__main__':
	# run in the real experiment
	rospy.init_node("implement_trained_model")
	rospy.wait_for_service("/xarm/move_lineb")
	for i in range(2):
		try:
			move_line = rospy.ServiceProxy("/xarm/move_lineb", Move)
			move_line(pose=[300, 200, 100, -math.pi, 0, 0], mvvelo=200, mvacc=2000, mvtime=0, mvradii=0)
		except rospy.ServiceException as e:
			rospy.logwarn("Service failed: " + str(e))
