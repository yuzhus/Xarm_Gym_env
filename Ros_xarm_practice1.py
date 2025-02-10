#!/usr/bin/env python
import rospy
import time
from xarm_msgs.srv import *
from xarm.wrapper import XArmAPI

ip = '192.168.1.214'
arm = XArmAPI(ip, is_radian=False)

arm_eef_index = 6
time.sleep(10)
def blended_motions():

	move_jointb = rospy.ServiceProxy('/xarm/move_jointb', Move)
	
	req_jointb = MoveRequest() 
	req_jointb.mvvelo = 0.2 # rad/s
	req_jointb.mvacc = 30 # rad/s^2
	req_jointb.mvtime = 0 
	req_jointb.mvradii = 25 # blending radius: 25 mm
	joint_list = [[ 1.15314274e-01, -7.57213274e-02, -4.36476083e-02,  1.15482100e-01, 1.13484611e-01], 
				[ 2.27850329e-01, -1.46599129e-01, -9.99646067e-02,  2.43139518e-01, 2.26207533e-01],
				[ 3.39709598e-01, -2.15343275e-01, -1.61202895e-01,  3.73653024e-01, 3.38236718e-01],
				[ 4.54699015e-01, -2.83541963e-01, -2.22938144e-01,  5.04044668e-01, 4.53326076e-01],
				[ 5.77838500e-01, -3.51625407e-01, -2.84046778e-01,  6.33607342e-01, 5.76483259e-01],
				[ 7.14626406e-01, -4.17368378e-01, -3.46946449e-01,  7.62545251e-01, 7.13211842e-01],
				[ 8.70580906e-01, -4.74864405e-01, -4.16722664e-01,  8.90045051e-01, 8.69042701e-01],
				[ 1.02762890e+00, -4.97889599e-01, -5.13035278e-01,  1.00952163e+00, 1.02617973e+00],
				[ 1.17636811e+00, -4.82054152e-01, -6.35804949e-01,  1.11642869e+00, 1.17512771e+00],
				[ 1.31099725e+00, -4.40825163e-01, -7.74368703e-01,  1.21360003e+00, 1.31001017e+00],
				[ 1.42602516e+00, -4.19560980e-01, -8.95703271e-01,  1.31370132e+00, 1.42524829e+00],
				[ 1.53292668e+00, -4.01994900e-01, -9.79016524e-01,  1.37985929e+00, 1.53223459e+00],
				[ 1.63653874e+00, -3.82947192e-01, -1.00910626e+00,  1.39162008e+00, 1.63589141e+00],
				[ 1.73803079e+00, -3.57878143e-01, -1.00123930e+00,  1.35924088e+00, 1.73742907e+00],
				[ 1.83726524e+00, -3.22744439e-01, -9.71301999e-01,  1.29450864e+00, 1.83671825e+00],
				[ 1.93318415e+00, -2.70565857e-01, -9.34033038e-01,  1.20520517e+00, 1.93270418e+00],
				[ 2.02419539e+00, -1.96004363e-01, -9.03225523e-01,  1.09982122e+00, 2.02379292e+00],
				[ 2.10945819e+00, -9.97347141e-02, -8.88866645e-01,  9.89070467e-01, 2.10913279e+00],
				[ 2.18842618e+00,  1.43306426e-02, -8.97425513e-01,  8.83355469e-01, 2.18817186e+00],
				[ 2.24256395e+00,  1.22054362e-01, -9.15207889e-01,  7.93309429e-01, 2.24241370e+00],
				[ 2.27296622e+00,  2.00160923e-01, -9.31241825e-01,  7.31173832e-01, 2.27288967e+00]
			   ]
	
	# position_list = [[300, 200, 100, -3.14, 0, 0],[500, 200, 100, -3.14, 0, 0],[500, -200, 100, -3.14, 0, 0]]
	# for i in range(len(position_list)):
	# 	code, jointpose = arm.get_inverse_kinematics(position_list[i], input_is_radian=False, return_is_radian=False)
	# 	if code:
	# 		print("No valid IK solution.")
	# 		continue
	# 	print(jointpose)

	ret = 0
	try:
		# blended linear motions: 
		for i in range(len(joint_list)):
			req_jointb.pose = joint_list[i]
			res = move_jointb(req_jointb)
			if res.ret:
				print("Something Wrong happened calling move_lineb service, index is %d, ret = %d"%(i, res.ret))
				ret = -1
				break
		return ret

	except rospy.ServiceException as e:
		print("motion Service call failed: %s"%e)
		return -1
	

def gohome_motions():

	move_line_tool = rospy.ServiceProxy('/xarm/move_line_tool', Move)
	
	req_jointb = MoveRequest() 
	req_jointb.mvvelo = 500 # mm/s
	req_jointb.mvacc = 30 # mm/s^2
	req_jointb.mvtime = 0 
	req_jointb.mvradii = 0 # blending radius: 25 mm
	position_list = [[0, 0, -200, 0, 0, 0],[300, 0, 0, 0, 0, 0]]

	ret = 0
	try:
		# blended linear motions: 
		for i in range(len(position_list)):
			req_jointb.pose = position_list[i]
			res = move_line_tool(req_jointb)
			if res.ret:
				print("Something Wrong happened calling move_line_tool service, index is %d, ret = %d"%(i, res.ret))
				ret = -1
				break
		return ret

	except rospy.ServiceException as e:
		print("motion Service call failed: %s"%e)
		return -1


if __name__ == "__main__":

	rospy.wait_for_service('/xarm/move_jointb')
	rospy.set_param('/xarm/wait_for_finish', True) # return after motion service finish
	
	motion_en = rospy.ServiceProxy('/xarm/motion_ctrl', SetAxis)
	set_mode = rospy.ServiceProxy('/xarm/set_mode', SetInt16)
	set_state = rospy.ServiceProxy('/xarm/set_state', SetInt16)
	get_position = rospy.ServiceProxy('/xarm/get_position_rpy', GetFloat32List)
	home = rospy.ServiceProxy('/xarm/go_home', Move)

	try:
		motion_en(8,1)
		set_mode(0)
		set_state(0)
		req = MoveRequest() # MoveRequest for go_home
		req.mvvelo = 0.2 # rad/s
		req.mvacc = 30 # rad/s^2
		req.mvtime = 0
		home(req)

	except rospy.ServiceException as e:
		print("go_home, service call failed: %s"%e)
		exit(-1)

	rospy.set_param('/xarm/wait_for_finish', False) # *CRITICAL for move_lineb and move_jointb, for successful blending

	if blended_motions() == 0:
		print("execution finished!")

	if gohome_motions() == 0:
		print("execution finished!")
		
	home(req)

	rospy.set_param('/xarm/wait_for_finish', True) # After sending all blending motion commands, you can set wait_for_finish back to true if needed.