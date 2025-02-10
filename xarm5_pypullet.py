import numpy as np
import pybullet as p
import time
import math 
import pybullet_data as pd
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped


rospy.init_node('test_xarm', anonymous=True)



useDebugMode = 0
xarmEndEffectorIndex = 5
arm_eef_index = 6
xarmNumDofs = 5
finger_left_index = 10
finger_right_index = 12
useFixedBase = True
flags = p.URDF_INITIALIZE_SAT_FEATURES#0#p.URDF_USE_SELF_COLLISION

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13

#rightShoulder=3
#rightElbow=4
#leftShoulder=6
#leftElbow = 7
#rightHip=9
#rightKnee=10
#rightAnkle=11
#leftHip = 12
#leftKnee=13
#leftAnkle=14



physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pd.getDataPath()) #optionally

table_pos_1 = [-0.7,0.5,-0.625]
table_orn_1 = p.getQuaternionFromEuler([0,0,-math.pi/2])
table_1 = p.loadURDF("table/table.urdf", table_pos_1, table_orn_1, flags = flags, useFixedBase=useFixedBase)

table_pos_2 = [0.7,0.5,-0.625]
table_orn_2 = p.getQuaternionFromEuler([0,0,-math.pi/2])
table_2 = p.loadURDF("table/table.urdf", table_pos_2, table_orn_2, flags = flags, useFixedBase=useFixedBase)

# cube_pos = [[0.2,0.2,0.68], [0.5,0,0.68], [0.2,-0.2,0.68]]
# cube1 = p.loadURDF("urdf/my_cube.urdf", cube_pos[0], flags = flags)
# cube2 = p.loadURDF("urdf/my_cube.urdf", cube_pos[1], flags = flags)
# cube3 = p.loadURDF("urdf/my_cube.urdf", cube_pos[2], flags = flags)

box_pos = [0.325,0,0]
box = p.loadURDF("urdf/box.urdf", box_pos, flags = flags, useFixedBase=useFixedBase)

obstacle_pos = [-0.095,0.6,0.22]
obstacle_orn=p.getQuaternionFromEuler([math.pi,0,-math.pi])
# obstacle = p.loadURDF("urdf/wall.urdf", obstacle_pos, obstacle_orn, flags = flags, useFixedBase=useFixedBase)

xarmPos = [-0.3, 0, 0]
xarmOrn=p.getQuaternionFromEuler([0,0,0])
print('The orn is:',obstacle_orn)
# xarmOrientation = p.getQuaternionFromEuler([0,0,-math.pi/2])
xarm5 = p.loadURDF("urdf/xarm5_with_gripper.urdf", xarmPos, xarmOrn, flags = flags, useFixedBase=useFixedBase)
p.setGravity(0, 0, -9.81)   # everything should fall down

cubePos = [-0.55, 0.35, 0.025]
cube = p.loadURDF('urdf/my_cube.urdf', cubePos)

jointIds = []
paramIds = []
numJoints = p.getNumJoints(xarm5)

# Object parameters of pybullet dynamic object
sphereRadius = 0.025
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])
mass = 1
visualShapeId = -1
basePosition = [-0.65, 0.35, 0.025]
baseOrientation = p.getQuaternionFromEuler([0,0,0])

Dynamic_object = p.createMultiBody(mass, colBoxId, visualShapeId, basePosition, baseOrientation)





if useDebugMode:	
	for j in range(p.getNumJoints(xarm5)):
		p.changeDynamics(xarm5, j, linearDamping=0, angularDamping=0)
		info = p.getJointInfo(xarm5, j)
		#print(info)
		jointName = info[1]
		jointType = info[2]
		if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
			jointIds.append(j)
			paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))
		#addUserDebugParameter lets you add custom sliders and buttons to tune parameters. It will
		#return a unique id. This lets you read the value of the parameter using readUserDebugParameter.


fov, aspect, nearplane, farplane = 60, 1.0, 0.055, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

def xarm_camera():
	# Center of mass position and orientation of last link
	com_p, com_o, _, _, _, _ = p.getLinkState(xarm5, 7, computeForwardKinematics=True)
	rot_matrix = p.getMatrixFromQuaternion(com_o)
	rot_matrix = np.array(rot_matrix).reshape(3, 3)
	# Initial vectors
	init_camera_vector = (0, 0, 1) # z-axis
	init_up_vector = (1, 0, 0) # y-axis
	# Rotated vectors
	camera_vector = rot_matrix.dot(init_camera_vector)
	up_vector = rot_matrix.dot(init_up_vector)
	view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
	img = p.getCameraImage(320, 200, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
	return img


def receive_human_pose_callback(PoseStamp):
	global humanPos
	# print('the pose received:', PoseStamp.pose.position)
	humanPos = [-PoseStamp.pose.position.x - 0.8, -PoseStamp.pose.position.y - 0.78, PoseStamp.pose.position.z - 0.7]
	print(humanPos)
	return humanPos

if __name__ == '__main__':
	global humanPos
	humanPos=[0,0,0.77]
	# while (1):
	for i in range(10000):
		Dynamic_object_targetPos= [-0.55, 0.35 + i * 0.01, 0.025]
		# print(Dynamic_object_targetPos)
		p.stepSimulation()
		p.performCollisionDetection()
		if useDebugMode:
			eef_pos = np.array(p.getLinkState(xarm5, arm_eef_index)[0])
			print('pos of eef is: ', eef_pos)
			for i in range(len(paramIds)):
				c = paramIds[i]
				targetPos = p.readUserDebugParameter(c)
				p.setJointMotorControl2(xarm5, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)	
			# print('the number of joint is: ', p.getNumJoints(xarm5))

		else:
			rospy.Subscriber('/vive/controller/right/pose', PoseStamped, receive_human_pose_callback)
			# print(humanPos)
			jointPoses = p.calculateInverseKinematics(xarm5, finger_left_index, humanPos, [1,0,0,0])
			# jointPoses = [0, 0, 0, 0, 0, 0, 0, -0.5, -0.5, 0, -0.5, -0.5, 0,0,0 ]
			# pos = [0.5,0,0.9]
			# print(pos)
			# orn = [1,0,0,0]
			# jointPoses = p.calculateInverseKinematics(xarm5, xarmEndEffectorIndex, pos,orn, maxNumIterations=50)
			# print("jointPoses=",jointPoses)
			for j in range(finger_left_index):
				p.setJointMotorControl2(xarm5, j, p.POSITION_CONTROL, jointPoses[j-1], force=5 * 240.)

			rightElbow = 4
			rightElbowRot = 1.2
			

			# p.setJointMotorControl2(obstacle,rightElbow, p.POSITION_CONTROL, targetPosition=rightElbowRot,force=1000)
		
		
		# p.resetBasePositionAndOrientation(Dynamic_object, humanPos, baseOrientation)

		
		
		xarm_camera()


		# p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL )
		time.sleep(1./240)
		# closestPoints = p.getClosestPoints(xarm5, xarm5, distance=10, linkIndexA = finger_left_index, linkIndexB=finger_right_index)
		# d_gripper = np.linalg.norm(closestPoints[0][5] - closestPoints[0][6], axis=-1)
		# print(d_gripper)


	p.disconnect()
