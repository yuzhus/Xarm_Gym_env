import os
import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
import pybullet_data as pd
import math

import rospy
import time
from xarm_msgs.srv import *
# from xarm.wrapper import XArmAPI
# ip = '192.168.1.214'
# arm = XArmAPI(ip, is_radian=False)


class XarmPickEnv(gym.GoalEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config, render_mode=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.config = config

        # robot parameters
        self.num_joints = 15
        self.gripper_base_index = 7
        self.arm_eef_index = 6
        self.finger_left_outer_joint = 8
        self.finger_right_outer_joint = 11
        self.finger_left_outer_link = 9
        self.finger_right_outer_link = 12
        self.robot_tcp = 14
        self.max_vel = 0.3
        self.max_gripper_vel = 0.05
        self.startPos = [-0.3, 0, 0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.joint_init_pos = [0, 0, 0, 0, 0, 0, 0, 0] + [0]*9

        # training parameters
        self.render_mode = render_mode
        self.timeStep=1./240
        self.n_substeps = 20
        self.dt = self.timeStep*self.n_substeps
        self.gamma = 0.1

        self.pos_space = spaces.Box(low=np.array([-1, -1 ,0]), high=np.array([1, 1, 1]))
        self.goal_space = spaces.Box(low=np.array([0.325, 0, 0.4]),high=np.array([0.325, 0, 0.4]))#box
        self.obj_space = spaces.Box(low=np.array([-0.55, 0.35, 0.025]), high=np.array([-0.75, 0.55,0.025]))#cube
        self.gripper_space = spaces.Box(low=0.01, high=0.04, shape=[1])
        
        self.distance_threshold=0.05
        self._max_episode_steps = 150 # mush as same as the one defined in the init.py for registering the env to gym
        self.reward_type = config['reward_type']

        self.theta = 0.1
        self.p = 8
        self.dref = 0.2
        
        # connect bullet
        if config['GUI']:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])

        # bullet setup
        self.seed()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        p.setGravity(0,0,-9.8)

        # load table
        # table1 parameters
        self.table_pos_1 = [-0.7,0.5,-0.625]
        self.table_orn_1 = p.getQuaternionFromEuler([0,0,-math.pi/2])
        self.table_1 = p.loadURDF("table/table.urdf", self.table_pos_1, self.table_orn_1, useFixedBase=True)
        # table2 parameters
        self.table_pos_2 = [0.7,0.5,-0.625]
        self.table_orn_2 = p.getQuaternionFromEuler([0,0,-math.pi/2])
        self.table_2 = p.loadURDF("table/table.urdf", self.table_pos_2, self.table_orn_2, useFixedBase=True)

        # load arm
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm5_with_gripper.urdf')
        self.xarm = p.loadURDF(fullpath, self.startPos, self.startOrientation, useFixedBase=True)
        # jointPoses = self._p.calculateInverseKinematics(self.xarm, self.arm_eef_index, self.startGripPos, [1,0,0,0])[:self.arm_eef_index]
        for i in range(self.num_joints):
            p.resetJointState(self.xarm, i, self.joint_init_pos[i])

        # load goal
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_cube.urdf')
        self.cube = p.loadURDF(fullpath)

        # load box
        # box parameters
        self.box_pos = [0.325,0,0]
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/box.urdf')
        self.box = p.loadURDF(fullpath, self.box_pos, useFixedBase=True)

        # load human
        # human parameters
        self.rightElbow = 4
        self.rightElbowRot = 1.5
        human_pos = [-0.17,0.63,0.22]
        human_orn = p.getQuaternionFromEuler([math.pi,0,-math.pi])
        self.human = p.loadURDF("urdf/wall.urdf", human_pos, human_orn, useFixedBase=True)
        

        # gym setup
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        ))
        

        p.stepSimulation()
        p.setRealTimeSimulation(True)
        p.performCollisionDetection()

        # sim2real ROS Service setup
        if config['Sim2Real']:
            rospy.wait_for_service('/xarm/move_jointb')
            rospy.set_param('/xarm/wait_for_finish', True) # return after motion service finish
            self.motion_en = rospy.ServiceProxy('/xarm/motion_ctrl', SetAxis)
            self.set_mode = rospy.ServiceProxy('/xarm/set_mode', SetInt16)
            self.set_state = rospy.ServiceProxy('/xarm/set_state', SetInt16)
            self.get_position = rospy.ServiceProxy('/xarm/get_position_rpy', GetFloat32List)
            self.home = rospy.ServiceProxy('/xarm/go_home', Move)

            try:
                self.motion_en(8,1)
                self.set_mode(0)
                self.set_state(0)
                self.req = MoveRequest() # MoveRequest for go_home
                self.req.mvvelo = 0.2 # rad/s
                self.req.mvacc = 30 # rad/s^2
                self.req.mvtime = 0
                self.home(self.req)

            except rospy.ServiceException as e:
                print("go_home, service call failed: %s"%e)
                exit(-1)
            rospy.set_param('/xarm/wait_for_finish', False) 


###################################################################################################################
    # sim2real methods
    # -------------------------

    def implement_realXarm(self, jointPoses):
        move_jointb = rospy.ServiceProxy('/xarm/move_jointb', Move)
        req_jointb = MoveRequest() 
        req_jointb.mvvelo = 0.3 # rad/s
        req_jointb.mvacc = 30 # rad/s^2
        req_jointb.mvtime = 0 
        req_jointb.mvradii = 5 # blending radius: 25 mm

        ret = 0
        try:
            req_jointb.pose = jointPoses[0:5]
            res = move_jointb(req_jointb)
            if res.ret:
                print("Something Wrong happened calling move_lineb service")
                ret = -1
            return ret

        except rospy.ServiceException as e:
            print("motion Service call failed: %s"%e)
            return -1

    def gohome_motions(self):

        move_line_tool = rospy.ServiceProxy('/xarm/move_line_tool', Move)
        
        req_jointb = MoveRequest() 
        req_jointb.mvvelo = 800 # mm/s
        req_jointb.mvacc = 200 # mm/s^2
        req_jointb.mvtime = 0 
        req_jointb.mvradii = 0 # blending radius: 25 mm
        position_list = [[0, 0, -200, 0, 0, 0],[350, 0, 0, 0, 0, 0]]

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


    # basic gym environment methods
    # -------------------------

    def step(self, action):
        self.num_steps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        jointPoses = np.array(self._set_action(action))
        # print("joint poses", jointPoses)

        ###################################################################
        if self.config['Sim2Real']:
            if self.implement_realXarm(jointPoses) == 0:
                print("Implementation finished!")  
        ####################################################################
        
        p.setGravity(0,0,-9.8)
        p.stepSimulation()
        obs = self._get_obs()

        # print("obs looks like: ", obs['observation'])
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'future_length': self._max_episode_steps - self.num_steps
        }
        reward = self.compute_reward(action, obs['achieved_goal'], obs['observation'], self.goal, info)
        done = (self.num_steps == self._max_episode_steps)
        return obs, reward, done, info


    def reset(self):
        super(XarmPickEnv, self).reset()

        ######################################################
        if self.config['Sim2Real']:
            # After sending all blending motion commands, you can set wait_for_finish back to true if needed.
            rospy.set_param('/xarm/wait_for_finish', True) 
            self.gohome_motions()
            self.home(self.req)
            print("gohome finished!") 
            rospy.set_param('/xarm/wait_for_finish', False) 
        ######################################################
        self._reset_sim()
        self.goal = self._sample_goal()
        self.d_eef2Cub_old = np.linalg.norm(p.getLinkState(self.xarm, self.robot_tcp)[0] - self.goal[0], axis=-1)
        self.d_cub2Box_old = np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.cube)[0]) - self.goal[1], axis=-1)
        self.num_steps = 0
        return self._get_obs()

    # GoalEnv methods
    # -------------------------

    def compute_reward(self, action, achieved_goal, observation, goal, info):
        self.c1 = 500
        self.c2 = 0
        self.c3 = 12
        d_eef2Cub = np.linalg.norm(achieved_goal[0] - achieved_goal[1] , axis=-1)
        d_cub2Box = np.linalg.norm(achieved_goal[1] - goal[1], axis=-1)
        
        if_grasp = len(p.getContactPoints(self.xarm, self.cube, self.finger_left_outer_link))!=0 and len(p.getContactPoints(self.xarm, self.cube, self.finger_right_outer_link))!=0
        # grip_pos = np.array(p.getLinkState(self.xarm, self.gripper_base_index)[0])

        Rt_1 = self.d_eef2Cub_old - d_eef2Cub
        self.d_eef2Cub_old = d_eef2Cub

        Rt = Rt_1
        if if_grasp:
            Rt += 10
            Rt_2 = self.d_cub2Box_old - d_cub2Box
            self.d_cub2Box_old = d_cub2Box
            Rt = Rt + Rt_2

        

        # the magnitude of the actions ========================
        Ra = -np.inner(action,action)



        # obstacle avoidance ========================
        closestPoint = np.array(observation[15:18])
        griperPoint = np.array(observation[3:6])
        closestDistance = np.linalg.norm(griperPoint - closestPoint, axis=-1)
        # print('closest distance',closestDistance)
        if closestDistance < 0.15:
            Ro = -1
            # print('give collision punishment!')
        else:
            Ro = 0

        if self.reward_type == 'dense_diff':
            R = self.c1*Rt + self.c2*Ra + self.c3*Ro 
            # print('Rt is', self.c1*Rt, 'Ra', self.c2*Ra, 'Ro', self.c3*Ro)
            return R

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, config, mode="rgb_array", width=500, height=500):
        if mode == 'rgb_array':
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(0.3, 0, 0.2),
                distance=1.2,
                yaw=45,
                pitch=-10,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))
            
            return rgb_array
        elif mode == 'human':
            time.sleep(self.dt)  # wait to seems like real speed

    def close(self):
        p.disconnect()

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        assert action.shape == (4,), 'action shape error'
        vel_control = np.clip(action, self.action_space.low, self.action_space.high)
        cur_pos = np.array(p.getLinkState(self.xarm, self.arm_eef_index)[0])
        new_pos = cur_pos + np.array(vel_control [:3]) * self.max_vel * self.dt
        new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)

        cur_gripper_pos = p.getJointState(self.xarm, self.finger_left_outer_link)[0]
        new_gripper_pos = np.clip(cur_gripper_pos + vel_control[3]*self.dt * self.max_gripper_vel, self.gripper_space.low, self.gripper_space.high)
        jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, new_pos, [1,0,0,0], maxNumIterations = self.n_substeps)
        
        for i in range(1, self.arm_eef_index):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1],force=5 * 240.)
        
        p.setJointMotorControl2(self.xarm, self.finger_left_outer_joint, p.POSITION_CONTROL, new_gripper_pos, force=1000)
        p.setJointMotorControl2(self.xarm, self.finger_right_outer_joint, p.POSITION_CONTROL, new_gripper_pos, force=1000)
        if_grasp = np.any([len(p.getContactPoints(self.xarm, self.cube, self.finger_left_outer_link))!=0 and len(p.getContactPoints(self.xarm, self.cube, self.finger_right_outer_link))!=0 ])
        if if_grasp:
            p.changeDynamics(self.xarm, self.finger_left_outer_link, lateralFriction = 100)
            p.changeDynamics(self.xarm, self.finger_right_outer_link, lateralFriction = 100)
        else:  # grasp success -> change friction
            p.changeDynamics(self.xarm, self.finger_left_outer_link, lateralFriction = 1)
            p.changeDynamics(self.xarm, self.finger_right_outer_link, lateralFriction = 1)
        # p.setJointMotorControl2(self.human,self.rightElbow, p.POSITION_CONTROL, targetPosition=self.rightElbowRot,force=1000)
        return jointPoses
    
    def _get_obs(self):
        ## robot state
        # tcp state
        tcp_state = p.getLinkState(self.xarm, self.robot_tcp, computeLinkVelocity=1)
        tcp_pos = np.array(tcp_state[0])
        tcp_vel = np.array(tcp_state[6])

        # gripper state
        gripper_state = p.getLinkState(self.xarm, self.gripper_base_index, computeLinkVelocity=1)
        gripper_pos = np.array(gripper_state[0])
        gripper_vel = np.array(gripper_state[6])

        robot_state = np.concatenate((tcp_pos, gripper_pos))



        # cube state
        cub_pos = np.array(p.getBasePositionAndOrientation(self.cube)[0])
        cub_rot = np.array(p.getBasePositionAndOrientation(self.cube)[1])

        cub_state = np.concatenate((cub_pos, cub_rot))

        achieved_goal = np.append([tcp_pos], [cub_pos], axis=0)


        # collision detection
        contact_points = p.getContactPoints(self.xarm, self.human, linkIndexA = self.finger_left_outer_link, linkIndexB = -1)

        if len(contact_points) !=0:
            collisionPoint = contact_points[0][5] # collision points on A
            # print('Collision occurs at', collisionPoint)
        else:
            collisionPoint = [0,0,0]

        # calculate the closest distance between the robot and obstacles
        closestPoints = p.getClosestPoints(self.xarm, self.human, distance=10, linkIndexA = self.finger_left_outer_link, linkIndexB=-1 )
        closestPoint = closestPoints[0][6] # closest points on B
        # print('closestPoint', closestPoint)

        obs = np.concatenate((
                    robot_state, cub_state, collisionPoint, closestPoint, tcp_vel, gripper_vel
        ))


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        # reset arm
        for i in range(self.num_joints):
            p.resetJointState(self.xarm, i, self.joint_init_pos[i])
        return True

    def _sample_goal(self):
        goal_getcube = np.array(self.obj_space.sample()) #cube
        goal_getbox = np.array(self.goal_space.sample()) #box
        goal = np.append([goal_getcube], [goal_getbox],axis=0)

        # print(goal)
        p.resetBasePositionAndOrientation(self.cube, goal[0], self.startOrientation)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal[1] - self.goal[1], axis=-1)
        return (d < self.distance_threshold).astype(np.float32)