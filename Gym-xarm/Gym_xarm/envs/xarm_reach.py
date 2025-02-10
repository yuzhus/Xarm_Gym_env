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
import rospkg
import time
from xarm_msgs.srv import *
from xarm.wrapper import XArmAPI
from geometry_msgs.msg import Point
from std_msgs.msg import Int64

# ip = '192.168.1.214'
# arm = XArmAPI(ip, is_radian=False)

rp = rospkg.RosPack()
package_path = rp.get_path('gym_ros_pybullet')
urdf_path = package_path + '/src/urdf'


class XarmReachEnv(gym.GoalEnv):

    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, config, render_mode=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # Listen to the camera
        # rospy.init_node('Gym_env', anonymous=True)
        

        # Config of environment
        self.config = config

        # robot parameters
        self.num_joints = 15
        self.gripper_driver_index = 8
        self.gripper_base_index = 7
        self.arm_eef_index = 6
        self.finger_left = 9
        self.finger_right = 12
        self.robot_tcp = 14
        self.max_vel = 0.5
        self.max_gripper_vel = 0.1
        self.startPos = [-0.3, 0, 0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.joint_init_pos = [0, 0, 0, 0, 0, 0, 0, 0] + [0]*9
        self.safe_flag = 1
        self.re_training = 1

        # training parameters
        self.render_mode = render_mode
        self.timeStep=1./240
        self.n_substeps = 20
        self.dt = self.timeStep*self.n_substeps
        self.gamma = 0.1

        self.pos_space = spaces.Box(low=np.array([-1, -1 ,0]), high=np.array([1, 1, 1]))
        self.goal_space = spaces.Box(low=np.array([-0.55, 0.35, 0.01]),high=np.array([-0.75, 0.55, 0.01]))
        self.obstacle_space = spaces.Box(low=np.array([-0.4, 0.25, 0.06]),high=np.array([-0.55, 0.5, 0.06]))
        self.gripper_space = spaces.Box(low=0.01, high=0.04, shape=[1])
        
        self.distance_threshold = 0.02
        self._max_episode_steps = 50 # must as same as the one defined in the init.py for registering the env to gym
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
        fullpath = os.path.join(os.path.dirname(__file__), urdf_path+ '/xarm5_with_gripper.urdf')
        self.xarm = p.loadURDF(fullpath, self.startPos, self.startOrientation, useFixedBase=True)
        # jointPoses = self._p.calculateInverseKinematics(self.xarm, self.arm_eef_index, self.startGripPos, [1,0,0,0])[:self.arm_eef_index]
        for i in range(self.num_joints):
            p.resetJointState(self.xarm, i, self.joint_init_pos[i])

        # load box
        # box parameters
        self.box_pos = [0.325,0,0]
        fullpath = os.path.join(os.path.dirname(__file__), urdf_path+'/box.urdf')
        self.box = p.loadURDF(fullpath, self.box_pos, useFixedBase=True)

        # load wall
        # human parameters
        self.rightElbow = 4
        self.rightElbowRot = 1.5
        human_pos = [-0.17,0.63,0.22]
        human_orn = p.getQuaternionFromEuler([math.pi,0,-math.pi])
        self.wall = p.loadURDF(urdf_path+ '/wall.urdf', human_pos, human_orn, useFixedBase=True)
        
        # load pybullet dynamic object
        if config['Digital_twin']:
            # Create sphere object for obstacles
            self.CylinderRadius = 0.06 # 12cm
            self.Cylindermass = 100000
            self.CylinderbasePosition = [-0.95, 0.35, 0.06] # intial position of obstacles, later will be updated by ros.
            self.CylinderbaseOrientation = p.getQuaternionFromEuler([0,0,0])
            self.CylindervisualShapeId = -1 # re-use the collision shape for multiple multibodies
            self.req_height_cylinder = 0.12
            self.CylcolSphereId = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.CylinderRadius, height=self.req_height_cylinder)
            self.obstacle_cylinder = p.createMultiBody(self.Cylindermass, self.CylcolSphereId, self.CylindervisualShapeId, self.CylinderbasePosition, self.CylinderbaseOrientation)
            self.req_pos_cylinder = self.CylinderbasePosition 

            # Create cube object for goal
            self.CubeRadius = 0.01 # 2cm
            self.Cubemass = 1
            self.CubebasePosition = [-0.65, 0.35, 0.01] # intial position of goal, later will be updated by ros.
            self.CubebaseOrientation = p.getQuaternionFromEuler([0,0,0])
            self.CubevisualShapeId = -1
            self.CubecolBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.CubeRadius, self.CubeRadius, self.CubeRadius])
            self.Dynamic_object = p.createMultiBody(self.Cubemass, self.CubecolBoxId, self.CubevisualShapeId, self.CubebasePosition, self.CubebaseOrientation)
            self.req_pos_cube = self.CubebasePosition 
        else:
            # load gym sampled virtual cube for train
            fullpath = os.path.join(os.path.dirname(__file__), urdf_path+'/my_cube.urdf')
            cube_pos = [-0.65, 0.35, 0.01]
            cube_ori = p.getQuaternionFromEuler([math.pi,0,-math.pi])
            self.goal_cube = p.loadURDF(fullpath, cube_pos, cube_ori, useFixedBase=True)

            obs_pos = [-0.50, 0.55, 0.06]
            obs_ori = p.getQuaternionFromEuler([math.pi,0,-math.pi])
            fullpath = os.path.join(os.path.dirname(__file__), urdf_path+'/my_cylinder.urdf')
            self.obstacle_cylinder = p.loadURDF(fullpath, obs_pos, obs_ori, useFixedBase=True)
        
        # gym setup
        if config['Digital_twin']:
            self.goal = self._receive_goal_from_ros(self.req_pos_cube)
            self.obstacle = self._receive_obstacle_from_ros(self.req_pos_cylinder, self.req_height_cylinder)
        else:
            self.goal = self._sample_goal()
            self.obstacle = self._sample_obstacle()
        
        obs = self._get_obs()

        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype='float32'),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype='float32'),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype='float32'),
                "obstacles": spaces.Box(-np.inf, np.inf, shape=obs["obstacles"].shape, dtype='float32'),
            }
        )

        p.stepSimulation()
        # p.setRealTimeSimulation(True) #If enable the real-time simulation, don't need to call 'stepSimulation'.
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

    # Sim2real methods
    # ----------------------------------------------------------------------------------
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
        position_list = [[0, 0, -250, 0, 0, 0],[350, 0, 0, 0, 0, 0]]

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


    # Basic gym environment methods
    # ----------------------------------------------------------------------------------
    def step(self, action):

        if self.config['Digital_twin']:
            rospy.Subscriber('/digital_twin_goal_position', Point, self._receive_goal_callback)
            rospy.Subscriber('/obstacle_size', Point, self._receive_obstacle_size_callback)
            rospy.Subscriber('/digital_twin_obstacle_position', Point, self._receive_obstacle_callback)
            pub_retrain = rospy.Publisher('/Re_taining', Int64)
            pub_retrain.publish(self.re_training)
            
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
            'is_success': self._is_success(obs['achieved_goal']),
            'future_length': self._max_episode_steps - self.num_steps,
            'is_safe': self.safe_flag,
            'Resume_training': self.re_training
        }

        reward = self.compute_reward(action, obs['achieved_goal'], obs['observation'], self.goal, info)

        done = ((self.num_steps == self._max_episode_steps))
       

        return obs, reward, done, info

    def reset(self):
        super(XarmReachEnv, self).reset()
        self.num_steps = 0
        ######################################################
        if self.config['Sim2Real']:
            # After sending all blending motion commands, you can set wait_for_finish back to true if needed.
            rospy.set_param('/xarm/wait_for_finish', True) 
            self.gohome_motions()
            self.home(self.req)
            print("gohome finished!") 
            rospy.set_param('/xarm/wait_for_finish', False) 
        ######################################################
        
        if self.config['Digital_twin']:
            p.removeBody(self.obstacle_cylinder)
            # p.removeCollisionShape(self.CylcolSphereId)
            self.goal = self._receive_goal_from_ros(self.req_pos_cube)
            self.obstacle = self._receive_obstacle_from_ros(self.req_pos_cylinder, self.req_height_cylinder)
            # print('Get real goal from camera:', self.goal)
        else:
            self.goal = self._sample_goal()
            self.obstacle = self._sample_obstacle()
            self.distance = np.linalg.norm(self.goal - self.obstacle, axis=-1)
            while (self.distance < 0.13):
                self.goal = self._sample_goal()
                self.obstacle = self._sample_obstacle()
                self.distance = np.linalg.norm(self.goal - self.obstacle, axis=-1)

            # print('Get sample goal:', self.goal)
            # print('Get sample obstacle:', self.obstacle)
            # print('self.req_pos_cube',self.req_pos_cube)

        self._reset_arm()
        self.safe_flag = 1
        self.d_old = np.linalg.norm(p.getLinkState(self.xarm, self.robot_tcp)[0] - self.goal, axis=-1)
        
        return self._get_obs()


    # GoalEnv methods
    # ----------------------------------------------------------------------------------
    def compute_reward(self, action, achieved_goal, observation, goal, info):
        self.c1 = 520
        self.c2 = 0
        self.c3 = 12
        d = np.linalg.norm(achieved_goal - goal, axis=-1)

        # Rt = 0.5 - d**0.3
        
        Rt = self.d_old - d 
        self.d_old = d

        # if abs(d) < self.theta:
        #     Rt = 0.5*d**2
        # else:
        #     Rt = self.theta*(abs(d)-0.5*self.theta)
         
        # the magnitude of the actions ========================
        Ra = -np.inner(action,action)
        
        # distance between the robot and obstacles ========================
        # collisionPoint = np.array(observation[6:9])
        # if np.inner(collisionPoint,collisionPoint) != 0:
        #     Ro = -1
        #     # print('give collision punishment!')
        # else:
        #     Ro = 0

        griperPoint = np.array(observation[3:6])

        closestPoint_wall_left = np.array(observation[6:9])
        closestPoint_wall_right = np.array(observation[9:12])
        closestDistance_wall_left = np.linalg.norm(griperPoint- closestPoint_wall_left, axis=-1)
        closestDistance_wall_right = np.linalg.norm(griperPoint- closestPoint_wall_right, axis=-1)


        # print('closest distance',closestDistance)
        closestPoint_obs_left = np.array(observation[12:15])
        closestPoint_obs_right = np.array(observation[15:18])
        closestDistance_obs_left = np.linalg.norm(griperPoint - closestPoint_obs_left, axis=-1)
        closestDistance_obs_right = np.linalg.norm(griperPoint - closestPoint_obs_right, axis=-1)

        # Ro = (self.dref/(closestDistance + self.dref))**self.p
        # print('Closest Distance',closestDistance)

        if (closestDistance_wall_left < 0.13 or closestDistance_wall_right < 0.13):
            Ro_1 = -1
            print('give collision punishment for collision with wall!')
        else:
            Ro_1 = 0

        if ((closestDistance_obs_left < 0.15 or closestDistance_obs_right < 0.15)):
            Ro_2 = -0.8
            print('give collision punishment for collision with obstacles!')
        else:
            Ro_2 = 0


        if self.reward_type == 'dense_diff':
            R = self.c1*Rt + self.c2*Ra + self.c3*(Ro_1 + Ro_2)
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


    # Private method
    # -----------------------------------------------------------------------------------
    def _set_action(self, action):
        assert action.shape == (4,), 'action shape error'

        cur_pos = np.array(p.getLinkState(self.xarm, self.arm_eef_index)[0])
        new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
        new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)

        cur_gripper_pos = p.getJointState(self.xarm, self.gripper_driver_index)[0]
        new_gripper_pos = cur_gripper_pos + action[3]*self.dt * self.max_gripper_vel
        jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, new_pos, [1,0,0,0], maxNumIterations = self.n_substeps)
        for i in range(1, self.arm_eef_index):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1],force=5 * 240.)
        for i in range(self.gripper_driver_index, self.num_joints):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, new_gripper_pos,force=5 * 240.)
        # p.setJointMotorControl2(self.wall,self.rightElbow, p.POSITION_CONTROL, targetPosition=self.rightElbowRot,force=1000)
        return jointPoses[0:self.arm_eef_index]
    
    def _get_obs(self):

        # tcp state
        tcp_state = p.getLinkState(self.xarm, self.robot_tcp, computeLinkVelocity=1)
        tcp_pos = np.array(tcp_state[0])
        #tcp_vel = np.array(tcp_state[6])

        # griper state
        griper_state = p.getLinkState(self.xarm, self.gripper_driver_index, computeLinkVelocity=1)
        griper_pos = np.array(griper_state[0])

        # collision detection (two finger of gripper)
        contact_points_wall_left = p.getContactPoints(self.xarm, self.wall, linkIndexA = self.finger_left, linkIndexB = -1)
        contact_points_wall_right = p.getContactPoints(self.xarm, self.wall, linkIndexA = self.finger_right, linkIndexB = -1)
        contact_points_obs_left = p.getContactPoints(self.xarm, self.obstacle_cylinder, linkIndexA = self.finger_left, linkIndexB = -1)
        contact_points_obs_right = p.getContactPoints(self.xarm, self.obstacle_cylinder, linkIndexA = self.finger_right, linkIndexB = -1)
        
        if (len(contact_points_wall_left) !=0 ):
            # collisionPoint = contact_points_wall[0][5] # collision points on A
            self.safe_flag = 0
            print('Collision occurs on the wall !!!!!')
        if (len(contact_points_wall_right) !=0 ):
            # collisionPoint = contact_points_wall[0][5] # collision points on A
            self.safe_flag = 0
            print('Collision occurs on the wall !!!!!')
        if (len(contact_points_obs_left) !=0):
            # collisionPoint = contact_points_wall[0][5] # collision points on A
            self.safe_flag = 0
            print('Collision occurs on the left finger with obstacles !!!!!')
        if (len(contact_points_obs_right) !=0):
            # collisionPoint = contact_points_wall[0][5] # collision points on A
            self.safe_flag = 0
            print('Collision occurs on the right finger with obstacles !!!!!')
        # else:
        #     collisionPoint = [0,0,0]
        # print('self.safe_flag',self.safe_flag)
            

        # calculate the closest distance between the robot and blue wall
        closestPoints_wall_left = p.getClosestPoints(self.xarm, self.wall, distance=10, linkIndexA = self.finger_left, linkIndexB=-1 )
        closestPoint_wall_left = closestPoints_wall_left[0][6] # closest points on B
        closestPoints_wall_right = p.getClosestPoints(self.xarm, self.wall, distance=10, linkIndexA = self.finger_right, linkIndexB=-1 )
        closestPoint_wall_right = closestPoints_wall_right[0][6] # closest points on B
        # print('closestPoint', closestPoint)

        # calculate the closest points between the robot and obstacles on obstacles
        closestPoints_obstacle_left = p.getClosestPoints(self.xarm, self.obstacle_cylinder, distance=10, linkIndexA = self.finger_left, linkIndexB=-1 )
        closestPoint_obstacle_left = closestPoints_obstacle_left[0][6] # closest points on B
        closestPoints_obstacle_right = p.getClosestPoints(self.xarm, self.obstacle_cylinder, distance=10, linkIndexA = self.finger_right, linkIndexB=-1 )
        closestPoint_obstacle_right = closestPoints_obstacle_right[0][6] # closest points on B
        # print('closestPoint', closestPoint)

        # # calculate the closest distance between the robot and obstacles
        # closestDistance_wall_left = np.linalg.norm(tcp_pos - closestPoint_wall_left, axis=-1)
        # closestDistance_wall_right = np.linalg.norm(tcp_pos - closestPoint_wall_right, axis=-1)

        obs = np.concatenate((
                    tcp_pos, griper_pos, closestPoint_wall_left, closestPoint_wall_right, closestPoint_obstacle_left, closestPoint_obstacle_right
        ))
        # print('observation', obs[6:9], obs[9:12])


        return {
            "observation": obs.copy(),
            "achieved_goal": np.squeeze(tcp_pos.copy()),
            "desired_goal": self.goal.copy(),
            "obstacles": self.obstacle.copy(),
        }

    def _reset_arm(self):
        # if self.config['Digital_twin']:
        #     p.removeBody(self.Dynamic_obstacle)
        #     p.removeCollisionShape(self.CylcolSphereId)
        
        # reset arm
        for i in range(self.num_joints):
            p.resetJointState(self.xarm, i, self.joint_init_pos[i])
        return True

    def _sample_goal(self):
        goal = np.array(self.goal_space.sample())
        p.resetBasePositionAndOrientation(self.goal_cube, goal, self.startOrientation)
        return goal.copy()
    
    def _sample_obstacle(self):
        obstacle = np.array(self.obstacle_space.sample())
        p.resetBasePositionAndOrientation(self.obstacle_cylinder, obstacle, self.startOrientation)
        return obstacle.copy()
    
    def _receive_goal_callback(self, data):
        self.req_pos_cube = np.array([data.x, data.y, data.z])

    def _receive_obstacle_callback(self, data):
        self.req_pos_cylinder = np.array([data.x, data.y, data.z])

    def _receive_obstacle_size_callback(self, data):
        self.req_height_cylinder = data.z
    
    def _receive_goal_from_ros(self, data):
        if len(data)!=0:
            goal = np.array([data[0], data[1], data[2]])
            p.resetBasePositionAndOrientation(self.Dynamic_object, goal, self.CubebaseOrientation)
        else:
            goal = [-0.65, 0.35, 0.025]
        return goal.copy()
    
    def _receive_obstacle_from_ros(self, data, height):
        if len(data)!=0:
            obstacles = np.array([data[0], data[1], data[2]])
            if height == 0.12:
                self.CylinderbasePosition = np.array([data[0], data[1], 0.06])
                self.CylcolSphereId = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.CylinderRadius, height=height)
                self.obstacle_cylinder = p.createMultiBody(self.Cylindermass, self.CylcolSphereId, self.CylindervisualShapeId, self.CylinderbasePosition, self.CylinderbaseOrientation)
            else: # height = 0.24
                self.CylinderbasePosition = np.array([data[0], data[1], 0.12])
                self.CylcolSphereId = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.CylinderRadius, height=height)
                self.obstacle_cylinder = p.createMultiBody(self.Cylindermass, self.CylcolSphereId, self.CylindervisualShapeId, self.CylinderbasePosition, self.CylinderbaseOrientation)
        return obstacles.copy()
    
    def _is_success(self, achieved_goal):
        d = np.linalg.norm(achieved_goal - self.goal, axis=-1)
        self.re_training = (d < self.distance_threshold).astype(np.float32) and self.safe_flag
        return (d < self.distance_threshold).astype(np.float32)
