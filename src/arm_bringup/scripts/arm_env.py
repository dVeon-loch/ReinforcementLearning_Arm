import os
import gym
from gym import spaces
import rospy
import actionlib
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState, FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController, LoadControllerRequest
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel, SpawnModelRequest
from rosgraph_msgs.msg import Clock
import tf
import tf2_ros
from scipy.spatial import distance

import numpy as np
import time

arm_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'arm_description',
                              'urdf', 'arm_gazebo.urdf')

sphere_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'arm_description',
                              'urdf', 'ball.urdf')


class AllJoints:
    def __init__(self,joint_names):
        self.action_server_client = actionlib.SimpleActionClient('arm/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
       # rospy.loginfo('Waiting for joint trajectory action')
        self.action_server_client.wait_for_server()
       # rospy.loginfo('Found joint trajectory action')
        self.jtp = rospy.Publisher('arm/arm_controller/command', JointTrajectory, queue_size=1)
        self.joint_names = joint_names
        self.jtp_zeros = np.zeros(len(joint_names))


    def move(self, pos):
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(1.0/60.0)
        msg.goal.trajectory.points.append(point)
        self.action_server_client.send_goal(msg.goal)
        return True

    def move_jtp(self, pos):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        point.time_from_start = rospy.Duration(1.0/60.0)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

    def reset_move(self, pos):
        jtp_msg = JointTrajectory()
        self.jtp.publish(jtp_msg)
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(0.0001)
        msg.goal.trajectory.points.append(point)
        self.action_server_client.send_goal(msg.goal)

    def reset_move_jtp(self):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.jtp_zeros
        point.time_from_start = rospy.Duration(1.0/60.0)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

class ArmEnvironment(gym.Env):
    """ Set arguments to desired values for static/moving goals, training/testing. 
    gentests refers to the generalisation tests for both the static goal and moving goal tasks."""
    def __init__(self, static_goal, slow_step=False, larger_radius=False, testing=False, gentests=False):
        
        self.arrived_counter = 0 # To store number of successful attempts during testing phase
        self.gentests = gentests
        self.testing = testing
        self.max_sim_time = 15
        if(larger_radius):
            self.goal_radius = 0.06 # Sphere radius in ball.urdf is slightly bigger to improve visuals
        else:
            self.goal_radius = 0.05
        self.distance_reward_coeff = 5
        self.static_goal_pos = np.array([0.2,0.1,0.15])
        self.static_goal = static_goal
        self.slow_step = slow_step
        self.eef_pos = np.zeros(3)
        self.goal_pos_list=np.array([[0.2,0.1,0.15],[-0.2,0.1,0.15],[-0.2,-0.1,0.15],[0.3,0.05,0.2],[0.15195154, 0.12716517, 0.10791227],[-0.2443342 , -0.26623122,  0.01693387],[ 0.20605877, -0.16932273,  0.18279588],[-0.25533529, -0.11042604,  0.14230928],[ 0.3006918 , -0.00099998,  0.16692274],[-0.07770072,  0.0790152 ,  0.29283088],[ 0.18422977, -0.16340892,  0.27168277],[ 0.28851676, -0.0598984 ,  0.20165042],[-0.09136536, -0.00854591,  0.20377172]])

        self.zero = np.array([0,0,0,0])
        self.num_joints = 4
        self.observation_space = spaces.Box(np.array([-1.5,-1.5,-1.5,-2.5,-0.4,-0.4,0]), np.array([1.5,1.5,1.5,2.5,0.4,0.4,0.4]))#(self.num_joints + 3,)
        self.action_space = spaces.Box(np.array([-0.2,-0.2,-0.2,-0.2]), np.array([0.2,0.2,0.2,0.2]))
        self.joint_names = ['plat_joint','shoulder_joint','forearm_joint','wrist_joint']
        self.all_joints = AllJoints(self.joint_names)
        self.starting_pos = np.array([0, 0, 0, 0])
        self.last_goal_distance = 0
        rospy.loginfo("Defining a goal position...")
        if(self.static_goal):
            self.goal_pos = self.static_goal_pos
        elif(not self.static_goal):
            index = np.random.randint(low=0,high=len(self.goal_pos_list)-1)
            self.goal_pos = self.goal_pos_list[index]
        elif(self.moving_gentests):
            while(True):
                x_y = np.random.uniform(low = -0.4, high = 0.4, size = 2)
                z = np.random.uniform(low = 0, high = 0.4, size = 1)
                self.goal_pos = np.concatenate([x_y,z],axis=0)
                if(np.linalg.norm(self.goal_pos)<0.4):
                    break
        rospy.loginfo("Goal position defined")
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics',Empty, persistent=False)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics',Empty, persistent=False)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty, persistent=False)
   
        self.load_controller_proxy = rospy.ServiceProxy('/arm/controller_manager/load_controller', LoadController, persistent=False)
        self.joint_state_controller_load = LoadControllerRequest()
        self.joint_state_controller_load.name = 'joint_state_controller'
        self.arm_controller_load = LoadControllerRequest()
        self.arm_controller_load.name = 'arm_controller'

        self.switch_controller_proxy = rospy.ServiceProxy('/arm/controller_manager/switch_controller', SwitchController, persistent=False)
        self.switch_controller = SwitchControllerRequest()
        self.switch_controller.start_controllers.append('joint_state_controller')
        self.switch_controller.start_controllers.append('arm_controller')
        self.switch_controller.strictness = 2

        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel, persistent=False)
        self.arm_urdf = open(arm_model_dir, "r").read()
        self.arm_model = SpawnModelRequest()
        self.arm_model.model_name = 'arm'  # the same with sdf name
        self.arm_model.model_xml = self.arm_urdf
        self.arm_model.robot_namespace = 'arm'
        self.initial_pose = Pose()
        self.initial_pose.position.z = 0.0305
        self.arm_model.initial_pose = self.initial_pose 
        self.arm_model.reference_frame = 'world'

        self.state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, persistent=False)
        self.config_request = SetModelStateRequest()
        self.config_request.model_state.model_name = 'simple_ball'
        sphere_pose = Pose()
        sphere_pose.position.x = 1
        sphere_pose.position.y = 2
        sphere_pose.position.z = 3
        self.config_request.model_state.pose = sphere_pose
        self.config_request.model_state.reference_frame = 'world'
        self.sphere_urdf = open(sphere_dir, "r").read()
        self.sphere = SpawnModelRequest()
        self.sphere.model_name = 'simple_ball'  # the same with sdf name
        self.sphere.model_xml = self.sphere_urdf
        self.sphere.robot_namespace = 'arm'
        self.sphere_initial_pose = Pose()
        self.sphere_initial_pose.position.x = self.goal_pos[0]
        self.sphere_initial_pose.position.y = self.goal_pos[1]
        self.sphere_initial_pose.position.z = self.goal_pos[2]
        self.sphere.initial_pose = self.sphere_initial_pose 
        self.sphere.reference_frame = 'world'
        self.unpause_physics()
        self.spawn_model(self.sphere)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Defining joint limits
        self.joint_pos_high = np.array([1.5, 1.5, 1.5, 2.5]) 
        self.joint_pos_low = np.array([-1.5, -1.5, -1.5, -2.5])
        self.joint_pos_range = self.joint_pos_high-self.joint_pos_low
        self.joint_pos_mid = self.joint_pos_range/2.0
        self.joint_pos = np.zeros(4)
        self.joint_state = np.zeros(self.num_joints)
        self.joint_state_subscriber = rospy.Subscriber('arm/arm_controller/state', JointTrajectoryControllerState, self.joint_state_subscriber_callback, queue_size=1)
        self.clock_subscriber = rospy.Subscriber('/clock', Clock, self.clock_subscriber_callback, queue_size=1)
        self.hit_floor = False
    
    def step(self, action):
        action = np.array(action)

        self.joint_pos = np.clip(self.joint_state + action,a_min=self.joint_pos_low,a_max=self.joint_pos_high)
        self.all_joints.move(self.joint_pos)
        if(self.slow_step):
            rospy.sleep(0.35)
        else:
            rospy.sleep(0.01)
        (joint_angles, time_runout, arrived) = self.get_state()
        reward = self.get_reward(time_runout, arrived)
        if(time_runout or arrived):
            done = True
        else:
            done = False
        state = np.concatenate([joint_angles,self.goal_pos])
        state = np.asarray(state, dtype = np.float32)

        return state, reward, done, {}
   
    def reset(self):

        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('arm')
        #self.del_model('simple_ball')
        rospy.sleep(0.5)
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        self.spawn_model(self.arm_model)

        rospy.wait_for_service('arm/controller_manager/load_controller')
        try:
            self.load_controller_proxy(self.joint_state_controller_load)
        except (rospy.ServiceException) as e:
            print('arm/controller_manager/load_controller service call failed')
        
        rospy.wait_for_service('arm/controller_manager/load_controller')
        try:
            self.load_controller_proxy(self.arm_controller_load)
        except (rospy.ServiceException) as e:
            print('arm/controller_manager/load_controller service call failed')

        rospy.wait_for_service('arm/controller_manager/switch_controller')
        try:
            self.switch_controller_proxy(self.switch_controller)
        except (rospy.ServiceException) as e:
            print('arm/controller_manager/switch_controller service call failed')

        self.set_new_goal()
        rospy.sleep(1)
       
        done = False
        joint_angles, time_runout, arrived = self.get_state()
       
        state = np.concatenate([joint_angles,self.goal_pos])
        state = np.asarray(state, dtype = np.float32)
        self.joint_state = np.zeros(4)
        self.joint_pos = np.zeros(4)

        return state
    
    def get_state(self):
        self.hit_floor = False
        joint_angles = self.joint_state
        current_sim_time = self.current_time
        trans = self.tf_buffer.lookup_transform('world', 'wrist_link', rospy.Time())
        rospy.sleep(0.1)
        end = self.tf_buffer.lookup_transform('world', 'dummy_eef', rospy.Time())
        if(trans.transform.translation.z<=0.02 or end.transform.translation.z<=0.01): 
            self.hit_floor=True
        if(self.get_goal_distance()<=self.goal_radius):
            arrived = True
        else: 
            arrived=False
        if(self.hit_floor or current_sim_time>=self.max_sim_time):
            time_runout = True
            #print("ran out of time or hit floor")
        else:
            time_runout=False
        return joint_angles, time_runout, arrived
    
    def get_reward(self, time_runout, arrive):
        """Returns reward (float) for the current state, 
        taking into account whether or not the arm has reached the goal or hit the floor"""
        reward = -1*self.distance_reward_coeff*self.get_goal_distance()
        if(self.hit_floor):
            reward = reward - 50.0
        if(time_runout and not arrive):
            reward = reward - 25.0
        if(arrive):
            self.arrived_counter += 1
            print("Arrived at goal")
            reward += 25.0
        return reward
    
    def set_model_state(self, set_state_request):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.state_proxy(set_state_request)
            return True
        except rospy.ServiceException as e:
            print('/gazebo/set_model_state service call failed')
            return False

    def clock_subscriber_callback(self, clock):
        self.current_time = clock.clock.secs

    def joint_state_subscriber_callback(self, joint_state):
        self.joint_state = np.array(joint_state.actual.positions)

    def pause_physics(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics_proxy()
            return True
        except rospy.ServiceException as e:
            print('/gazebo/pause_physics service call failed')
            return False

    def unpause_physics(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            print('/gazebo/unpause_physics service call failed')

    def get_goal_distance(self):
        """Gets current eef position using a tf lookup_transform, 
        and returns euclidean distance from eef to goal position"""
        try:
            trans = self.tf_buffer.lookup_transform('world', 'dummy_eef', rospy.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            trans = np.array([x,y,z])
            self.eef_pos = trans            
            goal_distance = distance.euclidean(trans,self.goal_pos)
            return goal_distance
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("tf lookupTransform error")

    def set_new_goal(self):
        """ Calculates new goal position within range and parameters set, 
        and move visual indicator to that position with the set model state gazebo service"""
        rospy.loginfo("Defining a goal position...")
        
        if(self.static_goal and self.testing and self.gentests):
            x_y = np.random.uniform(low = -0.1, high = 0.1, size = 2)
            z = np.random.uniform(low = -0.05, high = 0.1, size = 1)
            perturb = np.concatenate([x_y,z],axis=0)
            self.goal_pos = self.static_goal_pos
            self.goal_pos = self.goal_pos + perturb
        elif(self.static_goal and self.testing):
            self.goal_pos = self.static_goal_pos
        elif(self.static_goal):
            self.goal_pos = self.static_goal_pos
        elif(self.testing and self.gentests):
            while(True):
                x_y = np.random.uniform(low = -0.4, high = 0.4, size = 2)
                z = np.random.uniform(low = 0, high = 0.4, size = 1)
                self.goal_pos = np.concatenate([x_y,z],axis=0) 
                if(np.linalg.norm(self.goal_pos)<0.4 and np.linalg.norm(self.goal_pos)>0.1):
                    break
        else:
            index = np.random.randint(low=0,high=len(self.goal_pos_list)-1)
            self.goal_pos = self.goal_pos_list[index]
            
        sphere_pose = Pose()
        sphere_pose.position.x = self.goal_pos[0]
        sphere_pose.position.y = self.goal_pos[1]
        sphere_pose.position.z = self.goal_pos[2]
        self.config_request.model_state.pose = sphere_pose
        self.set_model_state(self.config_request)

    def spawn_model(self, model):
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            self.spawn_model_proxy(model.model_name, model.model_xml, model.robot_namespace, model.initial_pose, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        self.unpause_physics()

    def seed(self,seed):
        """Needed for tf2rl, is used in openAI gym environments but is not needed here"""
        pass

    

