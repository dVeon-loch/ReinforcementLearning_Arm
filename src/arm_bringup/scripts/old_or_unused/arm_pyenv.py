from arm_env import AllJoints
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

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
import tf
import tf2_ros
from scipy.spatial import distance

import numpy as np
import rospy
import os
arm_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'arm_description',
                              'urdf', 'arm_gazebo.urdf')

sphere_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'arm_description',
                              'urdf', 'ball.urdf')
class ArmEnv(py_environment.PyEnvironment):
    def __init__(self):
        self.goal_distance_threshold = 0.05
        self.distance_reward_coeff = 1

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=[-0.5,-0.5,-0.5,-0.5], maximum=[-0.5,-0.5,-0.5,0.5], name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(7,), dtype=np.float32, minimum=[-1.5,-1.5,-1.5,-2.5,-0.4,-0.4,0], maximum=[1.5,1.5,1.5,2.5,0.4,0.4,0.4], name='observation')
        self.joint_angles = np.zeros(4)
        self._episode_ended = False
        self._reached_goal = False

        self.joint_names = ['plat_joint','shoulder_joint','forearm_joint','wrist_joint']
        self.all_joints = AllJoints(self.joint_names)
        # Physics Service Proxies
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        # Sim Reset Proxy
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # Controller Loading Service
        self.load_controller_proxy = rospy.ServiceProxy('/arm/controller_manager/load_controller', LoadController)
        self.joint_state_controller_load = LoadControllerRequest()
        self.joint_state_controller_load.name = 'joint_state_controller'
        self.arm_controller_load = LoadControllerRequest()
        self.arm_controller_load.name = 'arm_controller'
        # Controller Switching Service
        self.switch_controller_proxy = rospy.ServiceProxy('/arm/controller_manager/switch_controller', SwitchController)
        self.switch_controller = SwitchControllerRequest()
        self.switch_controller.start_controllers.append('joint_state_controller')
        self.switch_controller.start_controllers.append('arm_controller')
        self.switch_controller.strictness = 2
        # Model Deleting and Spawning Services, Defining Arm Model Params
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.arm_urdf = open(arm_model_dir, "r").read()
        self.arm_model = SpawnModelRequest()
        self.arm_model.model_name = 'arm'  # the same with sdf name
        self.arm_model.model_xml = self.arm_urdf
        self.arm_model.robot_namespace = 'arm'
        self.initial_pose = Pose()
        self.initial_pose.position.z = 0.0305
        self.arm_model.initial_pose = self.initial_pose 
        self.arm_model.reference_frame = 'world'
        # Defining Sphere (Goal Position Indicator) Model Params
        self.sphere_urdf = open(sphere_dir, "r").read()
        self.sphere = SpawnModelRequest()
        self.sphere.model_name = 'simple_ball'  # the same with sdf name
        self.sphere.model_xml = self.sphere_urdf
        self.sphere.robot_namespace = 'arm'
        self.sphere_initial_pose = Pose()
        # self.sphere_initial_pose.position.x = self.goal_pos[0]
        # self.sphere_initial_pose.position.y = self.goal_pos[1]
        # self.sphere_initial_pose.position.z = self.goal_pos[2]
        # self.sphere.initial_pose = self.sphere_initial_pose 
        self.sphere.reference_frame = 'world'
        self.set_new_goal()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.joint_pos_high = np.array([1.5, 1.5, 1.5, 2.5])
        self.joint_pos_low = np.array([-1.5, -1.5, -1.5, -2.5])
        self.joint_pos_range = self.joint_pos_high-self.joint_pos_low
        self._state = np.zeros(7)
        print("INIT SHAPE BOIII",np.shape(self._state))


        self.joint_state_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_state_subscriber_callback)
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        action = np.array(action)
        joint_pos = np.clip(self.joint_angles + action,a_min=self.joint_pos_low,a_max=self.joint_pos_high)
        self.all_joints.move(joint_pos)
        #rospy.sleep(0.8)
        reward = self.get_reward()

        self.get_state()
        return ts.transition(self._state, reward=reward, discount=1.0)
    def _reset(self):
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('arm')
        self.del_model('simple_ball')

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
        #rospy.sleep(1)
        self._episode_ended = False
        self.get_state()

        return ts.restart(np.array(self._state, dtype=np.float))

    def set_new_goal(self):
        rospy.loginfo("Defining a goal position...")
        while(True):
            x_y = np.random.uniform(low = -0.4, high = 0.4, size = 2)
            z = np.random.uniform(low = 0, high = 0.4, size = 1)
            self.goal_pos = np.concatenate([x_y,z],axis=0) #np.array([-0.13242582 , 0.29086919 , 0.20275278])
            if(np.linalg.norm(self.goal_pos)<0.5):
                break
        rospy.loginfo("Goal position defined")
        rospy.loginfo("Goal position: "+str(self.goal_pos))
        # self.sphere_urdf = open(sphere_dir, "r").read()
        # self.sphere = SpawnModelRequest()
        # self.sphere.model_name = 'simple_ball'  # the same with sdf name
        # self.sphere.model_xml = self.sphere_urdf
        # self.sphere.robot_namespace = 'arm'
        # self.sphere_initial_pose = Pose()
        self.sphere_initial_pose.position.x = self.goal_pos[0]
        self.sphere_initial_pose.position.y = self.goal_pos[1]
        self.sphere_initial_pose.position.z = self.goal_pos[2]
        self.sphere.initial_pose = self.sphere_initial_pose 
        # self.sphere.reference_frame = 'world'
        self.spawn_model(self.sphere)
    
    def normalize_joint_state(self,joint_angles):
        min = -2.5
        max = 2.5
        ave = (min+max)/2.0
        rge = (max-min)/2.0
        normed_joint_pos = (joint_angles - average) / rge
        return normed_joint_pos

    def normalize_goal_pos(self, goal_pos):
        min = np.array([-0.4, -0.4, 0])
        max = np.array([0.4, 0.4, 0.4])
        ave = (min+max)/2.0
        rge = (max-min)/2.0
        normed_goal_pos = (goal_pos - average) / rge
        return normed_goal_pos
    
    def get_state(self):
        # TODO investigate normalizing, e.g. if tf-agents normalizes automatically
        self._state = np.concatenate([self.joint_angles,self.goal_pos]).reshape(7,)
        self._state = self._state.astype(np.float32, copy=False)
        print("SHAPE BOIII",np.shape(self._state))
        # if(self.get_goal_distance()<=self.goal_distance_threshold):
        #     self._reached_goal = True

    def joint_state_subscriber_callback(self, joint_state):
        self.joint_angles = joint_state
        
    def get_goal_distance(self):
        try:
            trans = self.tf_buffer.lookup_transform('world', 'dummy_eef', rospy.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            trans = np.array([x,y,z])
            #print("EEF Position: {}".format(trans))
            
            goal_distance = distance.euclidean(trans,self.goal_pos)
            #print("Goal is at: {} \n".format(np.array2string(self.goal_pos)))
            return goal_distance
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("tf lookupTransform error")

    def get_reward(self):
        reward = -1.0*self.distance_reward_coeff*self.get_goal_distance()
        if(self.get_goal_distance()<=self.goal_distance_threshold):
            reward += 100.0
        return reward

    def spawn_model(self, model):
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            self.spawn_model_proxy(model.model_name, model.model_xml, model.robot_namespace, model.initial_pose, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        self.unpause_physics()
    
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