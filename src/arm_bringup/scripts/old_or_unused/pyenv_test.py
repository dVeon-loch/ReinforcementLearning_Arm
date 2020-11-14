import tensorflow as tf
import numpy as np
import rospy
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from arm_pyenv import ArmEnv
# source devel/setup.bash
# roslaunch arm_bringup sim_bringup.launch world:=empty
rospy.init_node("test")
tf.compat.v1.enable_v2_behavior()

environment = ArmEnv()

timed_env = wrappers.TimeLimit(
    environment,
    900
)
utils.validate_py_environment(timed_env, episodes=5)
print('action_spec:', environment.action_spec())
print('time_step_spec:', environment.time_step_spec())
print('time_step_spec.observation:', environment.time_step_spec().observation)
print('time_step_spec.step_type:', environment.time_step_spec().step_type)
print('time_step_spec.discount:', environment.time_step_spec().discount)
print('time_step_spec.reward:', environment.time_step_spec().reward)

