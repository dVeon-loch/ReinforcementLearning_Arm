from arm_pyenv import ArmEnv
import numpy as np
import rospy

from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

from tf_agents.networks import actor_distribution_network, value_network

import base64
#import imageio
#import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

tempdir = tempfile.gettempdir()


fc_layer_params = (100,)
importance_ratio_clipping
lambda_value 

train_timed_env = wrappers.TimeLimit(
    ArmEnv(),
    1000
)

eval_timed_env = wrappers.TimeLimit(
    ArmEnv(),
    1000
)

train_env = tf_py_environment(train_timed_env)

eval_env = tf_py_environment(eval_timed_env)

observation_tensor_spec, action_spec, time_step_tensor_spec = (
spec_utils.get_tensor_specs(train_env))
normalized_observation_tensor_spec = tf.nest.map_structure(
lambda s: tf.TensorSpec(
dtype=tf.float32, shape=s.shape, name=s.name
),
observation_tensor_spec
)

actor_net = actor_distribution_network.ActorDistributionNetwork(
normalized_observation_tensor_spec, ...)
value_net = value_network.ValueNetwork(
normalized_observation_tensor_spec, ...)
# Note that the agent still uses the original time_step_tensor_spec
# from the environment.
agent = ppo_clip_agent.PPOClipAgent(
time_step_tensor_spec, action_spec, actor_net, value_net, ...)

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}


