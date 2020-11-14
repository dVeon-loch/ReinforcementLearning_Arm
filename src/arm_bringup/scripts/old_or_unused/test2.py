from tf2rl.td3.policies import TD3Policy
from tf2rl.td3.td3 import TD3
from arm_env import ArmEnvironment
import rospy

rospy.init_node("testing")
env = ArmEnvironment()#gym.make("LunarLanderContinuous-v2")
model = TD3(TD3Policy, env, log_dir_path="td3_results_3")
print(env.action_space)
print(env.observation_space)
model.learn(total_timesteps=1000000)