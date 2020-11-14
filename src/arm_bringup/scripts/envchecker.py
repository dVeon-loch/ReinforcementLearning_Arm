from stable_baselines.common.env_checker import check_env
from arm_env import ArmEnvironment
import rospy 

rospy.init_node("test")
env = ArmEnvironment()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)