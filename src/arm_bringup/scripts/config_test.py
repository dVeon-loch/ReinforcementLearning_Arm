import rospy
from arm_env import ArmEnvironment, AllJoints
import numpy as np

rospy.init_node("test")

env = ArmEnvironment(static_goal=True)
env.reset()
rospy.sleep(1)
print("Goal Distance: ",env.get_goal_distance())


while(True):
    action = input("Enter an action:").split(' ')
    floatarr = []
    for i in action:
        if(i=='0'):
            floatarr.append(0)
        else:
            floatarr.append(float(i))
    action = np.array(floatarr)
    env.step(floatarr)
    print("Goal Distance: ",env.get_goal_distance())


# alljoints.move(pos)
# #rospy.sleep(3)
# env.pause_physics()
# env.unpause_physics()

# env.set_model_config()

# env.pause_physics()

# alljoints.move(np.zeros(4))
# rospy.sleep(1)
# env.unpause_physics()

