# Replacing Conventional Motion Planners/IK solvers with Reinforcement Learning Agents
Application of reinforcement learning to a simulated ROS based robotic arm, to allow it to move to a desired point in 3D space. This repo differs from the previous one in that it is redone using ROS Noetic and Python 3.

## Results
These results are taken from the policy learned by the [Soft Actor-Critic](https://arxiv.org/abs/1812.05905 "Soft Actor-Critic Algorithms and Applications") algorithm. Algorithms were implemented using [ keiohta /
tf2rl ](https://github.com/keiohta/tf2rl#tf2rl "TF2RL Github Repo")
### Before and After (Static Goal)

<img src="before.gif" width="250" height="250"/><img src="after.gif" width="250" height="250"/>

### Before and After (Moving Goal)

<img src="before_moving.gif" width="250" height="250"/><img src="after_moving.gif" width="250" height="250"/>

## Info

All scripts for testing/training/collecting data for graphs etc. can be found in [src/arm_bringup/scripts](https://github.com/dVeon-loch/EEE4022_RL_Arm_noetic/tree/master/src/arm_bringup/scripts).
All models that represent those mentioned in the report are contained in the separate model folders. Note the suffixes. In order to run the various algorithms the "insert_algorithm_acronym"_train_test.py files must be used. In order to train, set the testing variable to False. In order to train with a static goal, set static_goal to True (and the opposite for a moving goal). To set number of test episodes, the num_tests variable is used.

The "slow" suffix refers to the hardcoded delay that was added to deal with the limitations imposed by Gazebo. The delay can be disabled by setting slow_step to False in the train/test code, however this will be much less stable.

Important parameters such as the acceptable goal radius, max sim time per episode and others must be set in arm_env.py, and can be found at the top of the ArmEnvironment __init__ function.
