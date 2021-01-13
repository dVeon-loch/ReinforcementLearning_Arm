# Replacing Conventional Motion Planners/IK solvers with Reinforcement Learning Agents
Application of reinforcement learning to a simulated ROS based robotic arm, to allow it to move to a desired point in 3D space. This repo differs from the previous one in that it is redone using ROS Noetic and Python 3.

## Results
These results are taken from the policy learned by the [Soft Actor-Critic](https://arxiv.org/abs/1812.05905 "Soft Actor-Critic Algorithms and Applications") algorithm. Algorithms were implemented using [ keiohta /
tf2rl ](https://github.com/keiohta/tf2rl#tf2rl "TF2RL Github Repo")
### Before and After (Static Goal)

<img src="media\before.gif" width="250" height="250"/><img src="media\after.gif" width="250" height="250"/>

### Before and After (Moving Goal)

<img src="media\before_moving.gif" width="250" height="250"/><img src="media\after_moving.gif" width="250" height="250"/>

## Info

All scripts for testing/training/collecting data for graphs etc. can be found in [src/arm_bringup/scripts](https://github.com/dVeon-loch/EEE4022_RL_Arm_noetic/tree/master/src/arm_bringup/scripts).
All models that represent those mentioned in the report are contained in the separate model folders. Note the suffixes. In order to run the various algorithms the "insert_algorithm_acronym"_train_test.py files must be used. In order to train, set the testing variable to False. In order to train with a static goal, set static_goal to True (and the opposite for a moving goal). To set number of test episodes, the num_tests variable is used.

The "slow" suffix refers to the hardcoded delay that was added to deal with the limitations imposed by Gazebo. The delay can be disabled by setting slow_step to False in the train/test code, however this will be much less stable.

Important parameters such as the acceptable goal radius, max sim time per episode and others must be set in arm_env.py, and can be found at the top of the ArmEnvironment __init__ function.

## Installation and Running Instructions

[Install ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu). Also:

`echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc`

`source ~/.bashrc`

[Install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)


The system has been tested on a virtualbox VM running Ubuntu 20.04 but the performance of such a setup is limited, dual-booting is recommended. As the project went through a number of iterations the conda environment.yml file may contain some unneeded packages. However, to be safe it is recommended to install the environment file as follows:

conda env create -f environment.yml

conda activate RL_arm_noetic
`conda env list`

This env also contains the necessary dependencies to run ROS commands. Thus, all ROS commands should be run in a terminal that has the conda env active. 

Build ROS package:

`catkin_make`

To activate the ROS core (recommended to do it seperately):

`roscore`

In a new terminal window launch the simulation:

`roslaunch arm_bringup sim_bringup.launch world:=empty`

Navigate to the scripts directory before running any scripts.

`cd ReinforcementLearning_Arm/src/arm_bringup/scripts`

To run scripts simply run the python file corresponding to the algorithm you are training/testing:

e.g. `python sac_train_test.py`

### Note: All algorithms are set up by default to test for 50 iterations using the learned models. To start from fresh you can delete/move the model folders elsewhere and set the algo_train_test.py file to training with either static goal set to true or false.

## Viewing Results

To view results in real time, you can use tensorboard. To open tensorboard enter the following into a new terminal and click on the given link:

`tensorboard --logdir /path/to/ReinforcementLearning_Arm/results`  (copy the path of the results folder and enter it here)

A custom graphing script, `results_graphing.py` was created to join all of the data together for each run for each algorithm, this is also in the scripts folder but it will need to be changed if you wish to use it with new data. Running this script as it is currently will display the results of the static goal training for all three algorithms. 

