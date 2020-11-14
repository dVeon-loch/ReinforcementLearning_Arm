from tf2rl.algos.td3 import TD3
from tf2rl.experiments.trainer import Trainer
from arm_env import ArmEnvironment
import rospy
import numpy as np

normalise_obs = False
static_goal = True
testing = True
slow_step = True
num_tests = 50
gentests = False

# conda activate RL_arm_noetic && source devel/setup.bash && tensorboard --logdir /home/devon/RL_Arm_noetic/results

parser = Trainer.get_argument()
parser = TD3.get_argument(parser)
parser.set_defaults(batch_size=100)
slow_suffix = "slow" if(slow_step) else ""

if(static_goal and normalise_obs):
    parser.set_defaults(model_dir='model_TD3_static_normed'+slow_suffix)
    parser.set_defaults(logdir='results/TD3_static_normed'+slow_suffix)
elif(static_goal):
    parser.set_defaults(model_dir='model_TD3_static'+slow_suffix)
    parser.set_defaults(logdir='results/TD3_static'+slow_suffix)
elif(normalise_obs):
    parser.set_defaults(model_dir='model_TD3_normed'+slow_suffix)
    parser.set_defaults(logdir='results/TD3_normed'+slow_suffix)
else:
    parser.set_defaults(model_dir='model_TD3'+slow_suffix)
    parser.set_defaults(logdir='results/TD3'+slow_suffix)

parser.set_defaults(normalise_obs=normalise_obs)
parser.set_defaults(save_model_interval=100)
parser.set_defaults(save_summary_interval=100)
parser.set_defaults(test_interval=500)
parser.set_defaults(max_steps=100000000)
parser.set_defaults(gpu=-1)
if(testing):
    parser.set_defaults(test_episodes=num_tests)


#parser.set_defaults(show_progress=True)
args = parser.parse_args()
rospy.init_node('RL_agent')

env = ArmEnvironment(static_goal=static_goal, slow_step=slow_step, larger_radius= not static_goal, )
test_env = ArmEnvironment(static_goal=static_goal, slow_step=slow_step, larger_radius= not static_goal, testing=testing, gentests=gentests)

print("Obs shape:",env.observation_space.shape)
print("action shape:",env.action_space.high.size)
policy = TD3(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=args.gpu,  # Run on CPU. If you want to run on GPU, specify GPU number
    memory_capacity=args.memory_capacity,
    max_action=env.action_space.high[0],
    batch_size=args.batch_size,
    n_warmup=0)
trainer = Trainer(policy, env, args, test_env=test_env)

if(testing):
    trainer.evaluate_policy(num_tests)
    print("Arrived "+str(test_env.arrived_counter)+" times out of "+str(num_tests)+" tests.")
else:
    trainer()