from tf2rl.algos.sac import SAC
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

parser = Trainer.get_argument()
parser = SAC.get_argument(parser)
slow_suffix = "slow" if(slow_step) else ""
#parser.set_defaults(max_steps=5)
if(static_goal and normalise_obs):
    parser.set_defaults(model_dir='model_SAC_static_normed'+slow_suffix)
    parser.set_defaults(logdir='results/SAC_static_normed'+slow_suffix)
elif(static_goal):
    parser.set_defaults(model_dir='model_SAC_static'+slow_suffix)
    parser.set_defaults(logdir='results/SAC_static'+slow_suffix)
elif(normalise_obs):
    parser.set_defaults(model_dir='model_SAC_normed'+slow_suffix)
    parser.set_defaults(logdir='results/SAC_normed'+slow_suffix)
else:
    parser.set_defaults(model_dir='model_SAC'+slow_suffix)
    parser.set_defaults(logdir='results/SAC'+slow_suffix)
#parser.set_defaults(dir_suffix='/home/devon/RL_Arm_noetic/')
parser.set_defaults(normalise_obs=normalise_obs)
parser.set_defaults(save_model_interval=100)
parser.set_defaults(save_summary_interval=100)
parser.set_defaults(test_interval=500)
if(testing):
    parser.set_defaults(test_episodes=num_tests)


parser.set_defaults(gpu=-1)
parser.set_defaults(max_steps=100000000)
parser.set_defaults(auto_alpha=True)

#parser.set_defaults(show_progress=True)
args = parser.parse_args()
rospy.init_node('RL_agent')


env = ArmEnvironment(static_goal=static_goal, slow_step=slow_step, larger_radius= not static_goal, )
test_env = ArmEnvironment(static_goal=static_goal, slow_step=slow_step, larger_radius= not static_goal, testing=testing, gentests=gentests)

policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=0,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha)
trainer = Trainer(policy, env, args, test_env=test_env)
#trainer.evaluate_policy_continuously()

if(testing):
    trainer.evaluate_policy(num_tests)
    print("Arrived "+str(test_env.arrived_counter)+" times out of "+str(num_tests)+" tests.")
else:
    trainer()