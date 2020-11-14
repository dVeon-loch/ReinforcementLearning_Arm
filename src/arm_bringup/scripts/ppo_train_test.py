from tf2rl.algos.ppo import PPO
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim
from arm_env import ArmEnvironment
import rospy
import numpy as np

normalise_obs = False
static_goal = True
testing = True
slow_step = True
num_tests = 50
gentests = False


rospy.init_node("RL_agent")
parser = OnPolicyTrainer.get_argument()
parser = PPO.get_argument(parser)

slow_suffix = "_slow" if(slow_step) else ""
#parser.set_defaults(max_steps=5)
if(static_goal and normalise_obs):
    parser.set_defaults(model_dir='model_PPO_static_normed'+slow_suffix)
    parser.set_defaults(logdir='results/PPO_static_normed'+slow_suffix)
elif(static_goal):
    parser.set_defaults(model_dir='model_PPO_static'+slow_suffix)
    parser.set_defaults(logdir='results/PPO_static'+slow_suffix)
elif(normalise_obs):
    parser.set_defaults(model_dir='model_PPO_normed'+slow_suffix)
    parser.set_defaults(logdir='results/PPO_normed'+slow_suffix)
else:
    parser.set_defaults(model_dir='model_PPO'+slow_suffix)
    parser.set_defaults(logdir='results/PPO'+slow_suffix)
parser.set_defaults(normalise_obs=normalise_obs)
parser.set_defaults(save_model_interval=100)
parser.set_defaults(save_summary_interval=100)
parser.set_defaults(test_interval=500)
if(testing):
    parser.set_defaults(test_episodes=num_tests)

#parser.set_defaults(horizon=1024)
#parser.set_defaults(batch_size=512)
parser.set_defaults(gpu=-1)
parser.set_defaults(max_steps=100000000)
parser.set_defaults(n_warmup=0)
#parser.set_defaults(enable_gae=True)
args = parser.parse_args()

env = ArmEnvironment(static_goal=static_goal, slow_step=slow_step, larger_radius= not static_goal, )
test_env = ArmEnvironment(static_goal=static_goal, slow_step=slow_step, larger_radius= not static_goal, testing=testing, gentests=gentests)


policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=(64, 64),
        critic_units=(64, 64),
        n_epoch=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_activation_actor="tanh",
        hidden_activation_critic="tanh",
        discount=0.99,
        lam=0.95,
        entropy_coef=0.001,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)

if(testing):
    trainer.evaluate_policy(num_tests)
    print("Arrived "+str(test_env.arrived_counter)+" times out of "+str(num_tests)+" tests.")
else:
    trainer()