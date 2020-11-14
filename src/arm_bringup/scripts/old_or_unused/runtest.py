import gym

from tf2rl.algos.apex import apex_argument, run
from tf2rl.algos.ddpg import DDPG
from tf2rl.misc.target_update_ops import update_target_variables
from arm_env import ArmEnvironment
import rospy


# Prepare env and policy function
class env_fn:
    def __init__(self, env):
        self.env = env

    def __call__(self):
        return self.env


def policy_fn(env, name, memory_capacity=int(1e6),
              gpu=-1, noise_level=0.3):
    return DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=gpu,
        name=name,
        sigma=noise_level,
        batch_size=100,
        lr_actor=0.001,
        lr_critic=0.001,
        actor_units=[400, 300],
        critic_units=[400, 300],
        memory_capacity=memory_capacity)


def get_weights_fn(policy):
    # TODO: Check if following needed
    import tensorflow as tf
    with tf.device(policy.device):
        return [policy.actor.weights,
                policy.critic.weights,
                policy.critic_target.weights]


def set_weights_fn(policy, weights):
    actor_weights, critic_weights, critic_target_weights = weights
    update_target_variables(
        policy.actor.weights, actor_weights, tau=1.)
    update_target_variables(
        policy.critic.weights, critic_weights, tau=1.)
    update_target_variables(
        policy.critic_target.weights, critic_target_weights, tau=1.)
args.env_name

if __name__ == '__main__':
    rospy.init_node("runtest")
    armenv = ArmEnvironment()
    parser = apex_argument()
    parser.add_argument('--env-name', type=str,
                        default="Pendulum-v0")
    parser = DDPG.get_argument(parser)
    args = parser.parse_args()

    run(args, env_fn(armenv), policy_fn, get_weights_fn, set_weights_fn)