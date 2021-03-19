import numpy as np
import gym
import argparse
from tensorboardX import SummaryWriter
from ddpg_model import DDPGModel
from ddpg_agent import DDPGAgent
from ddpg import DDPG
from replay_buffer import ReplayBuffer

WARMUP_STEPS = 1e4
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 100
GAMMA = 0.99
DECAY = 0.995
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise


# Run episode for training
def run_train_episode(agent, env, replay_buffer):
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0

    while not done:
        episode_steps += 1
        # Select action randomly or according to policy
        if replay_buffer.cur_size < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim) * max_action
        else:
            action = agent.sample(obs)

        # Perform action
        next_obs, reward, done, _ = env.step(action)
        terminal = float(done) if episode_steps < env._max_episode_steps else 0

        # Store data in replay memory
        replay_buffer.store(obs, action, reward, next_obs, terminal)
        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data
        if replay_buffer.cur_size >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = replay_buffer.sample(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return episode_reward, episode_steps


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    print("------------------ DDPG ---------------------")
    print('Env: {}, Seed: {}'.format(args.env, args.seed))
    print("---------------------------------------------")

    env = gym.make(args.env)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize model, algorithm, agent, replay_memory
    model = DDPGModel(obs_dim, action_dim, max_action)
    algorithm = DDPG(
        model, gamma=GAMMA, decay=DECAY, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = DDPGAgent(algorithm, action_dim, expl_noise=EXPL_NOISE)
    replay_buffer = ReplayBuffer(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    test_flag = 0

    writer = SummaryWriter()
    while total_steps < args.train_total_steps:
        # Train episode
        episode_reward, episode_steps = run_train_episode(agent, env, replay_buffer)
        total_steps += episode_steps

        writer.add_scalar('train/episode_reward', episode_reward,
                               total_steps)
        print('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
            writer.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            print('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="HalfCheetah-v1", help='OpenAI gym environment name')
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument(
        "--train_total_steps",
        default=5e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
