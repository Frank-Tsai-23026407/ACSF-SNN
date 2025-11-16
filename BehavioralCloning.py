"""
This script is used to train a simple behavioral cloning agent.
"""
import gym_compat  # Compatibility layer for gym 0.26+
from tools import utils
import argparse
import gym
import torch
import numpy as np
import torch.nn as nn


class BC(nn.Module):
    """
    This class implements a simple Behavioral Cloning model.
    """
    def __init__(self, state_dim, action_dim, max_action):
        """
        Initializes the Behavioral Cloning model.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            max_action (float): The maximum action value.
        """
        super(BC, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        """
        Defines the forward pass of the Behavioral Cloning model.

        Args:
            state: The input state.

        Returns:
            The output action.
        """
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a


def eval_policy(net, env_name, seed, eval_episodes=10):
    """
    Evaluates the policy.

    Args:
        net: The network to evaluate.
        env_name (str): The name of the environment.
        seed (int): The seed for the environment.
        eval_episodes (int, optional): The number of episodes to evaluate for. Defaults to 10.

    Returns:
        (float, torch.Tensor): The mean reward and the reward for each episode.
    """
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    reward_all = torch.zeros(eval_episodes)

    for index in range(eval_episodes):
        avg_reward = 0.
        state, done = eval_env.reset(), False
        while not done:
            input = torch.tensor(state).to(device)
            input = input.to(torch.float32)
            action = net(input).detach().cpu().numpy()
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
        reward_all[index] = avg_reward

    mean_reward = torch.mean(reward_all)
    std = torch.std(reward_all)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {mean_reward:.3f} +- {std:.3f}")
    print("---------------------------------------")
    return mean_reward, reward_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Walker2d-v3")  # OpenAI gym environment name
    parser.add_argument("--seed", default=123, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")  # Prepends name to filename
    parser.add_argument("--buffer", default="DDPG_9853", type=str)
    parser.add_argument("--gpu", default=0, type=int)

    args = parser.parse_args()
    env = gym.make(args.env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    if args.buffer == "DDPG_9853":
        buffer_name = f"{args.buffer_name}_{args.env}_9853"
        replay_buffer.load(f"/data3/ql/buffers/DDPG_Buffer_9853/{buffer_name}")
    elif args.buffer == "DDPG_0":
        buffer_name = f"{args.buffer_name}_{args.env}_0"
        replay_buffer.load(f"/data3/ql/buffers/DDPG_Buffer_0/{buffer_name}")
    elif args.buffer == "TD3":
        buffer_name = f"{args.buffer_name}_TD3_{args.env}_9853"
        replay_buffer.load(f"/data3/ql/buffers/TD3_Buffer/{buffer_name}")
    print("Env:", args.env)
    net = BC(state_dim, action_dim, max_action)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()
    max_step = 1000000
    eval_step = 5000
    evaluations = []
    for _ in range(int(max_step / eval_step)):
        for _ in range(eval_step):
            state, action, next_state, reward, not_done = replay_buffer.sample(100)
            output = net(state.to(device))
            loss = loss_func(output, action.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        eval_reward, _ = eval_policy(net, args.env, args.seed)
        evaluations.append(eval_reward)
    np.save(f"./results/BC_{args.env}_{args.seed}", evaluations)
    eval_policy(net, args.env, args.seed)
