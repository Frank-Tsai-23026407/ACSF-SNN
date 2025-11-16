"""
This script contains utility functions and classes for the project.
"""
import numpy as np
import torch


class ReplayBuffer(object):
	"""
	A replay buffer for storing and sampling transitions.

	Args:
		state_dim (int): The dimension of the state space.
		action_dim (int): The dimension of the action space.
		device: The device to run the models on.
		max_size (int, optional): The maximum size of the buffer. Defaults to int(1e6).
	"""
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device

	def add(self, state, action, next_state, reward, done):
		"""
		Adds a transition to the buffer.

		Args:
			state: The state.
			action: The action.
			next_state: The next state.
			reward: The reward.
			done (bool): Whether the episode is done.
		"""
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		"""
		Samples a batch of transitions from the buffer.

		Args:
			batch_size (int): The batch size.

		Returns:
			(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): A tuple of tensors containing the states, actions, next states, rewards, and not_dones.
		"""
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def AESample(self):
		"""
		Samples all the states and actions from the buffer.

		Returns:
			(torch.Tensor, torch.Tensor): A tuple of tensors containing the states and actions.
		"""
		return (torch.FloatTensor(self.state).to(self.device),
			torch.FloatTensor(self.action).to(self.device))

	def convert_D4RL(self, dataset):
		"""
		Converts a D4RL dataset to the replay buffer format.

		Args:
			dataset: The D4RL dataset.
		"""
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1, 1)
		self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
		self.size = self.state.shape[0]

	def save(self, save_folder):
		"""
		Saves the replay buffer to a folder.

		Args:
			save_folder (str): The folder to save the buffer to.
		"""
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)

	def load(self, save_folder, size=-1):
		"""
		Loads a replay buffer from a folder.

		Args:
			save_folder (str): The folder to load the buffer from.
			size (int, optional): The size of the buffer to load. Defaults to -1.
		"""
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

	def normalize_states(self, eps=1e-3):
		"""
		Normalizes the states in the buffer.

		Args:
			eps (float, optional): A small value to avoid division by zero. Defaults to 1e-3.

		Returns:
			(np.ndarray, np.ndarray): The mean and standard deviation of the states.
		"""
		mean = self.state.mean(0, keepdims=True)
		std = self.state.std(0, keepdims=True) + eps
		self.state = (self.state - mean) / std
		self.next_state = (self.next_state - mean) / std
		return mean, std
