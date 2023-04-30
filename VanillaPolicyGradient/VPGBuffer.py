import numpy as np
from typing import Tuple, Dict, SupportsFloat

import torch

import utils


class VPGBuffer:
	def __init__(
			self,
			observation_dim: [int, Tuple],
			action_dim: [int, Tuple],
			size: int,
			gamma: float,
			lam: float,
	):
		# First of all, we need to store agent-environment interactions,
		# that is, trajectories {observation, action, reward}.
		self.observation_buf = np.zeros(
			utils.combined_shape(size, observation_dim),
			dtype=np.float32,
		)
		self.action_buf = np.zeros(
			utils.combined_shape(size, action_dim),
			dtype=np.float32,
		)
		self.reward_buf = np.zeros(size, dtype=np.float32)
		
		# To update policy, we need to store each time step's
		# advantage function and log_prob of the selected action.
		self.advantage_buf = np.zeros(size, dtype=np.float32)
		self.log_prob_buf = np.zeros(size, dtype=np.float32)
	
		# To compute advantage function, we need to compute delta. And to compute delta,
		# we need to compute each time steps' value estimate. So we need to update value_function.
		# To update value_function, we need to store each time step's value estimate and return (reward-to-go).
		self.value_buf = np.zeros(size, dtype=np.float32)
		self.return_buf = np.zeros(size, dtype=np.float32)
		
		# gamma is the discount factor. lam is the parameter in GAE.
		# They are both used to compute advantage estimate.
		self.gamma = gamma
		self.lam = lam
		
		# ptr and path_start_idx are used in storing and retrieving data from buffers.
		self.ptr, self.path_start_idx, self.max_size = 0, 0, size
	
	def store(
			self,
			observation: np.ndarray,
			action: np.ndarray,
			reward: [float, SupportsFloat],
			value: float,
			log_prob: float,
	):
		"""
		Append one timestep of agent-environment interaction to the buffer.
		"""
		# Note from tutorial:
		# buffer has to have room, so you can store.
		assert self.ptr < self.max_size
		
		self.observation_buf[self.ptr] = observation
		self.action_buf[self.ptr] = action
		self.reward_buf[self.ptr] = reward
		self.value_buf[self.ptr] = value
		self.log_prob_buf[self.ptr] = log_prob
		
		# Move ptr for the next storing.
		self.ptr += 1
	
	def finish_path(self, last_value):
		"""
		Using rewards and value estimates from the whole trajectory to
		compute advantage estimates with GAE-lambda.
		"""
		# Retrieve current episode's rewards and value estimates from buffer.
		# And add the last value.
		# If the episode is truncated, or interrupted by the epoch's ending,
		# value estimate of the last observation will be used to replace last reward.
		path_slice = slice(self.path_start_idx, self.ptr)
		rewards = np.append(self.reward_buf[path_slice], last_value)
		values = np.append(self.value_buf[path_slice], last_value)
		
		# Implement GAE-lambda advantage estimate calculation.
		# Note that all buffers have the same length.
		deltas = - values[:-1] + rewards[:-1] + self.gamma * values[1:]
		self.advantage_buf[path_slice] = utils.discount_cumsum(deltas, self.gamma * self.lam)
		
		# Compute reward-to-go.
		# We can't know the true reward-to-go of the last observation,
		# since the episode is ended, so we don't count it.
		self.return_buf[path_slice] = utils.discount_cumsum(rewards, self.gamma)[:-1]
		
		# Set new path_start_idx for recording next episode.
		self.path_start_idx = self.ptr

	def get(self) -> Dict:
		# Note from tutorial:
		# buffer has to be full before you can get.
		assert self.ptr == self.max_size
		self.path_start_idx, self.ptr = 0, 0
		# Note that 'return' can not be used as a key.
		data = dict(
			observation=torch.tensor(self.observation_buf, dtype=torch.float32),
			action=torch.tensor(self.action_buf, dtype=torch.float32),
			advantage=torch.tensor(self.advantage_buf, dtype=torch.float32),
			ret=torch.tensor(self.return_buf, dtype=torch.float32),
			log_prob=torch.tensor(self.log_prob_buf, dtype=torch.float32),
		)
		return data
