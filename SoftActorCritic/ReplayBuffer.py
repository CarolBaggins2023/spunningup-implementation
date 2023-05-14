import numpy as np
from typing import SupportsFloat

import torch


def combine_shape(capacity, shape):
	if np.isscalar(shape):
		return capacity, shape
	else:
		return capacity, *shape


class ReplayBuffer:
	def __init__(self, observation_space, action_space, capacity):
		self.observation_buf = np.zeros(
			shape=combine_shape(capacity, observation_space.shape[0]),
			dtype=np.float32,
		)
		self.action_buf = np.zeros(
			shape=combine_shape(capacity, action_space.shape[0]),
			dtype=np.float32,
		)
		self.reward_buf = np.zeros(shape=capacity, dtype=np.float32)
		self.next_observation_buf = np.zeros_like(
			self.observation_buf,
			dtype=np.float32,
		)
		self.done_buf = np.zeros_like(self.reward_buf, dtype=np.float32)
		self.ptr, self.size, self.capacity = 0, 0, capacity
	
	def store(
			self,
			observation: np.ndarray,
			action: np.ndarray,
			reward: SupportsFloat,
			next_observation: np.ndarray,
			done: bool,
	):
		self.observation_buf[self.ptr] = observation
		self.action_buf[self.ptr] = action
		self.reward_buf[self.ptr] = reward
		self.next_observation_buf[self.ptr] = next_observation
		self.done_buf[self.ptr] = done
		
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)
	
	def sample_batch(self, batch_size: int):
		idxs = np.random.randint(low=0, high=self.size, size=batch_size)
		batch = dict(
			observation=self.observation_buf[idxs],
			action=self.action_buf[idxs],
			reward=self.reward_buf[idxs],
			next_observation=self.next_observation_buf[idxs],
			done=self.done_buf[idxs],
		)
		batch_tensor = {
			key: torch.tensor(value, dtype=torch.float32) for key, value in batch.items()
		}
		return batch_tensor
