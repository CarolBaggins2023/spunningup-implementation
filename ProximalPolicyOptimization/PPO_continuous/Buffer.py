import numpy as np
from typing import SupportsFloat


def combine_shape(capacity, space_shape):
	if np.isscalar(space_shape):
		return capacity, space_shape
	else:
		return capacity, *space_shape


def compute_cumsum(data: np.ndarray, discount: float) -> np.ndarray:
	cumsum = 0
	cumsum_list = list()
	for elem in data[::-1]:
		cumsum = discount * cumsum + elem
		cumsum_list.append(cumsum)
	cumsum_list.reverse()
	return np.array(cumsum_list)


class Buffer:
	def __init__(
			self,
			capacity: int,
			observation_space,
			action_space,
			gamma: float,
			lam: float,
	):
		self.observation_buf = np.zeros(
			combine_shape(capacity, observation_space.shape),
			dtype=np.float32,
		)
		self.action_buf = np.zeros(
			combine_shape(capacity, action_space.shape),
			dtype=np.float32,
		)
		self.reward_buf = np.zeros(capacity, dtype=np.float32)
		
		# To update policy, we need
		self.action_log_prob_buf = np.zeros(capacity, dtype=np.float32)
		self.value_buf = np.zeros(capacity, dtype=np.float32)
		self.advantage_buf = np.zeros(capacity, dtype=np.float32)
		self.gamma, self.lam = gamma, lam
		
		# To update value estimate function, we need
		# value's buffer has been defined.
		self.return_buf = np.zeros(capacity, dtype=np.float32)
		
		self.ptr, self.trajectory_begin_ptr, self.max_size = 0, 0, capacity
	
	def store(
			self,
			observation: np.ndarray,
			action: np.ndarray,
			reward: SupportsFloat,
			action_log_prob: float,
			value: float,
	):
		assert self.ptr < self.max_size
		self.observation_buf[self.ptr] = observation
		self.action_buf[self.ptr] = action
		self.reward_buf[self.ptr] = reward
		self.action_log_prob_buf[self.ptr] = action_log_prob
		self.value_buf[self.ptr] = value
		self.ptr += 1
	
	def finish_trajectory(self, last_value: float):
		trajectory_slice = slice(self.trajectory_begin_ptr, self.ptr)
		
		# Compute advantages.
		# V_(s_(t+1)) is in need to calculate delta_t.
		# So, extend value_estimates by append a last_value.
		values_estimate = np.append(self.value_buf[trajectory_slice], last_value)
		# To fairly estimate the reward-to-go of the last observation of
		# a truncated trajectory of length T, sum of (R_T, ...) should be
		# appended in the end of reward_buf.
		# Sum of (R_T, ...) is approximated by last_value here.
		rewards = np.append(self.reward_buf[trajectory_slice], last_value)
		delta = - values_estimate[:-1] + rewards[:-1] + self.gamma * values_estimate[1:]
		self.advantage_buf[trajectory_slice] = compute_cumsum(
			delta, self.gamma * self.lam
		)
		
		# Compute returns (rewards-to-go).
		# Critical: execute [:-1],
		# otherwise it will be a list with a length of T+1, not T.
		self.return_buf[trajectory_slice] = compute_cumsum(
			rewards, self.gamma
		)[:-1]
		
		self.trajectory_begin_ptr = self.ptr
	
	def get_data(self):
		assert self.ptr == self.max_size
		# Execute advantage normalization.
		advantage_mean = np.mean(self.advantage_buf)
		advantage_std = np.std(self.advantage_buf)
		self.advantage_buf = (self.advantage_buf - advantage_mean) / advantage_std
		data = dict(
			observation=self.observation_buf,
			action=self.action_buf,
			action_log_prob=self.action_log_prob_buf,
			ret=self.return_buf,
			advantage=self.advantage_buf,
		)
		self.ptr, self.trajectory_begin_ptr = 0, 0
		return data
