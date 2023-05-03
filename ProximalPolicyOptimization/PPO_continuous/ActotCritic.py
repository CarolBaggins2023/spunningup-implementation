import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int):
		super().__init__()
		self.fc1 = nn.Linear(in_features=in_dim, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=out_dim)
		self.activation = nn.Tanh()
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.activation(self.fc1(x))
		x = self.fc2(x)
		return x


class NormalActor(nn.Module):
	def __init__(self, observation_dim: int, action_dim: int):
		super().__init__()
		log_std = - 0.5 * np.ones(action_dim, dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.tensor(log_std, dtype=torch.float32))
		self.mu_net = MLP(in_dim=observation_dim, out_dim=action_dim)
	
	def distribution(self, observation: torch.Tensor) -> Normal:
		mu = self.mu_net(observation)
		std = torch.exp(self.log_std)
		action_distribution = Normal(mu, std)
		return action_distribution
	
	@staticmethod
	def log_prob_from_distribution(
			action_distribution: Normal,
			action: torch.Tensor,
	) -> torch.Tensor:
		# Critical:
		# sum(axis=-1) is necessary.
		# Difference between discrete and continuous action space.
		action_log_prob = action_distribution.log_prob(action).sum(axis=-1)
		return action_log_prob
	
	def forward(
			self,
			observation: torch.Tensor,
			action: torch.Tensor,
	) -> (Normal, torch.Tensor):
		action_distribution = self.distribution(observation)
		action_log_prob = self.log_prob_from_distribution(
			action_distribution, action
		)
		return action_distribution, action_log_prob


class Critic(nn.Module):
	def __init__(self, observation_dim):
		super().__init__()
		self.value_net = MLP(in_dim=observation_dim, out_dim=1)
	
	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		value_estimate = self.value_net(observation)
		return value_estimate


class ActorCritic(nn.Module):
	def __init__(self, observation_space, action_space):
		super().__init__()
		observation_dim = observation_space.shape[0]
		# Critical:
		# action_space of datatype Box has no attribute n.
		# Difference between discrete and continuous action space.
		action_dim = action_space.shape[0]
		self.actor = NormalActor(observation_dim, action_dim)
		self.critic = Critic(observation_dim)
	
	def step(
			self,
			observation: torch.Tensor,
	) -> (np.ndarray, float, float):
		with torch.no_grad():
			action_distribution = self.actor.distribution(observation)
			action = action_distribution.sample()
			action_log_prob = self.actor.log_prob_from_distribution(
				action_distribution, action
			)
			value_estimate = self.critic(observation)
		return (
			action.numpy(),
			action_log_prob.numpy(),
			value_estimate.numpy(),
		)
	
	def act(self, observation: torch.Tensor) -> np.ndarray:
		action = self.step(observation)[0]
		return action
