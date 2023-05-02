import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


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


class CategoricalActor(nn.Module):
	def __init__(self, observation_dim: int, action_dim: int):
		super().__init__()
		self.policy_net = MLP(in_dim=observation_dim, out_dim=action_dim)
	
	def distribution(self, observation: torch.Tensor) -> Categorical:
		action_logits = self.policy_net(observation)
		action_distribution = Categorical(logits=action_logits)
		return action_distribution
	
	@staticmethod
	def log_prob_from_distribution(
			action_distribution: Categorical,
			action: torch.Tensor,
	) -> torch.Tensor:
		action_log_prob = action_distribution.log_prob(action)
		return action_log_prob
	
	def forward(
			self,
			observation: torch.Tensor,
			action: torch.Tensor,
	):
		action_distribution = self.distribution(observation)
		action_log_prob = self.log_prob_from_distribution(
			action_distribution, action
		)
		return action_distribution, action_log_prob.cpu()
		

class Critic(nn.Module):
	def __init__(self, observation_dim):
		super().__init__()
		self.value_net = MLP(in_dim=observation_dim, out_dim=1)
	
	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		value_estimate = self.value_net(observation)
		return value_estimate.cpu()


class ActorCritic(nn.Module):
	def __init__(self, observation_space, action_space):
		super().__init__()
		observation_dim = observation_space.shape[0]
		action_dim = action_space.n
		self.actor = CategoricalActor(observation_dim, action_dim)
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
			action.cpu().numpy(),
			action_log_prob.cpu().numpy(),
			value_estimate.cpu().numpy(),
		)
	
	def act(self, observation: torch.Tensor) -> np.ndarray:
		action = self.step(observation)[0]
		return action
