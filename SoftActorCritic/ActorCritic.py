import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch.distributions import Normal


class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, output_activation):
		super().__init__()
		self.fc1 = nn.Linear(in_features=in_dim, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=256)
		self.fc3 = nn.Linear(in_features=256, out_features=out_dim)
		self.activation = nn.ReLU()
		self.output_activation = output_activation
	
	def forward(self, x: Tensor) -> Tensor:
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		x = self.fc3(x)
		x = self.output_activation(x)
		return x


class Actor(nn.Module):
	def __init__(self, observation_dim: int, action_dim: int, action_limit: float):
		super().__init__()
		self.intermediate_net = MLP(
			in_dim=observation_dim, out_dim=256, output_activation=nn.ReLU()
		)
		self.mu_net = MLP(
			in_dim=256, out_dim=action_dim, output_activation=nn.Identity()
		)
		self.log_std_net = MLP(
			in_dim=256, out_dim=action_dim, output_activation=nn.Identity()
		)
		self.action_limit = action_limit
	
	def forward(
			self,
			observation: Tensor,
			deterministic: bool = False,
			with_logprob: bool = True,
	) -> (Tensor, Tensor):
		intermediate_output = self.intermediate_net(observation)
		mu = self.mu_net(intermediate_output)
		log_std = self.log_std_net(intermediate_output)
		log_std = torch.clamp(log_std, -20, 2)
		std = torch.exp(log_std)
		
		# Distribution and action before squash.
		action_distribution = Normal(mu, std)
		if deterministic:
			# At test time, we take deterministic action.
			# In Gaussian distribution, mean point gets the largest value.
			action = mu
		else:
			# distribution.sample() samples point from distribution directly.
			# distribution.rsample() first samples point from Gaussian(0, 1),
			# then output (distribution.mean + sampled_value * distribution.std).
			action = action_distribution.rsample()
		
		if with_logprob:
			action_logprob = action_distribution.log_prob(action).sum(axis=-1)
			action_logprob -= (
					2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action))
			).sum(axis=1)
		else:
			action_logprob = None
		
		action = torch.tanh(action)
		# Critical:
		# Constrain action value in a legal bound.
		action = self.action_limit * action
		
		return action, action_logprob


class Critic(nn.Module):
	def __init__(self, observation_dim: int, action_dim: int):
		super().__init__()
		self.value_function = MLP(
			in_dim=observation_dim + action_dim,
			out_dim=1,
			output_activation=torch.nn.Identity()
		)
	
	def forward(self, observation: Tensor, action: Tensor) -> Tensor:
		action_value = self.value_function(torch.cat([observation, action], dim=-1))
		# Critical:
		# Make sure to flatten action values.
		return torch.squeeze(action_value, dim=-1)


class ActorCritic(nn.Module):
	def __init__(self, observation_space, action_space):
		super().__init__()
		observation_dim = observation_space.shape[0]
		action_dim = action_space.shape[0]
		action_limit = action_space.high[0]
		
		self.actor = Actor(observation_dim, action_dim, action_limit)
		self.critic1 = Critic(observation_dim, action_dim)
		self.critic2 = Critic(observation_dim, action_dim)
	
	def act(
			self,
			observation: Tensor,
			deterministic: bool = False,
	) -> np.ndarray:
		with torch.no_grad():
			action, _ = self.actor(observation, deterministic, with_logprob=False)
		return action.numpy()
