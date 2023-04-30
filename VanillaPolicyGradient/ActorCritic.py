import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from gymnasium.spaces import Discrete, Box


class MLP(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.fc1 = nn.Linear(in_features=in_dim, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=out_dim)
		self.activation = nn.Tanh()

	def forward(self, x):
		x = self.activation(self.fc1(x))
		x = self.fc2(x)
		return x
	

class MLPCategoricalActor(nn.Module):
	def __init__(self, observation_dim, action_dim, device):
		super().__init__()
		self.device = device
		self.logits_net = MLP(in_dim=observation_dim, out_dim=action_dim).to(device)
	
	def forward(
			self,
			observation: torch.Tensor,
			action: torch.Tensor,
	) -> (Categorical, torch.Tensor):
		distribution = self.distribution(observation)
		if action is not None:
			log_prob = self.log_prob_from_distribution(distribution, action)
		else:
			log_prob = None
		return distribution, log_prob.cpu()
	
	def distribution(self, observation: torch.Tensor) -> Categorical:
		logits = self.logits_net(observation)
		distribution = Categorical(logits=logits)
		return distribution
	
	def log_prob_from_distribution(
			self,
			distribution: Categorical,
			action: torch.Tensor,
	) -> torch.Tensor:
		log_prob = distribution.log_prob(action)
		return log_prob


class MLPGaussianActor(nn.Module):
	def __init__(self, observation_dim, action_dim, device):
		super().__init__()
		self.device = device
		log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.tensor(log_std)).to(device)
		self.mu_net = MLP(in_dim=observation_dim, out_dim=action_dim).to(device)
	
	def forward(
			self,
			observation: torch.Tensor,
			action: torch.Tensor,
	) -> (Normal, torch.Tensor):
		distribution = self.distribution(observation)
		if action is not None:
			log_prob = self.log_prob_from_distribution(distribution, action)
		else:
			log_prob = None
		return distribution, log_prob.cpu()
	
	def distribution(self, observation: torch.Tensor) -> Normal:
		std = torch.exp(self.log_std).to(self.device)
		mu = self.mu_net(observation)
		distribution = Normal(loc=mu, scale=std)
		return distribution
	
	def log_prob_from_distribution(
			self,
			distribution: Normal,
			action: torch.Tensor,
	) -> torch.Tensor:
		# Note from the tutorial:
		# last axis sum needed for Torch Normal distribution.
		log_prob = distribution.log_prob(action).sum(axis=-1)
		return log_prob


class MLPCritic(nn.Module):
	def __init__(self, observation_dim: int, device):
		super().__init__()
		self.device = device
		self.value_net = MLP(in_dim=observation_dim, out_dim=1).to(device)
	
	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		# Note from the tutorial:
		# critical to ensure value has right shape.
		value = torch.squeeze(self.value_net(observation), -1)
		return value.cpu()


class MLPActorCritic(nn.Module):
	def __init__(self, observation_space, action_space, device):
		super().__init__()
		observation_dim = observation_space.shape[0]
		
		self.device = device
		
		# Construct the policy based on the type of action_space
		if isinstance(action_space, Discrete):
			action_dim = action_space.n
			self.policy = MLPCategoricalActor(observation_dim, action_dim, self.device)
		elif isinstance(action_space, Box):
			action_dim = action_space.shape[0]
			self.policy = MLPGaussianActor(observation_dim, action_dim, self.device)
		
		self.value_function = MLPCritic(observation_dim, self.device)
	
	def step(
			self,
			observation: torch.Tensor,
	) -> (np.ndarray, np.ndarray, np.ndarray):
		# We will not execute Tensor.backward() after step(),
		# so there is no need to compute gradients.
		with torch.no_grad():
			action_distribution = self.policy.distribution(observation)
			action = action_distribution.sample()
			value = self.value_function(observation)
			# Note that we can not directly perform log_prob = action_distribution.log_prob(action).
			# Because if there is a Gaussian distribution, we also need to sum up the log_prob,
			# just as MLPGaussianActor.log_prob_from_distribution().
			# So we should call policy.log_prob_from_distribution() here.
			log_prob = self.policy.log_prob_from_distribution(action_distribution, action)
			
			# value is out put by value_function.forward(), so it is already on cpu.
			# Note that we can't convert cuda:0 device type tensor to numpy directly.
			# We should use Tensor.cpu() to copy the tensor to host memory first.
		return action.cpu().numpy(), value.numpy(), log_prob.cpu().numpy()
	
	def act(self, observation: torch.Tensor) -> np.ndarray:
		action = self.step(observation)[0]
		return action
