import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int):
		super().__init__()
		self.fc1 = nn.Linear(in_features=in_dim, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=256)
		self.fc3 = nn.Linear(in_features=256, out_features=out_dim)
		self.activation = nn.ReLU()
		self.tanh = nn.Tanh()
	
	def forward(self, x: Tensor) -> Tensor:
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		x = self.fc3(x)
		x = self.tanh(x)
		return x


class Actor(nn.Module):
	def __init__(self, observation_dim: int, action_dim: int, action_limit: float):
		super().__init__()
		self.policy = MLP(in_dim=observation_dim, out_dim=action_dim)
		self.action_limit = action_limit
	
	def forward(self, observation: Tensor) -> Tensor:
		action = self.action_limit * self.policy(observation)
		return action


class Critic(nn.Module):
	def __init__(self, observation_action_dim: int):
		super().__init__()
		self.value_function = MLP(observation_action_dim, 1)
	
	def forward(self, observation: Tensor, action: Tensor) -> Tensor:
		action_value = self.value_function(torch.cat([observation, action], dim=-1))
		return torch.squeeze(action_value)


class ActorCritic(nn.Module):
	def __init__(self, observation_space, action_space):
		super().__init__()
		observation_dim = observation_space.shape[0]
		action_dim = action_space.shape[0]
		action_limit = action_space.high[0]
		
		self.actor = Actor(observation_dim, action_dim, action_limit)
		self.critic = Critic(observation_dim+action_dim)
	
	def act(self, observation: Tensor) -> np.ndarray:
		with torch.no_grad():
			action = self.actor(observation).numpy()
			return action
