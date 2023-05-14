from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
import numpy as np
from copy import deepcopy
import time
import os
# Essential for implementing MuJoCo environment
os.add_dll_directory("C://Users//lenovo//.mujoco//mjpro150//bin")
os.add_dll_directory("C://Users//lenovo//.mujoco//mujoco-py//mujoco_py")
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from ActorCritic import ActorCritic
from ReplayBuffer import ReplayBuffer
from log_data import EpochLogger


def count_variable(module: nn.Module) -> int:
	num_variables = sum([np.prod(p.shape) for p in module.parameters()])
	return num_variables


def setup_logger_kwargs(exp_name: str, seed: int, data_dir: str):
	# Make a seed-specific subfolder in the experiment directory.
	hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
	logger_kwargs = dict(
		output_dir=os.path.join(data_dir, subfolder), exp_name=exp_name
	)
	return logger_kwargs


def sac(
		make_env,
		seed: int,
		logger_kwargs: Dict,
		replay_buffer_capacity: int = int(1e6),
		entropy_weight: float = 0.2,
		discount_factor: float = 0.99,
		lr: float = 1e-3,
		batch_size: int = 100,
		polyak_coef: float = 0.995,
		num_epochs: int = 100,
		steps_per_epoch: int = 4_000,
		start_steps: int = 10_000,
		update_after: int = 1_000,
		update_every: int = 50,
		save_freq: int = 1,
		num_test_episodes: int = 10,
):
	# Set up a logger.
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())
	
	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate train and test environments.
	env = make_env()
	test_env = make_env()
	
	# Instantiate main and target agents.
	agent = ActorCritic(env.observation_space, env.action_space)
	target_agent = deepcopy(agent)
	for parameter in target_agent.parameters():
		parameter.requires_grad = False
	
	num_variables = tuple(
		count_variable(module) for module in [
			agent.actor, agent.critic1, agent.critic2
		]
	)
	logger.log(
		"\nNumber of variables:\tactor: %d, critic1: %d, critic2: %d\n" %
		num_variables
	)
	
	# Instantiate a replay buffer.
	replay_buffer = ReplayBuffer(
		env.observation_space, env.action_space, replay_buffer_capacity
	)
	
	# Set up loss functions.
	def compute_actor_loss(data: Dict) -> (Tensor, Dict):
		observation = data['observation']
		
		action, action_logprob = agent.actor(observation)
		
		action_value1 = agent.critic1(observation, action)
		action_value2 = agent.critic2(observation, action)
		
		loss = -(
				torch.min(action_value1, action_value2) -
				entropy_weight * action_logprob
		).mean()
		
		information = dict(Action_Logprob=action_logprob.detach().numpy())
		
		return loss, information
	
	def compute_critic_loss(data) -> (Tensor, Tensor, Dict):
		observation, action, reward, next_observation, done =\
			data['observation'], data['action'], data['reward'],\
			data['next_observation'], data['done']
		
		with torch.no_grad():
			# Note to distinguish:
			# next_action comes from 'current' 'main' actor.
			# next_action_value is computed by 'target' critic
			next_action, next_action_logprob = agent.actor(next_observation)
			next_action_value1 = target_agent.critic1(next_observation, next_action)
			next_action_value2 = target_agent.critic2(next_observation, next_action)
			td_target = reward + discount_factor * (1 - done) * (
				torch.min(next_action_value1, next_action_value2) -
				entropy_weight * next_action_logprob
			)
		
		action_value1 = agent.critic1(observation, action)
		action_value2 = agent.critic2(observation, action)
		
		loss_critic1 = torch.mean((action_value1 - td_target) ** 2)
		loss_critic2 = torch.mean((action_value2 - td_target) ** 2)
		
		information = dict(
			Action_Value1=action_value1.detach().numpy(),
			Action_Value2=action_value2.detach().numpy(),
		)
		
		return loss_critic1, loss_critic2, information
		
	# Set up optimizers.
	actor_optimizer = Adam(params=agent.actor.parameters(), lr=lr)
	critic1_optimizer = Adam(params=agent.critic1.parameters(), lr=lr)
	critic2_optimizer = Adam(params=agent.critic2.parameters(), lr=lr)
	
	# Set up model saving.
	logger.setup_pytorch_saver(agent)
	
	# Set up an update function.
	def update():
		data = replay_buffer.sample_batch(batch_size)
		
		# Update main critics.
		critic1_optimizer.zero_grad()
		critic2_optimizer.zero_grad()
		loss_critic1, loss_critic2, critic_information = compute_critic_loss(data)
		loss_critic1.backward()
		loss_critic2.backward()
		critic1_optimizer.step()
		critic2_optimizer.step()
		
		logger.store(
			Loss_Critic1=loss_critic1.item(),
			Loss_Critic2=loss_critic2.item(),
			**critic_information,
		)
		
		# Update main actor.
		# Before computing actor's loss.
		for parameter in agent.critic1.parameters():
			parameter.requires_grad = False
		for parameter in agent.critic2.parameters():
			parameter.requires_grad = False
		
		actor_optimizer.zero_grad()
		loss_actor, actor_information = compute_actor_loss(data)
		loss_actor.backward()
		actor_optimizer.step()
		
		logger.store(
			Loss_Actor=loss_actor.item(),
			**actor_information,
		)
		
		# After computing actor's loss.
		for parameter in agent.critic1.parameters():
			parameter.requires_grad = True
		for parameter in agent.critic2.parameters():
			parameter.requires_grad = True
		
		# Update target agent's actor and critics.
		with torch.no_grad():
			for parameter, target_parameter in zip(
				agent.parameters(), target_agent.parameters()
			):
				target_parameter.mul_(polyak_coef)
				target_parameter.add_((1 - polyak_coef) * parameter.data)
	
	def test_agent():
		for _ in range(num_test_episodes):
			episode_len, episode_reward = 0, 0
			observation, _ = test_env.reset()
			done = False
			while not done:
				action = agent.act(
					torch.tensor(observation, dtype=torch.float32),
					deterministic=True,
				)
				next_observation, reward, terminated, truncated, _ =\
					test_env.step(action)
				episode_len += 1
				episode_reward += reward
				done = terminated or truncated
				# Critical!!!
				observation = next_observation
			logger.store(
				Test_Episode_Len=episode_len,
				Test_Episode_Reward=episode_reward,
			)
	
	def main_loop():
		start_time = time.time()
		episode_len, episode_reward = 0, 0
		observation, _ = env.reset()
		for step in range(num_epochs * steps_per_epoch):
			if step < start_steps:
				action = env.action_space.sample()
			else:
				action = agent.act(torch.tensor(observation, dtype=torch.float32))
			
			next_observation, reward, terminated, truncated, _ = env.step(action)
			
			episode_len += 1
			episode_reward += reward
			replay_buffer.store(
				observation, action, reward, next_observation, done=terminated
			)
			
			# Critical!!!
			observation = next_observation
			
			# Episode ending handling.
			if terminated or truncated:
				logger.store(
					Episode_Len=episode_len,
					Episode_Reward=episode_reward,
				)
				episode_len, episode_reward = 0, 0
				observation, _ = env.reset()
			
			# Update handling.
			if (step > update_after) and (step % update_every == 0):
				for _ in range(update_every):
					update()
		
			# Epoch ending handling.
			if (step + 1) % steps_per_epoch == 0:
				epoch = (step + 1) // steps_per_epoch
				
				# Save model.
				if (epoch % save_freq == 0) or (epoch == num_epochs):
					logger.save_state({'env': env}, None)
				
				# Test agent's performance.
				test_agent()
				
				# Log information about this epoch.
				logger.log_tabular('Epoch', epoch)
				logger.log_tabular('Episode_Reward', with_min_and_max=True)
				logger.log_tabular('Episode_Len', average_only=True)
				logger.log_tabular('Test_Episode_Reward', with_min_and_max=True)
				logger.log_tabular('Test_Episode_Len', average_only=True)
				logger.log_tabular('Total_Interactions', step)
				logger.log_tabular('Action_Value1', with_min_and_max=True)
				logger.log_tabular('Action_Value2', with_min_and_max=True)
				logger.log_tabular('Action_Logprob', with_min_and_max=True)
				logger.log_tabular('Loss_Actor', average_only=True)
				logger.log_tabular('Loss_Critic1', average_only=True)
				logger.log_tabular('Loss_Critic2', average_only=True)
				logger.log_tabular('Time', time.time() - start_time)
				logger.dump_tabular()
	
	main_loop()


def main():
	env_name = 'HalfCheetah-v3'
	exp_name = 'SAC_HalfCheetah'
	max_episode_steps = 1_000
	num_runs = 3
	seeds = [10 * i for i in range(num_runs)]
	data_dir = ''.join(
		['./data/', time.strftime("%Y-%m-%d_%H-%M-%S_"), exp_name]
	)
	for seed in seeds:
		logger_kwargs = setup_logger_kwargs(
			exp_name, seed, data_dir
		)
		sac(
			lambda: gym.make(env_name, max_episode_steps=max_episode_steps),
			seed,
			logger_kwargs,
		)


if __name__ == '__main__':
	main()
