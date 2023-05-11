from typing import Dict
import numpy as np
import torch.random
from copy import deepcopy
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
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
	num_variable = sum(
		[np.prod(parameter.shape) for parameter in module.parameters()]
	)
	return num_variable


def setup_logger_kwargs(exp_name: str, seed: int, data_dir: str):
	# Make a seed-specific subfolder in the experiment directory.
	hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
	logger_kwargs = dict(
		output_dir=os.path.join(data_dir, subfolder), exp_name=exp_name
	)
	return logger_kwargs


def td3(
		make_env,
		seed: int,
		logger_kwargs: Dict,
		capacity: int = 1_000_000,
		target_noise: float = 0.2,
		noise_clip: float = 0.5,
		gamma: float = 0.99,
		actor_lr: float = 1e-3,
		critic_lr: float = 1e-3,
		batch_size: int = 100,
		delayed_update_coef: int = 2,
		polyak_coef: float = 0.995,
		num_epochs: int = 100,
		steps_per_epoch: int = 4_000,
		act_noise: float = 0.1,
		start_steps: int = 10_000,
		update_after: int = 1_000,
		update_every: int = 50,
		save_freq: int = 2,
		num_test_episodes: int = 10,
):
	# Set up logger.
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())
	
	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate environment and test environment (special for off-policy case).
	env, test_env = make_env(), make_env()
	
	# Instantiate agent.
	agent = ActorCritic(env.observation_space, env.action_space)
	# Since target parameters should be equal to main parameters,
	# we can't instantiate ActorCritic again to construct the target agent.
	target_agent = deepcopy(agent)
	
	# Parameters in target actor and critic should be frozen, since they are only
	# updated by polyak averaging.
	for target_parameter in target_agent.parameters():
		target_parameter.requires_grad = False
	
	num_variables = tuple(
		count_variable(m) for m in [agent.actor, agent.critic1, agent.critic2]
	)
	logger.log(
		"\nNumber of variables:\tactor: %d, critic1: %d, critic2: %d\n"
		% num_variables
	)
	
	# Instantiate experience buffer.
	replay_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity)
	
	# Set up loss function.
	# actor loss function
	def compute_actor_loss(data: Dict) -> Tensor:
		observation = data['observation']
		action = agent.actor(observation)
		action_value = agent.critic1(observation, action)
		loss = -torch.mean(action_value)
		return loss
	
	# critic loss function
	def compute_critic_loss(data: Dict) -> (Tensor, Tensor, Dict):
		observation, action, reward, next_observation, done = \
			data['observation'], data['action'], data['reward'], \
			data['next_observation'], data['done']
		
		# Though parameters in target agent don't have gradients, gradients
		# will be computed if we don't specify torch.no_grad().
		with torch.no_grad():
			# Target policy smoothing.
			target_action = target_agent.actor(next_observation)
			noise = torch.clamp(
				input=target_noise * torch.randn_like(target_action),
				min=-noise_clip,
				max=noise_clip
			)
			next_action = torch.clamp(
				input=target_action + noise,
				min=-env.action_space.high[0],
				max=env.action_space.high[0]
			)
			# Compute minimum td target.
			td_target = reward + gamma * (1 - done) * torch.min(
				target_agent.critic1(next_observation, next_action),
				target_agent.critic2(next_observation, next_action)
			)
		
		# Compute action values and losses for two critic.
		action_value1 = agent.critic1(observation, action)
		loss_critic1 = torch.mean((action_value1 - td_target) ** 2)
		action_value2 = agent.critic2(observation, action)
		loss_critic2 = torch.mean((action_value2 - td_target) ** 2)
		
		loss_info = dict(
			Action_Value1=action_value1.detach().numpy(),
			Action_Value2=action_value2.detach().numpy(),
		)
		
		return loss_critic1, loss_critic2, loss_info
		
	# Construct optimizers.
	actor_optimizer = Adam(params=agent.actor.parameters(), lr=actor_lr)
	critic1_optimizer = Adam(params=agent.critic1.parameters(), lr=critic_lr)
	critic2_optimizer = Adam(params=agent.critic2.parameters(), lr=critic_lr)
	
	# Set up model saving.
	logger.setup_pytorch_saver(agent)
	
	# Set up update function.
	def update(update_cnt):
		data = replay_buffer.sample_batch(batch_size)
		
		# Update two critics.
		critic1_optimizer.zero_grad()
		critic2_optimizer.zero_grad()
		loss_critic1, loss_critic2, loss_info = compute_critic_loss(data)
		loss_critic1.backward()
		loss_critic2.backward()
		critic1_optimizer.step()
		critic2_optimizer.step()
		
		logger.store(
			Loss_Critic1=loss_critic1.item(),
			Loss_Critic2=loss_critic2.item(),
			**loss_info,
		)
		
		# Delayed update for main actor, target actor and target critics.
		if update_cnt % delayed_update_coef == 0:
			# Freeze parameters in main critics temporally. (only for efficiency)
			# Since we only use main critic1, we only freeze critic1.
			for parameter in agent.critic1.parameters():
				parameter.requires_grad = False
			
			# Update main actor.
			actor_optimizer.zero_grad()
			loss_actor = compute_actor_loss(data)
			loss_actor.backward()
			actor_optimizer.step()
			
			logger.store(Loss_Actor=loss_actor.item())
			
			# Unfreeze parameters in main critics.
			for parameter in agent.critic1.parameters():
				parameter.requires_grad = True
			
			# Update target actor and two critics.
			with torch.no_grad():
				for parameter, target_parameter in zip(
						agent.parameters(), target_agent.parameters()
				):
					target_parameter.data.mul_(polyak_coef)
					target_parameter.data.add_((1 - polyak_coef) * parameter.data)
	
	def get_noisy_action(observation: np.ndarray, noise_scale) -> np.ndarray:
		action = agent.act(torch.tensor(observation, dtype=torch.float32))
		noise = np.random.randn(env.action_space.shape[0])
		noisy_action = np.clip(
			a=action + noise_scale * noise,
			a_min=-env.action_space.high[0],
			a_max=env.action_space.high[0],
		)
		return noisy_action
	
	def test_agent():
		for _ in range(num_test_episodes):
			observation, _ = test_env.reset()
			episode_reward, episode_len = 0, 0
			done = False
			while not done:
				action = get_noisy_action(observation, 0)
				next_observation, reward, terminated, truncated, _ = \
					test_env.step(action)
				episode_len += 1
				episode_reward += reward
				done = terminated or truncated
				observation = next_observation
			logger.store(
				Test_Episode_Reward=episode_reward,
				Test_Episode_Len=episode_len,
			)
	
	def main_loop():
		tensorboard_idx = 0
		start_time = time.time()
		episode_reward, episode_len = 0, 0
		observation, _ = env.reset()
		for time_step in range(num_epochs * steps_per_epoch):
			if time_step < start_steps:
				action = env.action_space.sample()
			else:
				action = get_noisy_action(observation, act_noise)
			next_observation, reward, terminated, truncated, _ = env.step(action)
			
			episode_len += 1
			episode_reward += reward
			
			done = terminated
			replay_buffer.store(
				observation, action, reward, next_observation, done
			)
			
			# Critical!!!
			observation = next_observation
			
			# End of episode handling.
			if terminated or truncated:
				logger.store(
					Episode_Reward=episode_reward,
					Episode_Len=episode_len,
				)
				observation, _ = env.reset()
				episode_reward, episode_len = 0, 0
			
			# Update handling.
			if (time_step > update_after) and (time_step % update_every == 0):
				for update_cnt in range(update_every):
					update(update_cnt)
			
			# End of epoch handling.
			if (time_step + 1) % steps_per_epoch == 0:
				epoch = (time_step + 1) // steps_per_epoch
				
				# Save model.
				if (epoch % save_freq == 0) or (epoch == num_epochs - 1):
					logger.save_state({'env': env}, None)
					
				# Test agent's performance. (special for off-policy case)
				test_agent()
				
				tensorboard_writer.add_scalar(
					'Test_Episode_Reward_s' + str(seed),
					logger.get_stats('Test_Episode_Reward')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Action_Value1_s' + str(seed),
					logger.get_stats('Action_Value1')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Action_Value2_s' + str(seed),
					logger.get_stats('Action_Value2')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Loss_Critic1_s' + str(seed),
					logger.get_stats('Loss_Critic1')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Loss_Critic2_s' + str(seed),
					logger.get_stats('Loss_Critic2')[0],
					tensorboard_idx,
				)
				tensorboard_idx += 1
				
				# Log information of this epoch.
				logger.log_tabular('Epoch', epoch)
				logger.log_tabular('Episode_Reward', with_min_and_max=True)
				logger.log_tabular('Episode_Len', average_only=True)
				logger.log_tabular('Test_Episode_Reward', with_min_and_max=True)
				logger.log_tabular('Test_Episode_Len', average_only=True)
				logger.log_tabular('Total_Interactions', time_step)
				logger.log_tabular('Action_Value1', with_min_and_max=True)
				logger.log_tabular('Action_Value2', with_min_and_max=True)
				logger.log_tabular('Loss_Actor', average_only=True)
				logger.log_tabular('Loss_Critic1', average_only=True)
				logger.log_tabular('Loss_Critic2', average_only=True)
				logger.log_tabular('Time', time.time() - start_time)
				logger.dump_tabular()
			
	main_loop()
	

def main():
	env_name = 'HalfCheetah-v3'
	experiment_name = 'TD3_HalfCheetah'
	max_episode_steps = 1_000
	num_runs = 3
	seeds = [10 * i for i in range(num_runs)]
	data_dir = ''.join(
		["./data/", time.strftime("%Y-%m-%d_%H-%M-%S_"), experiment_name]
	)
	for seed in seeds:
		logger_kwargs = setup_logger_kwargs(
			experiment_name, seed, data_dir
		)
		td3(
			lambda: gym.make(env_name, max_episode_steps=max_episode_steps),
			seed,
			logger_kwargs,
		)
	
	
if __name__ == '__main__':
	tensorboard_writer = SummaryWriter()
	main()
	tensorboard_writer.close()
