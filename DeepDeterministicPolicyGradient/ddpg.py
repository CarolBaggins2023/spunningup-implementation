import numpy as np
from typing import Dict, Tuple
from copy import deepcopy
import time
import os
# Essential for implementing MuJoCo environment
os.add_dll_directory("C://Users//lenovo//.mujoco//mjpro150//bin")
os.add_dll_directory("C://Users//lenovo//.mujoco//mujoco-py//mujoco_py")

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from ActorCritic import ActorCritic
from ReplayBuffer import ReplayBuffer
from log_data import EpochLogger


def count_variables(module: nn.Module) -> int:
	count = sum([np.prod(parameter.shape) for parameter in module.parameters()])
	return count


def setup_logger_kwargs(exp_name: str, seed: int, data_dir: str):
	# Make a seed-specific subfolder in the experiment directory.
	hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
	
	logger_kwargs = dict(output_dir=os.path.join(data_dir, subfolder), exp_name=exp_name)
	return logger_kwargs


def ddpg(
		make_env,
		seed: int,
		logger_kwargs: Dict,
		replay_buffer_size: int = int(1e6),
		gamma: float = 0.99,
		actor_lr: float = 1e-3,
		critic_lr: float = 1e-3,
		batch_size: int = 100,
		polyak_coef: float = 0.995,
		steps_per_epoch: int = 4_000,
		num_epochs: int = 100,
		start_steps: int = 10_000,
		action_noise_coef: float = 0.1,
		update_after_interaction: int = 1_000,
		update_every_interaction: int = 50,
		save_state_freq: int = 1,
		num_test_episodes: int = 10,
):
	# Set up logger.
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate environment.
	env, test_env = make_env(), make_env()
	
	# Instantiate agent.
	agent = ActorCritic(env.observation_space, env.action_space)
	agent_target = deepcopy(agent)
	# Target policy network and target value function network
	# are only updated by polyak average. So, their parameters don't
	# need to store gradient.
	for parameter in agent_target.parameters():
		parameter.requires_grad = False
	variable_count = tuple(count_variables(module) for module in [agent.actor, agent.critic])
	logger.log("\nNumber of parameters:\tactor: %d, critic: %d\n" % variable_count)
	
	# Instantiate experience buffer.
	replay_buffer = ReplayBuffer(
		env.observation_space, env.action_space, capacity=replay_buffer_size
	)
	
	# Set up loss function.
	def compute_actor_loss(data: Dict) -> Tensor:
		observation = data['observation']
		action_value = agent.critic(observation, agent.actor(observation))
		loss = - torch.mean(action_value)
		return loss
	
	def compute_critic_loss(data: Dict) -> Tuple[Tensor, Dict]:
		observation, action, reward, next_observation, done = \
			data['observation'], data['action'], data['reward'], data['next_observation'], data['done']
		action_value = agent.critic(observation, action)
		with torch.no_grad():
			next_action_value = agent_target.critic(next_observation, agent.actor(next_observation))
			td_target = reward + gamma * (1 - done) * next_action_value
		loss = torch.mean((action_value - td_target) ** 2)
		loss_info = dict(action_value=action_value.detach().numpy())
		return loss, loss_info
	
	# Set up optimizers.
	actor_optimizer = Adam(agent.actor.parameters(), actor_lr)
	critic_optimizer = Adam(agent.critic.parameters(), critic_lr)
	
	# Set up model saving.
	logger.setup_pytorch_saver(agent)
	
	# Set up update function.
	def update():
		data = replay_buffer.sample_batch(batch_size)
		
		# Run one gradient descent to update critic.
		critic_optimizer.zero_grad()
		critic_loss, critic_loss_info = compute_critic_loss(data)
		critic_loss.backward()
		critic_optimizer.step()
		
		# Run one gradient ascent to update actor.
		# Note:
		# Freeze parameters of critic before updating and unfreeze them after updating.
		# Because otherwise gradients of both actor and critic will be computed, while we only need critic's gradient.
		# Actually, gradients coming from actor's update will not affect the correctness of algorithm,
		# since we will execute critic_optimizer.zero_grad() before updating critic's parameters, so there is no
		# gradient overlying.
		# So these two steps are to improve code efficiency.
		for parameter in agent.critic.parameters():
			parameter.requires_grad = False
		
		actor_optimizer.zero_grad()
		actor_loss = compute_actor_loss(data)
		actor_loss.backward()
		actor_optimizer.step()
		
		for parameter in agent.critic.parameters():
			parameter.requires_grad = True
			
		# Record update information.
		logger.store(
			Critic_Loss=critic_loss.item(),
			Actor_Loss=actor_loss.item(),
			Action_Value=critic_loss_info['action_value'],
		)
		
		# Update target actor and critic by polyak averaging.
		with torch.no_grad():
			for parameter, target_parameter in zip(agent.parameters(), agent_target.parameters()):
				target_parameter.data.mul_(polyak_coef)
				target_parameter.data.add_((1-polyak_coef) * parameter.data)
	
	def get_noisy_action(observation: np.ndarray, action_noise) -> Tuple[np.ndarray, np.ndarray]:
		action = agent.act(torch.tensor(observation, dtype=torch.float32))
		noisy_action = action + action_noise * np.random.randn(env.action_space.shape[0])
		# We must constrain the value of action in a valid range.
		action_limit = env.action_space.high[0]
		clip_ratio = np.mean(np.less(noisy_action, -action_limit) | np.greater(noisy_action, action_limit))
		noisy_action = np.clip(noisy_action, -action_limit, action_limit)
		return noisy_action, clip_ratio
	
	def test_agent_performance():
		for _ in range(num_test_episodes):
			observation, _ = test_env.reset()
			episode_reward, episode_len = 0, 0
			episode_forward_reward, episode_ctrl_cost = 0, 0
			episode_clip_ratio = list()
			terminated, truncated = False, False
			while not (terminated or truncated):
				action, clip_ratio = get_noisy_action(observation, action_noise=0)
				next_observation, reward, terminated, truncated, info = test_env.step(action)
				episode_len += 1
				episode_reward += reward
				episode_forward_reward += info['reward_run']
				episode_ctrl_cost += info['reward_ctrl']
				episode_clip_ratio.append(clip_ratio)
				# Critical!!!
				observation = next_observation
				
			logger.store(
				Test_Episode_Reward=episode_reward,
				Test_Episode_Len=episode_len,
				Test_Episode_Forward_Reward=episode_forward_reward,
				Test_Episode_Ctrl_Cost=episode_ctrl_cost,
				Test_Episode_Clip_Ratio=np.mean(episode_clip_ratio),
			)
	
	def main_loop():
		# Run the main loop.
		tensorboard_idx = 0
		total_steps = steps_per_epoch * num_epochs
		start_time = time.time()
		observation, _ = env.reset()
		episode_reward, episode_len = 0, 0
		# t is the number of interactions between agent and environment.
		for t in range(total_steps):
			# Before t reaching start_steps, we randomly sample actions.
			if t < start_steps:
				action = env.action_space.sample()
			else:
				action, _ = get_noisy_action(observation, action_noise_coef)
			
			next_observation, reward, terminated, truncated, _ = env.step(action)
			episode_len += 1
			episode_reward += reward
			
			# In off-policy case, 'done' only means the agent reaching the terminal state,
			# excluding the situation that the episode is interrupted by time limit.
			done = terminated
			replay_buffer.store(observation, action, reward, next_observation, done)
			
			# Critical!!!
			observation = next_observation
			
			# End of episode handling.
			if terminated or truncated:
				logger.store(Episode_Reward=episode_reward, Episode_Len=episode_len)
				observation, _ = env.reset()
				episode_reward, episode_len = 0, 0
			
			# Update.
			if t >= update_after_interaction and t % update_every_interaction == 0:
				# The number of updates is equal to the number of interactions.
				for _ in range(update_every_interaction):
					update()
			
			# End of epoch handling.
			if (t + 1) % steps_per_epoch == 0:
				epoch = (t + 1) // steps_per_epoch
				
				# Save model.
				if (epoch % save_state_freq) == 0 or epoch == num_epochs:
					logger.save_state({'env': env}, None)
				
				# Test the performance of the deterministic version of the agent.
				test_agent_performance()
				
				tensorboard_writer.add_scalar(
					'Test_Episode_Reward_s' + str(seed),
					logger.get_stats('Test_Episode_Reward')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Test_Episode_Forward_Reward_s' + str(seed),
					logger.get_stats('Test_Episode_Forward_Reward')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Test_Episode_Ctrl_Cost_s' + str(seed),
					logger.get_stats('Test_Episode_Ctrl_Cost')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Test_Episode_Clip_Ratio_s' + str(seed),
					logger.get_stats('Test_Episode_Clip_Ratio')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Action_Value_s' + str(seed),
					logger.get_stats('Action_Value')[0],
					tensorboard_idx,
				)
				tensorboard_writer.add_scalar(
					'Critic_Loss_s' + str(seed),
					logger.get_stats('Critic_Loss')[0],
					tensorboard_idx,
				)
				tensorboard_idx += 1
				
				# Log information about this epoch.
				logger.log_tabular('Epoch', epoch)
				logger.log_tabular('Episode_Reward', with_min_and_max=True)
				logger.log_tabular('Episode_Len', average_only=True)
				logger.log_tabular('Test_Episode_Reward', with_min_and_max=True)
				logger.log_tabular('Test_Episode_Forward_Reward', average_only=True)
				logger.log_tabular('Test_Episode_Ctrl_Cost', average_only=True)
				logger.log_tabular('Test_Episode_Len', average_only=True)
				logger.log_tabular('Test_Episode_Clip_Ratio', average_only=True)
				logger.log_tabular('Total_Interactions', t + 1)
				logger.log_tabular('Action_Value', with_min_and_max=True)
				logger.log_tabular('Actor_Loss', average_only=True)
				logger.log_tabular('Critic_Loss', average_only=True)
				logger.log_tabular('Time', time.time() - start_time)
				logger.dump_tabular()
			
	main_loop()


def main():
	env_name = 'HalfCheetah-v3'
	experiment_name = 'DDPG_HalfCheetah'
	max_episode_steps = 1_000
	num_runs = 3
	seeds = [10 * i + 10 for i in range(num_runs)]
	data_dir = ''.join(['./data/', time.strftime("%Y-%m-%d_%H-%M-%S_"), experiment_name])
	for seed in seeds:
		logger_kwargs = setup_logger_kwargs(experiment_name, seed, data_dir)
		ddpg(
			make_env=lambda: gym.make(env_name, max_episode_steps=max_episode_steps),
			seed=seed,
			logger_kwargs=logger_kwargs,
		)


if __name__ == '__main__':
	tensorboard_writer = SummaryWriter()
	main()
	tensorboard_writer.close()
