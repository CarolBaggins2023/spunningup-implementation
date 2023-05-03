from typing import Dict
import numpy as np
import time
import os

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from ActotCritic import ActorCritic
from Buffer import Buffer
from log_data import EpochLogger


def count_variables(module):
	variables_number = sum(
		[np.prod(p.shape) for p in module.parameters()]
	)
	return variables_number


def setup_logger_kwargs(exp_name: str, seed: int, data_dir: str):
	# Make a seed-specific subfolder in the experiment directory.
	hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
	
	logger_kwargs = dict(output_dir=os.path.join(data_dir, subfolder), exp_name=exp_name)
	return logger_kwargs


def ppo(
		make_env,
		seed: int,
		steps_per_epoch: int = 4_000,
		gamma: float = 0.99,
		lam: float = 0.97,
		clip_ratio: float = 0.2,
		actor_lr: float = 3e-4,
		critic_lr: float = 1e-3,
		policy_update_iterations: int = 80,
		value_estimate_update_iterations: int = 80,
		target_policy_kl_divergence: float = 0.01,
		num_epochs: int = 50,
		model_save_frequency: int = 10,
		logger_kwargs: Dict = None,
):
	# Set up logger.
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())
	
	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate environment.
	env = make_env()
	
	# Construct agent.
	agent = ActorCritic(env.observation_space, env.action_space)
	variables_number = tuple(
		count_variables(module) for module in [agent.actor, agent.critic]
	)
	logger.log("Number of variables:\tactor: %d,\tcritic: %d" % variables_number)
	
	# Instantiate experience buffer.
	buffer = Buffer(
		steps_per_epoch,
		env.observation_space,
		env.action_space,
		gamma,
		lam,
	)
	
	# Set up callable loss function
	def compute_actor_loss(data) -> (torch.Tensor, Dict):
		# Diagnostics to the algorithm:
		# (1) KL_divergence of old_policy and new_policy (just an approximation here),
		# (2) entropy of new_policy,
		# (3) clip_fraction (since we use PPO-clip).
		
		# Note that the prefix old refers to the policy used to collect trajectories.
		observation = torch.tensor(data["observation"], dtype=torch.float32)
		action = torch.tensor(data["action"], dtype=torch.float32)
		old_action_log_prob = torch.tensor(data["action_log_prob"], dtype=torch.float32)
		advantage = torch.tensor(data["advantage"], dtype=torch.float32)
		
		action_distribution, action_log_prob = agent.actor(observation, action)
		
		# Compute entropy of new_policy
		entropy = action_distribution.entropy().mean().item()
		
		# Compute clip_fraction.
		policy_ratio = torch.exp(action_log_prob - old_action_log_prob)
		is_clipped = policy_ratio.gt(1 + clip_ratio) | policy_ratio.lt(1 - clip_ratio)
		clip_fraction = np.mean(is_clipped.numpy())
		
		# Compute KL divergence of old_policy and new_policy
		kl_divergence = (old_action_log_prob - action_log_prob).mean().item()
		
		# Compute loss
		clipped_policy_ratio = torch.clamp(
			policy_ratio, 1 - clip_ratio, 1 + clip_ratio
		)
		clipped_advantage = clipped_policy_ratio * advantage
		# Critical:
		# When updating policy, loss should be negative,
		# since we are going to execute gradient ascent.
		loss = - (torch.min(policy_ratio * advantage, clipped_advantage)).mean()
		
		actor_information = dict(
			kl_divergence=kl_divergence,
			entropy=entropy,
			clip_fraction=clip_fraction,
		)
		
		return loss, actor_information
	
	def compute_critic_loss(data) -> torch.Tensor:
		observation = torch.tensor(data['observation'], dtype=torch.float32)
		ret = torch.tensor(data['ret'], dtype=torch.float32)
		value_estimate = agent.critic(observation)
		loss = ((value_estimate - ret) ** 2).mean()
		return loss
	
	# Set up optimizer for policy and value function
	actor_optimizer = Adam(agent.actor.parameters(), actor_lr)
	critic_optimizer = Adam(agent.critic.parameters(), critic_lr)

	# Set up model saving
	logger.setup_pytorch_saver(agent)
	
	# Set up update function
	def update():
		data = buffer.get_data()
		
		old_actor_loss, old_actor_information = compute_actor_loss(data)
		old_actor_loss = old_actor_loss.item()
		old_critic_loss = compute_critic_loss(data)
		old_critic_loss = old_critic_loss.item()
		
		# Perform multiple steps of gradient ascent to update policy
		# using the same trajectories.
		idx = 0
		latest_actor_loss = None
		latest_actor_information = None
		for idx in range(policy_update_iterations):
			actor_optimizer.zero_grad()
			actor_loss, actor_information = compute_actor_loss(data)
			# It is the kl_divergence between the latest agent.actor's policy
			# and the policy used to collect trajectories.
			kl_divergence = actor_information['kl_divergence']
			# It means the policy has updated enough and should be early-stopped.
			if kl_divergence > 1.5 * target_policy_kl_divergence:
				logger.log("Early stopping at step %d due to reaching max KL divergence." % (idx + 1))
				break
			actor_loss.backward()
			actor_optimizer.step()
			
			latest_actor_loss = actor_loss.item()
			latest_actor_information = actor_information
		
		logger.store(Early_Stop_Iter=idx)

		# Value estimate function learning.
		latest_critic_loss = None
		for _ in range(value_estimate_update_iterations):
			critic_optimizer.zero_grad()
			critic_loss = compute_critic_loss(data)
			critic_loss.backward()
			critic_optimizer.step()
			
			latest_critic_loss = critic_loss.item()
		
		# Log changes from updates.
		logger.store(
			Loss_Policy=old_actor_loss,
			Loss_Value=old_critic_loss,
			KL_Divergence=latest_actor_information['kl_divergence'],
			Entropy=old_actor_information['entropy'],
			Clip_Fraction=latest_actor_information['clip_fraction'],
			Delta_Loss_Policy=(latest_actor_loss - old_actor_loss),
			Delta_Loss_Value=(latest_critic_loss - old_critic_loss),
		)
	
	# Let agent interact with environment.
	def main_loop():
		start_time = time.time()
		
		tensorboard_idx = 0
		episode_len, episode_reward = 0, 0
		observation, _ = env.reset()
		for epoch in range(num_epochs):
			for timestep in range(steps_per_epoch):
				action, action_log_prob, value_estimate = agent.step(
					torch.tensor(observation, dtype=torch.float32)
				)
				# Flag truncated is controlled by steps_per_episode.
				next_observation, reward, terminated, truncated, _ = env.step(action)
				
				episode_len += 1
				episode_reward += reward
				buffer.store(observation, action, reward, action_log_prob, value_estimate)
				logger.store(Value_Estimate=value_estimate)
				
				# Critical !!!
				observation = next_observation
				
				epoch_end = (timestep == steps_per_epoch - 1)
				if epoch_end or terminated or truncated:
					if epoch_end and not (terminated or truncated):
						print('Warning: trajectory cut off by epoch ', end='')
						print('at %d steps.' % episode_len, flush=True)
					
					# When the agent has not reached terminal state, and is interrupted
					# by environment.
					if epoch_end or truncated:
						_, _, last_value = agent.step(
							torch.tensor(observation, dtype=torch.float32)
						)
					# When the agent has reached terminal state.
					else:
						last_value = 0
					buffer.finish_trajectory(last_value)
					
					if terminated:
						logger.store(
							Episode_Reward=episode_reward,
							Episode_Length=episode_len,
						)
						
					observation, _ = env.reset()
					episode_len, episode_reward = 0, 0
			
			# Save environment and agent.
			if (epoch % model_save_frequency == 0) or (epoch == num_epochs - 1):
				logger.save_state({'environment': env, 'agent': agent}, None)
			
			# Perform PPO update!
			update()
			
			tensorboard_writer.add_scalar(
				'Entropy_s'+str(seed), logger.get_stats('Entropy')[0], tensorboard_idx
			)
			tensorboard_writer.add_scalar(
				'Episode_Reward_s'+str(seed), logger.get_stats('Episode_Reward')[0], tensorboard_idx
			)
			tensorboard_writer.add_scalar(
				'Clip_Fraction_s'+str(seed), logger.get_stats('Clip_Fraction')[0], tensorboard_idx
			)
			tensorboard_idx += 1
			
			# Log information of this epoch.
			logger.log_tabular('Epoch', epoch)
			logger.log_tabular('Episode_Reward', with_min_and_max=True)
			logger.log_tabular('Episode_Length', average_only=True)
			logger.log_tabular('Value_Estimate', with_min_and_max=True)
			logger.log_tabular('Total_Interaction', (epoch + 1) * steps_per_epoch)
			logger.log_tabular('Loss_Policy', average_only=True)
			logger.log_tabular('Loss_Value', average_only=True)
			logger.log_tabular('Delta_Loss_Policy', average_only=True)
			logger.log_tabular('Delta_Loss_Value', average_only=True)
			logger.log_tabular('Entropy', average_only=True)
			logger.log_tabular('KL_Divergence', average_only=True)
			logger.log_tabular('Clip_Fraction', average_only=True)
			logger.log_tabular('Early_Stop_Iter', average_only=True)
			logger.log_tabular('Time', time.time() - start_time)
			logger.dump_tabular()
	
	# Run the main loop.
	main_loop()


def main():
	env_name = 'CartPole-v1'
	experiment_name = 'PPO_CartPole'
	max_episode_steps = 1_000
	num_runs = 3
	seeds = [10 * i for i in range(num_runs)]
	data_dir = ''.join(['./data/', time.strftime("%Y-%m-%d_%H-%M-%S_"), experiment_name])
	for seed in seeds:
		logger_kwargs = setup_logger_kwargs(experiment_name, seed, data_dir)
		ppo(
			make_env=lambda: gym.make(env_name, max_episode_steps=max_episode_steps),
			seed=seed,
			logger_kwargs=logger_kwargs,
		)


if __name__ == '__main__':
	tensorboard_writer = SummaryWriter()
	main()
	tensorboard_writer.close()
