from typing import Dict
import numpy as np
import time

import torch
from torch.optim import Adam

from ActotCritic import ActorCritic
from Buffer import Buffer
from log_data import EpochLogger


def count_variables(module):
	variables_number = sum(
		[np.prod(p.shape) for p in module.parameters()]
	)
	return variables_number


def vpg(
		make_env,
		seed: int,
		steps_per_epoch: int,
		gamma: float,
		lam: float,
		clip_ratio: float,
		actor_lr: float,
		critic_lr: float,
		policy_update_iterations: int,
		value_estimate_update_iterations: int,
		target_policy_kl_divergence: float,
		num_epochs: int,
		logger_kwargs: Dict,
):
	# Set up device.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Set up logger.
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())
	
	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate environment.
	env = make_env()
	
	# Construct agent.
	agent = ActorCritic(env.observation_space, env.action_space).to(device)
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
		observation = data["observation"]
		action = data["action"]
		old_action_log_prob = data["action_log_prob"]
		advantage = data["advantage"]
		
		action_distribution, action_log_prob = agent.actor(
			torch.tensor(observation, dtype=torch.float32).to(device),
			torch.tensor(action, dtype=torch.float32).to(device),
		)
		
		# Compute entropy of new_policy
		entropy = action_distribution.entropy().cpu().item()
		
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
		loss = torch.min(policy_ratio*advantage, clipped_advantage)
		
		actor_information = dict(
			kl_divergence=kl_divergence,
			entropy=entropy,
			clip_fraction=clip_fraction,
		)
		
		return loss, actor_information
	
	def compute_critic_loss(data) -> torch.Tensor:
		observation = data['observation']
		ret = data['ret']
		value_estimate = agent.critic(torch.tensor(observation).to(device))
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
		while idx < policy_update_iterations:
			actor_optimizer.zero_grad()
			actor_loss, actor_information = compute_actor_loss(data)
			# It is the kl_divergence between the latest agent.actor's policy
			# and the policy used to collect trajectories.
			kl_divergence = actor_information['kl_divergence']
			# It means the policy has updated enough and should be early-stopped.
			if kl_divergence > 1.5 * target_policy_kl_divergence:
				logger.log("Early stopping at step %d due to reaching max KL divergence." % idx)
				break
			actor_loss.backward()
			actor_optimizer.step()
			
			latest_actor_loss = actor_loss.item()
			latest_actor_information = actor_information
			
			idx += 1
		
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
			Entropy=old_actor_information['Entropy'],
			Clip_Fraction=latest_actor_information['clip_fraction'],
			Delta_Loss_Policy=(latest_actor_loss - old_actor_loss),
			Delta_Loss_Value=(latest_critic_loss - old_critic_loss),
		)
	
	# Let agent interact with environment.
	def main_loop():
		start_time = time.time()
		
		episode_step, episode_reward = 0, 0
		observation, _ = env.reset()
		for epoch in range(num_epochs):
			for timestep in range(steps_per_epoch):
				action, action_log_prob, value_estimate = agent.step(observation)
				# Flag terminated is controlled by steps_per_episode.
				next_observation, reward, terminated, truncated, _ = env.step(action)
				
				episode_step += 1
				episode_reward += reward
				buffer.store(observation, action, reward, action_log_prob, value_estimate)
				logger.store(Value_Estimate=value_estimate)
				
				# Critical !!!
				observation = next_observation
				
				epoch_end = (timestep == steps_per_epoch - 1)
				if epoch_end or terminated or truncated:
					if epoch_end and not (terminated or truncated):
						print('Warning: trajectory cut off by epoch ', end='')
						print('at %d steps.' % episode_step, flush=True)
					
					if epoch or terminated:
						_, _, last_value = agent.step(observation)
					else:
						last_value = 0
					buffer.finish_trajectory(last_value)
					
					if terminated:
						logger.store(
							Episode_Reward=episode_reward,
							Episode_Length=episode_step,
						)
						
					observation, _ = env.reset()
					episode_step, episode_reward = 0, 0
			
			# Save model.
			update()
			
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
