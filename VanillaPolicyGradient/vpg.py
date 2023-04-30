from typing import Dict
import numpy as np
import time
import os
# Essential for implementing MuJoCo environment
os.add_dll_directory("C://Users//lenovo//.mujoco//mjpro150//bin")
os.add_dll_directory("C://Users//lenovo//.mujoco//mujoco-py//mujoco_py")

import torch
from torch.optim import Adam
# When you want to visualize metrics in tensorboard
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from log_data import EpochLogger
from VPGBuffer import VPGBuffer
from ActorCritic import MLPActorCritic
import utils


def vpg(
		make_env,
		perform_continuous: bool,
		seed: int,
		logger_kwargs: Dict,
		use_gpu: bool = False,
		steps_per_epoch: int = 4_000,
		epochs: int = 50,
		gamma: float = 0.99,
		lam: float = 0.97,
		max_episode_steps: int = 1_000,
		policy_lr: int = 3e-4,
		value_function_lr: float = 1e-3,
		save_freq: int = 10,
):
	# Set device.
	device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
	
	# Set up logger.
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())
	
	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate environment.
	env = make_env()
	observation_dim = env.observation_space.shape
	action_dim = env.action_space.shape
	
	# Construct actor-critic module.
	agent = MLPActorCritic(env.observation_space, env.action_space, device)
	# Count agent's variables.
	variable_counts = tuple(
		utils.count_variables(module)
		for module in [agent.policy, agent.value_function]
	)
	logger.log(
		"\nNumber of parameters: \t policy: %d, \t value_function: %d\n" % variable_counts
	)
	
	# Instantiate experience buffer.
	buffer = VPGBuffer(observation_dim, action_dim, steps_per_epoch, gamma, lam)
	
	# Set up callable loss function.
	# Set up loss function for policy.
	def compute_loss_policy(data):
		# observations and actions will be put into network,
		# so they need to be put on device.
		observations = data["observation"].to(device)
		actions = data["action"].to(device)
		advantages = data["advantage"]
		log_probs_old = data["log_prob"]
		
		# Policy loss
		# Note that 'mean()' in following code refers to average over time steps.
		action_distributions, log_probs = agent.policy(observations, actions)
		loss_policy = -(log_probs * advantages).mean()
		
		# Useful extra information
		# KL divergence can measure the distance between two probability distribution.
		# So, it can be used to measure the difference between two policies,
		# indicating how much the policy has changed by the update.
		# But kl_divergence here is not the true KL divergence, but an approximation.
		kl_divergence = (log_probs_old - log_probs).mean().item()
		# Entropy can measure the stochasticity of the policy.
		# Policy's entropy is larger if the probability of choosing each action
		# is more evenly distributed.
		policy_entropy = action_distributions.entropy().mean().item()
		policy_info = dict(kl_divergence=kl_divergence, policy_entropy=policy_entropy)
		
		return loss_policy, policy_info
	
	# Set up loss function for value_function.
	def compute_loss_value_function(data):
		# observations will be put into network, so it need to be put on device.
		observations = data["observation"]
		returns = data["ret"]
		values = agent.value_function(observations)
		loss_value_function = ((values - returns) ** 2).mean()
		return loss_value_function
	
	# Instantiate optimizers.
	policy_optimizer = Adam(agent.policy.parameters(), policy_lr)
	value_function_optimizer = Adam(agent.value_function.parameters(), value_function_lr)
	
	# Set up model saving.
	logger.setup_pytorch_saver(agent)
	
	# Set up update function.
	# If tensorboard is usd, define it as update(tensorboard_idx):
	def update():
		data = buffer.get()
		
		# Get loss before update.
		loss_policy_old, policy_info_old = compute_loss_policy(data)
		loss_policy_old = loss_policy_old.item()
		loss_value_function_old = compute_loss_value_function(data)
		loss_value_function_old = loss_value_function_old.item()
		
		"""
		Check if there is any difference between loss_policy_old and loss_policy.
		
		Answer:
		Since we only execute single step here, loss_policy_old and loss_policy are the same.
		They will be different when executing multiple steps.
		"""
		
		# Train policy.
		policy_optimizer.zero_grad()
		loss_policy, policy_info = compute_loss_policy(data)
		loss_policy.to(device)
		loss_policy.backward()
		policy_optimizer.step()
		
		# Train value_function.
		value_function_optimizer.zero_grad()
		loss_value_function = compute_loss_value_function(data)
		loss_value_function.to(device)
		loss_value_function.backward()
		value_function_optimizer.step()
		
		# Log policy's changes from update.
		# When executing policy update for multiple steps,
		# kl_divergence is related to the latest policy and
		# the policy used to collect the data in buffer.
		# But note that the policy_entropy relates to the old one, not the new one.
		kl_divergence, policy_entropy = policy_info["kl_divergence"], policy_info_old["policy_entropy"]
		# Note that Loss_policy, Loss_value_function, and Policy_entropy
		# all record the information of the old policy,
		# but not the policy after update.
		logger.store(
			Loss_Policy=loss_policy_old,
			Loss_Value_Function=loss_value_function_old,
			KL_Divergence=kl_divergence,
			Policy_Entropy=policy_entropy,
			Delta_Loss_Policy=(loss_policy.item() - loss_policy_old),
			Delta_Loss_Value_Function=(loss_value_function.item() - loss_value_function_old),
		)
		
		# Note that loss_policy here means nothing.
		# There is no connection between it and performance.
		# With performance, we should only care about average return.
		# writer.add_scalar("Loss_Policy", loss_policy_old, tensorboard_idx)
		# writer.add_scalar("Loss_Value_Function", loss_value_function_old, tensorboard_idx)
		# writer.add_scalar("KL_Divergence", kl_divergence, tensorboard_idx)
		# writer.add_scalar("Policy_Entropy", policy_entropy, tensorboard_idx)
	
	# Run the main loop.
	start_time = time.time()
	
	# Set up tensor board writer.
	# writer = SummaryWriter(log_dir=logger_kwargs['output_dir'])
	# tensorboard_idx_reward = 0
	# tensorboard_idx_update = 0
	
	# Critical!!!
	# Do not forget the placeholder, since env.reset() have two returns.
	observation, _ = env.reset()
	episode_reward = 0
	episode_len = 0
	for epoch in range(epochs):
		for epoch_steps in range(steps_per_epoch):
			action, value, log_prob = agent.step(
				torch.tensor(observation, dtype=torch.float32).to(device)
			)
			
			next_observation, reward, terminated, truncated, _ = env.step(action)
			episode_reward += reward
			episode_len += 1
			
			# Save and log sub-episode.
			buffer.store(observation, action, reward, value, log_prob)
			logger.store(Value_Estimate=value)
			
			# Critical!!!
			observation = next_observation
			
			# Check whether this episode is terminated or truncated,
			# or it should be interrupted due to the end of the epoch.
			epoch_end = (epoch_steps == steps_per_epoch - 1)
			# Once one of the flags is true, the episode ends.
			if terminated or truncated or epoch_end:
				# The episode is interrupted due to the end of the epoch.
				if epoch_end and not (terminated and truncated):
					print("Warning: trajectory is cut off by ", end='')
					print("epoch's end at %d steps." % episode_len, flush=True)
				
				# If this episode has achieved termination, the agent
				# will not get any reward, indicating the value of its
				# last observation is 0.
				if terminated:
					value = 0
				# If this episode has not achieved termination, the agent
				# should have get more rewards, so we need to approximate
				# the value of its last observation.
				else:
					_, value, _ = agent.step(
						torch.tensor(observation, dtype=torch.float32).to(device)
					)
				buffer.finish_path(last_value=value)
				
				# Log episode's reward and length.
				# Note that we only care about HalfCheetah-v3 environment.
				# Because its episode will only be truncated, terminated is always False.
				# So we consider truncated as terminated.
				if perform_continuous:
					if terminated or truncated:
						logger.store(
							Episode_Reward=episode_reward,
							Episode_Length=episode_len+1,
						)
				else:
					if terminated:
						logger.store(
							Episode_Reward=episode_reward,
							Episode_Length=episode_len+1,
						)
					
					# Note that frequent calls to tensor board will slow down the speed.
					# writer.add_scalar("Episode_Reward", episode_reward, global_step=tensorboard_idx_reward)
					# tensorboard_idx_reward += 1
				
				# Reset the environment.
				observation, _ = env.reset()
				episode_reward = 0
				episode_len = 0
			
			# Note one difference with what we have done before.
			# When an episode ends, a new episode will being in the same epoch.
			# There is no 'break'.
		
		# Save model.
		if (epoch % save_freq == 0) or (epoch == epochs - 1):
			logger.save_state({"Env": env}, None)
		
		# Critical !!!
		# Perform VPG update.
		update()
		# update(tensorboard_idx_update)
		# tensorboard_idx_update += 1
		
		# Log information about this epoch.
		logger.log_tabular("Epoch", epoch)
		logger.log_tabular("Episode_Reward", with_min_and_max=True)
		logger.log_tabular('Episode_Length', average_only=True)
		logger.log_tabular('Value_Estimate', with_min_and_max=True)
		logger.log_tabular('Total_Interactions', (epoch + 1) * steps_per_epoch)
		logger.log_tabular('Loss_Policy', average_only=True)
		logger.log_tabular('Loss_Value_Function', average_only=True)
		logger.log_tabular('Delta_Loss_Policy', average_only=True)
		logger.log_tabular('Delta_Loss_Value_Function', average_only=True)
		logger.log_tabular('Policy_Entropy', average_only=True)
		logger.log_tabular('KL_Divergence', average_only=True)
		logger.log_tabular('Time', time.time() - start_time)
		logger.dump_tabular()
	
	# Close tensor board writer.
	# writer.close()


def main():
	# Set discrete and continuous tasks' parameters.
	discrete_env = dict(
		exp_name='vpg_CartPole',
		env_name='CartPole-v1',
		epochs=250,
	)
	# Setting of following epochs aims to align with spinningup's benchmark.
	continuous_env = dict(
		exp_name='vpg_HalfCheetah',
		env_name='HalfCheetah-v3',
		# env_name='LunarLander-v2',
		epochs=750,
	)
	# To perform discrete or continuous task.
	perform_continuous = False
	
	# Since we only use small model here, using gpu even slow down training process.
	# It may be attributed to moving data from gpu to cpu, and sometimes the opposite.
	# The training speed of gpu and cpu will be close when there are 5 full-connected
	# linear layers in policy and value function networks.
	use_gpu = False
	
	max_episode_steps = 1_000
	num_runs = 5
	seeds = [10 * i for i in range(num_runs)]
	data_dir = ''.join([
		'../data/',
		time.strftime("%Y-%m-%d_%H-%M-%S_"),
		continuous_env['exp_name'] if perform_continuous else discrete_env['exp_name'],
	])
	for seed in seeds:
		logger_kwargs = utils.setup_logger_kwargs(
			continuous_env['exp_name'] if perform_continuous else discrete_env['exp_name'],
			seed,
			data_dir,
		)
		
		vpg(
			make_env=lambda: gym.make(
				continuous_env['env_name'] if perform_continuous else discrete_env['env_name'],
				max_episode_steps=max_episode_steps,
			),
			perform_continuous=perform_continuous,
			seed=seed,
			logger_kwargs=logger_kwargs,
			epochs=continuous_env['epochs'] if perform_continuous else discrete_env['epochs'],
		)


if __name__ == '__main__':
	main()
