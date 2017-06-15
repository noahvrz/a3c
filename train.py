import gym

import numpy as np

import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.nn.functional import softmax, log_softmax
from torch.nn.utils import clip_grad_norm

from config import *
from agent import Agent
from resize import resize

def check_grads(shared_agent, agent):
	for (param, shared_param) in zip(agent.parameters(), shared_agent.parameters()):
		if shared_param.grad is not None:
			return
		shared_param._grad = param.grad

def train(rank, shared_agent, optimizer):

	# Create a gym environment, get the action space and observation shape
	env = gym.make(GAME_NAME)
	action_space = env.action_space

	# Seed the RNGs
	torch.manual_seed(SEED + rank)
	env.seed(SEED + rank)

	# Create a new model for this learner, set it to train mode
	agent = Agent(RESIZE_SHAPE, action_space)
	agent.train()

	# Start a new episode, get the state
	state = resize(env.reset())

	done = True
	episode_length = 0

	# Continue until we reach the max number of iterations
	# [TODO: Stop the process when convergence is reached]
	num_iters = 0
	while True:

		if num_iters > MAX_ITERS:
			return

		# Save the shared state to a file every so often
		if num_iters % SAVE_INTERVAL == 0 and rank == 0:
			f = "{}/{}{}{}".format(SAVE_FOLDER, FILE_PREFIX, num_iters, EXTENSION)
			print("Saving iteration {} to file {}".format(num_iters, f))
			torch.save(shared_agent.state_dict(), f)
		
		num_iters += 1
		episode_length += 1

		# At the beginning of each iteration, load from shared model
		agent.load_state_dict(shared_agent.state_dict())

		if done: # last episode finished
			c_x = Variable(torch.zeros(1, 256))
			h_x = Variable(torch.zeros(1, 256))
		else: # last episode unfinished
			c_x = Variable(c_x.data)
			h_x = Variable(h_x.data)

		values = []
		log_probabilities = []
		rewards = []
		entropies = []

		for step in range(NUM_A3C_STEPS):

			# Get estimated value, policy, LSTM cell for new state
			value, policy, (h_x, c_x) = agent((state, (h_x, c_x)))
			values.append(value)

			# Use log probabilites for numerical stability
			action_probs = softmax(policy)
			action_log_probs = log_softmax(policy)

			# Calculate and store entropy
			entropy = -(action_log_probs * action_probs).sum(1)
			entropies.append(entropy)

			# Pick an action from the softmax probabilities
			# Here, we want to choose from a distribution so we can explore
			action = action_probs.multinomial().data

			# Save the log probabilities of chosen action to later calculate entropy
			log_probability = action_log_probs.gather(1, Variable(action))
			log_probabilities.append(log_probability)

			# Act on the environment with the chosen action, find observation and reward
			observation, reward, done, _ = env.step(action.numpy())
			state = resize(observation)

			# Clip reward to -1<=r<=1
			reward = max(-1, min(reward, 1)) 
			rewards.append(reward)

			# Limit the episode length
			if episode_length >= MAX_EPISODE_LENGTH:
				done = True

			# If the episode is over, reset it
			if done:
				episode_length = 0
				state = resize(env.reset())
				break

		# Now, to train the network
		R = torch.zeros(1, 1)
		if not done:
			value, _, _ = agent((state, (h_x, c_x)))
			R = value.data

		values.append(Variable(R))
		R = Variable(R)

		policy_loss = 0
		value_loss = 0
		general_advantage_estimate = torch.zeros(1, 1)

		# Calculate the loss for the policy and value networks
		for i in reversed(range(len(rewards))):
			
			R = GAMMA * R + rewards[i]
			advantage = R - values[i]
			value_loss = value_loss + 0.5 * advantage.pow(2)

			delta = rewards[i] + GAMMA*values[i+1].data - values[i].data
			general_advantage_estimate = LAMBDA*GAMMA*general_advantage_estimate + delta

			policy_loss = policy_loss - (log_probabilities[i] * Variable(general_advantage_estimate)) - (ENTROPY_EFFECT * entropies[i])

		# Clear the gradients from last step
		optimizer.zero_grad()

		# Perform backpropogation on the sum of the loss functions, clip gradients
		(policy_loss + 0.5 * value_loss).backward()
		clip_grad_norm(agent.parameters(), GRAD_CLIP)

		# Make sure that the shared model has grads everywhere
		check_grads(shared_agent, agent)

		# Perform an optimization step using the shared Adam optimizer
		optimizer.step()