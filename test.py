import gym

import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.nn.functional import softmax

from collections import deque
import time

from config import *
from agent import Agent

from resize import resize

def test(rank, shared_agent):


	# Create a gym environment, get the action space and observation shape
	env = gym.make(GAME_NAME)
	action_space = env.action_space

	# Seed the RNGs
	env.seed(SEED + rank)
	torch.manual_seed(SEED + rank)

	# Create a new model for this learner, set it to evaluation mode
	agent = Agent(RESIZE_SHAPE, action_space)
	agent.eval()

	# Start a new episode, get the state
	state = resize(env.reset())

	# Keep track of the maximum reward we've achieved
	max_reward = -1e10
	total_reward = 0

	done = True
	episode_length = 0

	num_episodes = 0
	actions = deque(maxlen = 100)

	start_time = time.time()
	
	# Loop forever [TODO: Make this process stop when training stops]
	while True:

		episode_length += 1

		if done: # last episode finished
			# At the beginning of each episode, load from shared model
			agent.load_state_dict(shared_agent.state_dict())

			# (volatile=true is used in testing only, speeds up computation)
			c_x = Variable(torch.zeros(1, 256), volatile=True)
			h_x = Variable(torch.zeros(1, 256), volatile=True)
		else: # last episode unfinished
			c_x = Variable(c_x.data, volatile=True)
			h_x = Variable(h_x.data, volatile=True)

		# Get estimated value, policy, LSTM cell for new state
		value, policy, (h_x, c_x) = agent((state, (h_x, c_x)))

		# Choose an action
		# Here, just greedily choose the action
		action_probs = softmax(policy)
		action = action_probs.max(1)[1].data.numpy()

		# Act on the environment with the chosen action, find observation and reward
		observation, reward, done, _ = env.step(action[0, 0])
		state = resize(observation)
		total_reward += reward

		# If we get stuck for long enough, end the episode
		actions.append(action[0][0])
		if actions.count(actions[0]) == actions.maxlen:
			done = True

		if episode_length >= MAX_EPISODE_LENGTH:
			done = True

		# If the episode is over, reset it
		if done:

			num_episodes += 1

			print("[{}] Episode {} completed in {} steps, with {} total reward".format(
				time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
				num_episodes, episode_length, total_reward))

			# If this particular model performed best, save it
			if total_reward > max_reward:
				f = "{}/snapshot_best{}".format(SAVE_FOLDER, EXTENSION)
				print("Saving to file {}".format(f))
				torch.save(shared_agent.state_dict(), f)
				max_reward = total_reward
			#[TODO: Somehow have this save when it consistently gets high scores?]

			total_reward = 0
			episode_length = 0
			actions.clear()
			state = resize(env.reset())
			time.sleep(30)