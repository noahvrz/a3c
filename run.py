import gym

import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.functional import softmax

from config import *
from agent import Agent
from resize import resize

import imageio # To create gifs of the run

# Create new gym environment
env = gym.make(GAME_NAME)
action_space = env.action_space

# Create a new model, set it to train mode
agent = Agent(RESIZE_SHAPE, action_space)
agent.eval()

f = "{}/snapshot_best.torch".format(SAVE_FOLDER)
state_dict = torch.load(f)
agent.load_state_dict(state_dict)

# Start a new environment, get the state
observation = env.reset()
state = resize(observation)

done = False

# Initialize LSTM states to 0
c_x = Variable(torch.zeros(1, 256))
h_x = Variable(torch.zeros(1, 256))

num_iters = 0

# Will add frames to this to output a gif
screens = []
screens.append(observation)

total_reward = 0
while not done:

	if DISPLAY:
		env.render()

	if num_iters != 0: 
		c_x = Variable(c_x.data)
		h_x = Variable(h_x.data)

	# Get estimated value, policy, LSTM cell for new state
	value, policy, (h_x, c_x) = agent((state, (h_x, c_x)))

	# Greedily pick an action from the policy
	action_probs = softmax(policy)
	action = action_probs.max(1, keepdim=True)[1].data.numpy()

	# Act on the environment with the chosen action, find observation and reward
	observation, reward, done, _ = env.step(action[0, 0])
	total_reward += reward	

	# Build up frames of the gif
	if num_iters % GIF_FRAME_SKIP == 0:
		screens.append(observation)

	state = resize(observation)

	if num_iters >= MAX_EPISODE_LENGTH:
		done = True

	num_iters += 1

print("Accumulated {} total reward".format(total_reward))

f = "images/{}.gif".format(GAME_NAME)
print("Saving .gif to {}".format(f))
imageio.mimsave(f, screens)
