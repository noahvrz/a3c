import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import elu

from config import *

# Weight intializations are copied from https://github.com/ikostrikov/pytorch-a3c
# These perform well for this type of network 
def normalized_columns_initializer(weights, std=1.0):
	out = torch.randn(weights.size())
	out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
	return out
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		weight_shape = list(m.weight.data.size())
		fan_in = np.prod(weight_shape[1:4])
		fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		m.weight.data.uniform_(-w_bound, w_bound)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		weight_shape = list(m.weight.data.size())
		fan_in = weight_shape[1]
		fan_out = weight_shape[0]
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		m.weight.data.uniform_(-w_bound, w_bound)
		m.bias.data.fill_(0)

# Creates a new neural network module with our model architecture
class Agent(nn.Module):

	def __init__(self, input_shape, action_space):

		super(Agent, self).__init__()

		# The architecture first consists of 4 CNN's to learn image input
		num_channels = input_shape[0]
		self.conv1 = nn.Conv2d(num_channels, CONV_NUM_FILTERS, CONV_FILTER_SIZE, stride=CONV_STRIDE, padding=1)
		self.conv2 = nn.Conv2d(CONV_NUM_FILTERS, CONV_NUM_FILTERS, CONV_FILTER_SIZE, stride=CONV_STRIDE, padding=1)
		self.conv3 = nn.Conv2d(CONV_NUM_FILTERS, CONV_NUM_FILTERS, CONV_FILTER_SIZE, stride=CONV_STRIDE, padding=1)
		self.conv4 = nn.Conv2d(CONV_NUM_FILTERS, CONV_NUM_FILTERS, CONV_FILTER_SIZE, stride=CONV_STRIDE, padding=1)

		# Then, an LSTM learns time dependence
		self.lstm = nn.LSTMCell(RNN_INPUT_SIZE, RNN_SIZE)

		# The policy output will be softmaxed to give a probability of choosing each action
		self.policy = nn.Linear(RNN_SIZE, action_space.n)
		
		# The value will have a single linear output
		self.value = nn.Linear(RNN_SIZE, 1)

		# Perform the weight initializations. Again, from https://github.com/ikostrikov/pytorch-a3c
		self.apply(weights_init)

		self.policy.weight.data = normalized_columns_initializer(self.policy.weight.data, 0.01)
		self.policy.bias.data.fill_(0)
		self.value.weight.data = normalized_columns_initializer(self.value.weight.data, 1.0)
		self.value.bias.data.fill_(0)

		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)
		
		# Set the agent to training mode
		self.train()

	# Define the forward pass through the network
	def forward(self, inputs):
		
		state, (h_x, c_x) = inputs

		# Use elu nonlinearity after each convolutional layer
		x = elu(self.conv1(state))
		x = elu(self.conv2(x))
		x = elu(self.conv3(x))
		x = elu(self.conv4(x))

		# reshape 1,32,3,3 -> 1,32*3*3
		x = x.view(-1, RNN_INPUT_SIZE)

		# run output through lstm, get cell, hidden state
		h_x, c_x = self.lstm(x, (h_x, c_x))
		x = h_x

		return self.value(x), self.policy(x), (h_x, c_x)
