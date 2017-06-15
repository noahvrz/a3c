import torch
import numpy as np
from scipy.misc import imresize

def resize(observation):

	# Resize image to (42, 42)
	observation = imresize(observation, (80,80))
	observation = imresize(observation, (42,42))

	# Average the RGB channels to flatten the image
	observation = observation.mean(2)

	# Convert to float, change to range (0, 1)
	observation = observation.astype(np.float32)
	observation *= (1.0 / 255.0)

	# Reshape so that channel is first
	observation = np.reshape(observation, [1, 42, 42])

	# Normalize the observation
	observation = (observation - observation.mean())/(observation.std() + 1e-8)

	state = torch.from_numpy(observation)
	return torch.autograd.Variable(state.unsqueeze(0))