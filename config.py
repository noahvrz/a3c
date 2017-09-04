# Many of the hyperparameters used in this implementation are
# from https://github.com/ikostrikov/pytorch-a3c, due to a lack
# of time to perform optimization, and proven good results

# OpenAI Gym Parameters
GAME_NAME = 'SpaceInvaders-v0'
SAVE_FOLDER = "spaceinvaders_snapshots"

RESIZE_SHAPE = (1, 42, 42) 
SEED = 1 # Random seed to replicate runs

DISPLAY = False

# Multiprocessing Parameters
NUM_WORKERS = 8 # This many will train, one more will test

# Learning Parameters
NUM_A3C_STEPS = 20
MAX_EPISODE_LENGTH = 10000
MAX_ITERS = 1000000
TEST_INTERVAL = 60 # number of seconds to sleep between tests

LEARNING_RATE = .0001 # LR for Adam optimizer
REWARD_CLIP = 1 # Clipping rewards encourages generality
GAMMA = 0.99 # Discount on future rewards
LAMBDA = 1.0 # Used in generalized advantage estimate
ENTROPY_EFFECT = .01 # Encourages exploration
GRAD_CLIP = 40 # Ensures reasonable sized gradients

# File I/O Parameters
SAVE_INTERVAL = 5000
FILE_PREFIX = "iteration_"
EXTENSION = ".torch"
GIF_FRAME_SKIP = 3 # Saving every screen results in a slow GIF

# Model Parameters
CONV_NUM_FILTERS = 32
CONV_FILTER_SIZE = 3
CONV_STRIDE = 2
RNN_INPUT_SIZE = 3*3*CONV_NUM_FILTERS # Inputs are flattened before RNN
RNN_SIZE = 256
