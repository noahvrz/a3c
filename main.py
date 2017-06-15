import gym

import os

import torch
from torch import multiprocessing as mp

from config import *
from agent import Agent
from train import train
from test import test
from adam import SharedAdam

if __name__ == "__main__":

	os.environ['OMP_NUM_THREADS'] = '1'  
	torch.manual_seed(SEED)

	# Create an environment just to get action space
	env = gym.make(GAME_NAME)
	action_space = env.action_space
	
	# Create a shared agent that will hold global parameters
	shared_agent = Agent(RESIZE_SHAPE, action_space)
	shared_agent.share_memory()

	# These global parameters are what will be optimized
	shared_optimizer = SharedAdam(shared_agent.parameters(), lr = LEARNING_RATE)
	shared_optimizer.share_memory()


	processes = []

	# Start a process that will test the results on the environment
	p = mp.Process(target = test, args = (NUM_WORKERS, shared_agent))
	p.start()
	processes.append(p)

	# Create training processes
	for r in range(0, NUM_WORKERS):
		p = mp.Process(target = train, args = (r, shared_agent, shared_optimizer))
		p.start()
		processes.append(p)
	
	for p in processes:
		p.join()
