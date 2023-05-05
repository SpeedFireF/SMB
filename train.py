from nes_py.wrappers import JoypadSpace 
import gym_super_mario_bros 
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
from agent import Agent
import numpy as np
import warnings
import torch
warnings.filterwarnings("ignore")

    
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

device = 'cpu'

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=4, n_actions=12, eps_end=0.01,
                input_dims=[240, 256, 3], lr=0.001, device=device)

n_games = 1
best_score = 0


for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()[0]
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        agent.store_transition(observation, action, reward, 
                                observation_, done)
        agent.learn()
        observation = observation_

    print('iteration: ', i, 'score %.2f' % score)
    if score > best_score:
        best_score = score
        agent.save_memory()