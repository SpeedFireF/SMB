from nes_py.wrappers import JoypadSpace 
import gym_super_mario_bros 
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
from agent import Agent
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    agent = Agent(epsilon=0, n_actions=12,
                  input_dims=[240, 256, 3])
    agent.load_memory()

    done = False
    observation = env.reset()[0]
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation = observation_