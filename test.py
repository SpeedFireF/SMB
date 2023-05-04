from nes_py.wrappers import JoypadSpace 
import gym_super_mario_bros 
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
from agent import Agent
import torch
import warnings
warnings.filterwarnings("ignore")


env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
device = 'cpu'

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'


agent = Agent(device=device)
agent.load_memory()

done = False
observation = env.reset()[0]
while not done:
    action = agent.play(observation)
    observation_, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    observation = observation_