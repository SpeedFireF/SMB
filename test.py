from nes_py.wrappers import JoypadSpace 
import gym_super_mario_bros 
from gym.spaces import Box
from gym.wrappers import FrameStack
from torchvision import transforms as T
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
from agent import Agent
from Preprocessing import SkipFrame, GrayScaleObservation, ResizeObservation
import torch
import warnings
import pickle
warnings.filterwarnings("ignore")

env = gym.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=128)
env = FrameStack(env, num_stack=4)


device = torch.device('cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

with open("agent.pkl", "rb") as f:
    agent = pickle.load(f)
agent.load_memory()


done = False
observation = env.reset()[0]
while not done:
    action = agent.play(observation)
    observation_, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    agent.replay_memory.store_transition(observation, action, reward, 
                                observation_, done)
    agent.learn()
    observation = observation_
    