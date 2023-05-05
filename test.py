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
device = torch.device('cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')


agent = Agent(gamma=0.99, batch_size=4, n_actions=12,
                input_dims=[240, 256, 3], lr=0.001, device=device)
agent.load_memory()


done = False
observation = env.reset()[0]
while not done:
    action = agent.play(observation)
    observation_, reward, terminated, truncated, info = env.step(action)
    agent.store_transition(observation, action, reward, 
                                observation_, done)
    agent.learn()
    done = terminated or truncated
    observation = observation_