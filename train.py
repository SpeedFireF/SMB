from nes_py.wrappers import JoypadSpace 
import gym_super_mario_bros 
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from agent import Agent
import warnings
import torch
from Preprocessing import SkipFrame, GrayScaleObservation, ResizeObservation

    
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=128)
env = FrameStack(env, num_stack=4)

device = torch.device('cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=7, eps_end=0.01,
                input_dims=[4, 128, 128], lr=0.0001, device=device, eps_dec=1e-5)

n_games = 500
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
        agent.replay_memory.store_transition(observation, action, reward, 
                                observation_, done)
        agent.learn()
        observation = observation_

    print('iteration: ', i, 'score %.2f' % score)
    if score > best_score:
        best_score = score
        agent.save_memory()
