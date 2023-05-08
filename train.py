from nes_py.wrappers import JoypadSpace 
import gym_super_mario_bros 
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
from agent import Agent
import warnings
import torch
from Preprocessing import SkipFrame, GrayScaleObservation, ResizeObservation

    
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


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=16, n_actions=12, eps_end=0.01,
                input_dims=[4, 128, 128], lr=0.001, device=device)

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
