from nes_py.wrappers import JoypadSpace 
import gym_super_mario_bros 
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
from agent import Agent
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=12, eps_end=0.01,
                  input_dims=[240, 256, 3], lr=0.001)
    scores, eps_history = [], []
    n_games = 500
    
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
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)