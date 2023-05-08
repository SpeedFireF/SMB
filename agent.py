import torch as T
import numpy as np
import matplotlib.pyplot as plt
from CNN import CNN
import pickle
from replay_memory import ReplayMemory

class Agent:
    def __init__(self, gamma=0.99, epsilon = 1, input_dims = [240, 256, 3], batch_size=8, n_actions=12,
                 max_mem_size=10000, eps_end=0.05, eps_dec=5e-4, lr= 0.001, device=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.replay_memory = ReplayMemory(max_mem_size, input_dims, batch_size)
        self.batch_size = batch_size
        self.device = device
        self.Q_eval = CNN(action_size=n_actions, learning_rate=lr, device=self.device)
        self.loss_history = []

    
    def save_memory(self):
        T.save(self.Q_eval.state_dict(), 'CNN_model.pth')
        with open("agent.pkl", "wb") as f:
            pickle.dump(self, f)
        
        self.replay_memory.save()

    def load_memory(self):
        self.Q_eval.load_state_dict(T.load('CNN_model.pth', map_location=self.device))
        self.replay_memory = self.replay_memory.load()


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = np.array(observation)
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def play(self, observation):
        observation = np.array(observation)
        state = T.tensor([observation]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()
        return action

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.show()
    
    def learn(self):
        if self.replay_memory.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
    
        state_batch, new_state_batch, action_batch, reward_batch, terminal_batch = self.replay_memory.sample_buffer(self.device)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        self.loss_history.append(loss.item())
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
