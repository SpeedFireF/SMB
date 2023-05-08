import numpy as np
import pickle
import torch as T

class ReplayMemory:
    def __init__(self, max_size, input_dims, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.name = 'memory.pkl'
        self.batch_size = batch_size
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1
    
    def sample_buffer(self, device):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch]).to(device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(device)
        
        return state_batch, new_state_batch, action_batch, reward_batch, terminal_batch
    def save(self):
        with open(self.name, 'wb') as f:
            pickle.dump(self, f)
    def load(self):
        with open(self.name, 'rb') as f:
            return pickle.load(f)