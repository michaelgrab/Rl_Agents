import numpy as np
from gymnasium.vector import VectorEnv
import torch
from typing import Tuple

class TrajectoryBuffer:
    """a buffer for reinforcement learing algorithms,
    that stores fixed length trajectory segments and
    loads them to the device specified, compatible with vectorized env
    """

    def __init__(self, length: int, env: VectorEnv, num_minibatches: int=10, device="cpu") -> None:
        self.device = device
        self.length = length
        self.env = env
        self.num_env = env.num_envs
        self.batch_size = self.length * self.num_env
        self.indicies = np.arange(self.batch_size)
        self.mb_length = self.batch_size // num_minibatches
        self.observations = torch.zeros((length, self.num_env) + env.single_observation_space.shape).to(device)
        self.actions = torch.zeros((length, self.num_env)).to(device)
        self.rewards = torch.zeros((length, self.num_env)).to(device)
        self.dones = torch.zeros((length, self.num_env)).to(device)
        self.logprobs = torch.zeros((length, self.num_env)).to(device)
        self.values = torch.zeros((length, self.num_env)).to(device)
        self.advantages = torch.zeros((length, self.num_env)).to(device)
        self.returns = torch.zeros((length, self.num_env)).to(device)

    def append(self, step, obs, act, rew, done, logprob=None, value=None) -> None:
        self.observations[step] = self._to_tensor(obs, dtype=torch.float)
        self.actions[step] = self._to_tensor(act, dtype=torch.long)
        self.rewards[step] = self._to_tensor(rew, dtype=torch.float)
        self.dones[step] = self._to_tensor(done, dtype=torch.float)
        if logprob is not None:
            self.logprobs[step] = self._to_tensor(logprob, dtype=torch.float)
        if value is not None:
            self.values[step] = self._to_tensor(value, dtype=torch.float)
    
    def gae_estimation(self, gamma: float, gae_lambda: float, next_value, next_done):
        next_value = self._to_tensor(next_value, dtype=torch.float)
        next_done = self._to_tensor(next_done, dtype=torch.float)
        with torch.no_grad():
            lastgae = 0
            for t in reversed(range(self.length)):
                if t == self.length - 1:
                    termination_mask = 1.0 - next_done 
                    bootstrap_value = next_value
                else:
                    termination_mask = 1.0 - self.dones[t]
                    bootstrap_value = self.values[t+1]

                delta = self.rewards[t] + gamma * bootstrap_value * termination_mask - self.values[t]
                self.advantages[t] = lastgae = delta + gamma * gae_lambda * termination_mask * lastgae
            self.returns = self.advantages + self.values 

    def _to_tensor(self, data, dtype=None) -> torch.Tensor:
        """Convert numpy array or list to tensor on correct device."""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif not isinstance(data, torch.Tensor):
            tensor = torch.tensor(data)
        else:
            tensor = data    
        if dtype is not None and data.dtype is not dtype:
            tensor = tensor.to(dtype)
            
        return tensor.to(self.device)
    
    def get_batches(self):
        """
        generator function that fetches randomly minibatch of taining data
        """
        b_obs = self.observations.reshape((-1,) + self.env.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.env.single_action_space.shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        # using numpy to randomly shuffle batch indecies
        np.random.shuffle(self.indicies)
        for start in range(0, self.batch_size, self.mb_length):
            end = start + self.mb_length
            mb_ind = self.indicies[start:end]
            yield b_obs[mb_ind], b_logprobs[mb_ind], b_actions[mb_ind], b_advantages[mb_ind], b_returns[mb_ind], b_values[mb_ind]
