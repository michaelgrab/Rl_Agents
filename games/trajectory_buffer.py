import numpy as np
from gymnasium import Env
import torch

class TrajectoryBuffer:
    """a buffer for reinforcement learing algorithms,
    that stores fixed length trajectory segments and
    loads them to the device specified
    """

    def __init__(self, length: int, env: Env, num_env: int=1, device="cpu") -> None:
        self.device = device
        self.observations = torch.zeros((length, num_env) + env.observation_space.shape).to(device)
        self.actions = torch.zeros((length, num_env)).to(device)
        self.rewards = torch.zeros((length, num_env)).to(device)
        self.dones = torch.zeros((length, num_env)).to(device)
        self.logprobs = torch.zeros((length, num_env)).to(device)
        self.values = torch.zeros((length, num_env)).to(device)

    def append(self, step, obs, act, rew, done, logprob=None, value=None) -> None:
        self.observations[step] = torch.Tensor(obs).to(self.device)
        self.actions[step] = torch.Tensor(act).to(self.device)
        self.rewards[step] = torch.Tensor(rew).to(self.device)
        self.dones[step] = torch.Tensor(done).to(self.device)
        if logprob is not None:
            self.logprobs[logprob] = torch.Tensor(obs).to(self.device)
        if value is not None:
            self.values[value] = torch.Tensor(obs).to(self.device)