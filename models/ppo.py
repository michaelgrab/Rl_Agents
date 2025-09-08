from models.vectorized_trainer import VectorizedTrainer
from games.trajectory_buffer import TrajectoryBuffer
from .network import FCNetwork

from typing import Dict, Any, Tuple, List
import numpy as np
import os
import torch
import torch.nn as nn
from torch.distributions import Categorical 

class PPOSeparateFC(FCNetwork):
    """
    Separate Actor-Critic Network for PPO (Classic Control)
    [Vectorized environment]
    """
    def __init__(self,
                envs,
                hidden_dim: int=64
                ):
        super().__init__(np.prod(envs.single_observation_space.shape),
                        hidden_dim=hidden_dim)
        self.policy_network = self._create_fc_layers(
            self.input_dim,
            envs.single_action_space.n,
            [hidden_dim, hidden_dim]
        )
        self.value_network = self._create_fc_layers(
            self.input_dim,
            1,
            [hidden_dim, hidden_dim]
        )

        self._initialize_weights()
        
    def _forward_through_layers(self, x: torch.Tensor, layers: nn.ModuleList,
                                tanh_activation: bool=True):
        if not tanh_activation:
            return super()._forward_through_layers(x, layers)
        for _, layer in enumerate(layers):
            x = torch.tanh(layer(x))
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # remove the batch dimension with squeeze
        action_logits = self._forward_through_layers(x, self.policy_network).squeeze(0)
        value = self._forward_through_layers(x, self.value_network)

        dist = Categorical(logits=action_logits)
        action = dist.sample()

        return(action, value)
    
    def _initialize_weights(self):
        for m in self.policy_network[:-1]:
            # nn.init.orthogonal_ is an in-place operation
            nn.init.constant_(m.bias, 0)
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.policy_network[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_network[-1].bias, 0)        
        for m in self.value_network[:-1]:
            nn.init.constant_(m.bias, 0)
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.value_network[-1].weight)        
        nn.init.constant_(self.value_network[-1].bias, 0)        

class PPOAgent(VectorizedTrainer):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.network = PPOSeparateFC(self.env)
        lr = self.train_cfg.get("learning_rate", 3e-4)
        self.optimizer = self._setup_optimizer(self.network, lr)
    
    def get_algorithm_name(self):
        return "ppo"
    
    def training_loop(self):
        next_obs, _ = self.env.reset()
        next_done = np.zeros(self.num_env)
        trajectory_buffer = TrajectoryBuffer(self.num_steps, self.env, self.num_env)

        for ep in range(self.num_episodes):
            for step in range(self.num_steps):
                # s_t = s_t+1
                obs = next_obs
                done = next_done
                action = self.act()
                next_obs, reward, next_terminated, next_truncated, info = self.env.step(action)
                next_done = np.logical_or(next_terminated, next_truncated)

                trajectory_buffer.append(step, obs, action, reward, done)
                
                # saving model weights
                if self.save_every and ep > 0 and ep % self.save_every == 0:
                    checkpoint_path = os.path.join(
                        self.experiment_logger.checkpoints_dir, 
                        f"checkpoint_ep_{ep}.pth"
                    )
                    self.save(checkpoint_path)
                self.print_rollout_info(info)

                self.steps_done += self.num_env
            self.episode += 1

    def print_rollout_info(self, info: Dict[str, Any]):
        """Print episode progress."""
        if "episode" in info:
            mask = info["_episode"]
            rewards = info["episode"]["r"][mask]
            lengths = info["episode"]["l"][mask]
            step = self.steps_done
            for reward, length in zip(rewards, lengths):
                print(f"Steps {step:4d} | reward {reward:8.2f} | length {length:4d}")

    def _get_eval_action(self, state):
        return super()._get_eval_action(state)
    
    def _update_networks(self, trajectory):
        return super()._update_networks(trajectory)
    
    def act_and_value(self, state):
        return super().act_and_value(state)
    
    def act(self):
        return self.env.action_space.sample()
    
    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self):
        pass
