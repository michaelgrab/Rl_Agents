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

    forward() method returns action logits and value
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
        action_logits = self._forward_through_layers(x, self.policy_network)
        value = self._forward_through_layers(x, self.value_network)

        return action_logits, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """ passes the observation to the policy function"""
        value = self._forward_through_layers(obs, self.value_network)
        return value

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
        # transfer the network parameters to the device
        self.network = PPOSeparateFC(self.env).to(self.device)
        lr = self.train_cfg.get("learning_rate", 3e-4)
        self.optimizer = self._setup_optimizer(self.network, lr)

    
    def get_algorithm_name(self):
        return "ppo"
    
    def training_loop(self):
        next_obs, _ = self.env.reset()
        next_done = np.zeros(self.num_env)
        minibatch_number = self.train_cfg.get("mb_num", 4)
        self.trajectory_buffer = TrajectoryBuffer(self.num_steps, self.env, minibatch_number, self.device)
        gamma = self.train_cfg.get("gamma", 0.99)
        gae_lambda = self.train_cfg.get("lambda", 0.95)

        for ep in range(self.num_episodes):
            # here add learning rate annealing
            for step in range(self.num_steps):
                # s_t = s_t+1
                obs = next_obs
                done = next_done
                with torch.no_grad():
                    action, logprob, value, _ = self.act_and_value(obs) 
                next_obs, reward, next_terminated, next_truncated, info = self.env.step(action.cpu().numpy())
                next_done = np.logical_or(next_terminated, next_truncated)

                self.trajectory_buffer.append(step, obs, action, reward, done, logprob, value)
                
                # saving model weights
                if self.save_every and ep > 0 and ep % self.save_every == 0:
                    checkpoint_path = os.path.join(
                        self.experiment_logger.checkpoints_dir, 
                        f"checkpoint_ep_{ep}.pth"
                    )
                    # self.save(checkpoint_path)
                self.print_rollout_info(info)

                self.steps_done += self.num_env
            with torch.no_grad():
                next_value = self.get_value(next_obs)

            # LEARNING PHASE
            # Generalized advantage estimation
            self.trajectory_buffer.gae_estimation(gamma, gae_lambda, next_value, next_done)
            self.optimize()
            self.episode += 1

    def optimize(self):
        lr_steps = self.train_cfg.get("lr_steps", 5)
        mb_size = self.train_cfg.get("minibatch_size", 100)
        clip_coef = self.train_cfg.get("clip_coef", 0.2)
        ent_coef = self.train_cfg.get("ent_coef", 0.01)
        vf_coef = self.train_cfg.get("vf_coef", 0.5)

        for step in range(lr_steps):
            training_data = self.trajectory_buffer.get_batches()
            for b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values in training_data:
                _, new_logprob, new_value, entropy = self.act_and_value(b_obs, b_actions)
                # CALCULATE PROBABILITY RATIO
                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()
                
                # here add advantage normalization

                # VALUE LOSS
                v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

                # POLICY LOSS
                p_loss_unclipped = -b_advantages * ratio
                p_loss_clipped = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                p_loss = torch.max(p_loss_unclipped, p_loss_clipped,).mean()

                entropy_loss = entropy.mean()
                loss = p_loss - ent_coef * entropy_loss + vf_coef * v_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(f"learning step: {step} loss: {loss:4.4f}, p_loss {p_loss:4.4f}, v_loss {v_loss:4.4f}")


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
    
    def act_and_value(self, state, action: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """select action using current policy, compute the value of the state, compute the entrop
        if provided action calculates logprob
        """
        if not isinstance(state, torch.Tensor):
            state_tensor = self._to_tensor(state, dtype=torch.float)
        else:
            state_tensor = state

        logits, value = self.network(state_tensor)
        # remove singleton dimension
        value = value.squeeze(-1)

        action_dist = Categorical(logits=logits)
        if action is None:
            action_tensor = action_dist.sample()
        else:
            action_tensor = action
        log_prob = action_dist.log_prob(action_tensor)

        entropy = action_dist.entropy()

        return action_tensor, log_prob, value, entropy
    
    def get_value(self, state) -> torch.Tensor:
        state = self._to_tensor(state, dtype=torch.float)
        return self.network.get_value(state).squeeze(-1)

    
    def act(self, state):
        action, _, _ =self.act_and_value(state)
        return action
    
    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self):
        pass
