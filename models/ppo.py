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

# Network from cleanrl for debug purpose --- 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
# ---

class PPOAgent(VectorizedTrainer):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.seed()
        # transfer the network parameters to the device
        if self.model_cfg.get("debug_mode", False):
            self.network = Agent(self.env).to(self.device)    
        else:
            self.network = PPOSeparateFC(self.env).to(self.device)
        lr = self.train_cfg.get("learning_rate", 3e-4)
        self.optimizer = self._setup_optimizer(self.network, lr, eps=1e-5)
        # temporary for comparison with cleanrl implementation -----
        if not self.record_stats:
            layout = {
                "Returns": {
                    "return_comparison": ["Multiline", ["global_step/reward", "charts/episodic_return"]],
                }
            }
            self.writer.add_custom_scalars(layout)
        # to be removed -----
    
    def get_algorithm_name(self):
        return "ppo"
    
    def training_loop(self):
        seed = self.env_cfg.get("seed", 1)
        
        next_obs, _ = self.env.reset(seed=seed)

        next_done = np.zeros(self.num_env)
        minibatch_number = self.train_cfg.get("mb_num", 4)
        gamma = self.train_cfg.get("gamma", 0.99)
        gae_lambda = self.train_cfg.get("lambda", 0.95)
        lr = self.train_cfg.get("learning_rate", 3e-4)
        lr_anneal = self.train_cfg.get("lr_anneal", True)
        self.trajectory_buffer = TrajectoryBuffer(self.num_steps, self.env, minibatch_number, self.device)

        for update in range(self.num_episodes):
            # learning rate annealing
            if lr_anneal:
                self.anneal_lr(update, self.num_episodes, lr)
            for step in range(self.num_steps):
                # s_t = s_t+1
                obs = next_obs
                done = next_done
                with torch.no_grad():
                    if self.debug_mode:
                        obs_tensor = torch.Tensor(obs).to(self.device)
                        self.seed()
                        action, logprob, _, value = self.network.get_action_and_value(obs_tensor)
                        value = value.flatten()
                    else:
                        action, logprob, value, _ = self.act_and_value(obs) 
                next_obs, reward, next_terminated, next_truncated, info = self.env.step(action.cpu().numpy())
                next_done = np.logical_or(next_terminated, next_truncated)

                self.trajectory_buffer.append(step, obs, action, reward, done, logprob, value)
                
                self._update_episode_info(info)

                self.steps_done += self.num_env
            with torch.no_grad():
                next_value = self.get_value(next_obs)
                if self.debug_mode:
                    next_value = next_value.reshape(1, -1)

            # LEARNING PHASE
            # Generalized advantage estimation
            self.trajectory_buffer.gae_estimation(gamma, gae_lambda, next_value, next_done)
            self.optimize()
            # saving model weights
            if self.save_every and update > 0 and update % self.save_every == 0 and not self.debug_mode:
                checkpoint_path = os.path.join(
                    self.experiment_logger.checkpoints_dir, 
                    f"checkpoint_ep_{self.episode}.pth"
                )
                self.save(checkpoint_path)

    def optimize(self):
        lr_steps = self.train_cfg.get("lr_steps", 5)
        clip_coef = self.train_cfg.get("clip_coef", 0.2)
        ent_coef = self.train_cfg.get("ent_coef", 0.01)
        vf_coef = self.train_cfg.get("vf_coef", 0.5)
        adv_norm = self.train_cfg.get("adv_norm", True)

        losses = []
        # the fraction of training data that triggered the clipped objective
        clipfracts = []

        for step in range(lr_steps):
            training_data = self.trajectory_buffer.get_batches()
            for b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values in training_data:
                if self.debug_mode:
                    _, new_logprob, entropy, new_value = self.network.get_action_and_value(b_obs, b_actions)
                    new_value = new_value.view(-1)
                else:
                    _, new_logprob, new_value, entropy = self.act_and_value(b_obs, b_actions)
                # CALCULATE PROBABILITY RATIO
                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()
                
                with torch.no_grad():
                    clipfracts+= [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    

                # advantage normalization
                if adv_norm:
                    b_advantages = self.normalize_adv(b_advantages)
                # VALUE LOSS

                if self.train_cfg.get("clip_vloss", True):
                    v_loss_unclipped = (new_value - b_returns) ** 2
                    v_clipped = b_values + torch.clamp(
                        new_value - b_values,
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:    
                    v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

                # POLICY LOSS
                p_loss_unclipped = -b_advantages * ratio
                p_loss_clipped = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                p_loss = torch.max(p_loss_unclipped, p_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = p_loss - ent_coef * entropy_loss + vf_coef * v_loss
                self.optimizer.zero_grad()
                # calcualte gradients
                loss.backward()
                # clipping gradients
                self._apply_gradient_clipping(self.network)

                self.optimizer.step()
                # record loss value
                losses.append(loss.item())
        if self.record_stats:
            self._log_batch_data(self.episode, losses)
            self.log_debug_data(self.optimizer, self.steps_done, v_loss.item(), p_loss.item(), entropy_loss.item(), clipfracts, old_approx_kl.item(), approx_kl.item())

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
    
    def save(self, path):
        """Save PPO checkpoint"""
        networks = {"network": self.network}
        optimizers = {"optimizer": self.optimizer}
        self.save_agent(path, networks, optimizers)

    def load(self, path):
        """Load PPO checkpoint"""
        networks = {"network": self.network}
        optimizers = {"optimizer": self.optimizer}
        self.load_agent(path, networks, optimizers)

    def evaluate(self):
        pass

    def normalize_adv(self, adv_tensor: torch.Tensor) -> torch.Tensor:
        """ normalize advantages """
        return (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    
    def anneal_lr(self, current_step, total_steps, original_lr):
        frac = 1 - current_step / total_steps
        new_lr = original_lr * frac
        self.optimizer.param_groups[0]["lr"] = new_lr

    def log_debug_data(self,
                        optimizer,
                        global_step, 
                        v_loss: float,
                        pg_loss: float,
                        entropy_loss: float,
                        clipfracs: List[float],
                        old_approx_kl: float,
                        approx_kl: float
                    ):
        self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", v_loss, global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss, global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl, global_step)

