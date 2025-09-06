from typing import Dict, Any, Tuple, List
from models.vectorized_trainer import VectorizedTrainer

import numpy as np
import os

class PPOAgent(VectorizedTrainer):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
    
    def get_algorithm_name(self):
        return "ppo"
    
    def training_loop(self):
        next_obs, _ = self.env.reset()
        next_done = np.zeros(self.num_env)

        for ep in range(self.num_episodes):
            for step in range(self.num_steps):
                # s_t = s_t+1
                obs = next_obs
                done = next_done
                action = self.act()
                next_obs, reward, next_terminated, next_truncated, info = self.env.step(action)
                next_done = np.logical_or(next_terminated, next_truncated)
                # here you should append to the buffer            
                
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
            mask_rewards = info["episode"]["_r"]
            mask_lengths = info["episode"]["_l"]
            rewards = info["episode"]["r"][mask_rewards]
            lengths = info["episode"]["l"][mask_lengths]
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
