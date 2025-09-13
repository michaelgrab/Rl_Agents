import numpy as np
import torch
import torch.nn as nn
import os
from typing import Any, Dict, List, Tuple, Optional
from abc import abstractmethod
from .model import DeepAgent

class BaseTrainer(DeepAgent):
    """
    Base trainer class that handles common training patterns.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    @abstractmethod
    def run_episode(self, episode: int, render_every: int) -> Tuple[float, int, List[float]]:
        """Run single episode and return (reward, steps, losses)."""
        pass
    
    def get_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override to add algorithm-specific metrics."""
        return episode_data

    def train(self) -> None:
        """Generic training loop for all algorithms."""
        num_episodes = self.train_cfg.get("num_episodes", 1000)
        save_every = self.train_cfg.get("save_every", 100)
        render_every = self.train_cfg.get("render_every", 0)
        stats_window = int(self.train_cfg.get("stats_window", 100))

        self.print_start_info(num_episodes)

        recent_rewards = []
        best_reward = float("-inf")
        worst_reward = float("inf")

        for ep in range(num_episodes):
            episode_reward, episode_steps, episode_losses = self.run_episode(ep, render_every)
            
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > stats_window:
                recent_rewards.pop(0)
            
            best_reward = max(best_reward, episode_reward)
            worst_reward = min(worst_reward, episode_reward)
            rolling_avg = float(np.mean(recent_rewards))
            rolling_std = float(np.std(recent_rewards))
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

            if hasattr(self, 'writer'):
                self.writer.add_scalar("episode/reward", episode_reward, ep)
                self.writer.add_scalar("episode/steps", episode_steps, ep)
                self.writer.add_scalar("episode/avg_loss", avg_loss, ep)
                if hasattr(self, 'rolling_avg'):
                    self.writer.add_scalar("episode/rolling_avg", rolling_avg, ep)
            
                episode_data = {
                'episode_steps': episode_steps,
                'reward': episode_reward,
                'loss': avg_loss,
                'best_reward': best_reward,
                'worst_reward': worst_reward,
                'rolling_avg': rolling_avg,
                'rolling_std': rolling_std,
                'algorithm': self.get_algorithm_name(),
            }
            
            episode_data = self.get_episode_metrics(episode_data)
            self._log_episode_stats(ep, **episode_data)
            
            self.print_episode_progress(ep, episode_reward, episode_steps, avg_loss, rolling_avg)

            if save_every and ep > 0 and ep % save_every == 0:
                checkpoint_path = os.path.join(
                    self.experiment_logger.checkpoints_dir, 
                    f"checkpoint_ep_{ep}.pth"
                )
                self.save(checkpoint_path)
                        
            self.episode += 1

        self.cleanup()

    def print_episode_progress(self, episode: int, reward: float, steps: int, loss: float, rolling_avg: float):
        """Print episode progress."""
        print(f"Ep {episode:4d} | R {reward:8.2f} | steps {steps:5d} | loss {loss:7.4f} | avg {rolling_avg:6.2f}")

    def _log_episode_stats(self, episode: int, **kwargs) -> None:
        base_episode_data = {
            'episode': episode,
            'environment': self.env_cfg['id'],
            'gamma': self.gamma,
        }
        base_episode_data.update(kwargs)
        self.experiment_logger.log_episode(base_episode_data)


    def evaluate_agent(self, num_episodes: int, networks_to_eval: List[nn.Module]) -> Tuple[float, float]:
        """Common evaluation pattern for all agents."""
        training_modes = []
        for network in networks_to_eval:
            training_modes.append(network.training)
            network.eval()
        
        env_or_game = getattr(self, 'env', None) or getattr(self, 'game', None)
        max_steps = getattr(self, 'max_episode_length', None)
        
        result = self._evaluate_agent(num_episodes, env_or_game, max_steps)
        
        for network, training_mode in zip(networks_to_eval, training_modes):
            network.train(training_mode)
        
        return result

    def _evaluate_agent(self, num_episodes: int, env_or_game, max_steps: int = None) -> Tuple[float, float]:
        """Common evaluation pattern."""
        rewards = []
        
        for _ in range(num_episodes):
            if hasattr(env_or_game, 'reset'):
                obs = env_or_game.reset()
                if isinstance(obs, tuple):
                    state = obs[0]
                else:
                    state = obs
            else:
                state = env_or_game.reset()
                
            total_reward = 0.0
            steps = 0
            
            while True:
                action = self._get_eval_action(state)
                
                step_result = env_or_game.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    next_state, reward, done, _ = step_result
                else:
                    next_state, reward, done = step_result
                
                state = next_state
                total_reward += float(reward)
                steps += 1
                
                if done or (max_steps and steps >= max_steps):
                    break
                    
            rewards.append(total_reward)
            
        return float(np.mean(rewards)), float(np.std(rewards))

    def _setup_environment(self) -> None:
        """Setup environment based on configuration"""
        env_id = self.env_cfg["id"]
        
        if "ALE/" in env_id or "Atari" in env_id:
            from games.atari_env import AtariGame
            self.game = AtariGame(
                env_id, 
                self.env_cfg.get("kwargs", {}), 
                self.env_cfg.get("seed")
            )
            self.env = self.game
            self.is_atari_env = True
        else:
            import gymnasium as gym
            render_enabled = self.train_cfg.get("render_every", 0) > 0
            render_mode = "human" if render_enabled else None
            self.env = gym.make(env_id, render_mode=render_mode)
            self.is_atari_env = False
    
    @abstractmethod
    def _get_eval_action(self, state) -> int:
        """Get action for evaluation."""
        pass