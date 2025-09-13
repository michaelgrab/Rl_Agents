
from typing import Any, Dict, Tuple, List 
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from abc import abstractmethod

import os
from .model import DeepAgent
import numpy as np


class VectorizedTrainer(DeepAgent):
    """
    Base trainer class for the vectorized environments
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    @abstractmethod
    def training_loop(self):
        pass

    def train(self) -> None:
        """training in a vectorized environment."""
        # setup variables needed for training
        self.num_steps = self.train_cfg.get("num_steps", 1000)
        self.total_steps = self.train_cfg.get("total_steps", 10e4)
        self.save_every = self.train_cfg.get("save_every", 100)

        self.num_episodes = self.total_steps // (self.num_steps * self.num_env)
        self.steps_done = 0
        self.recent_rewards = []
        self.best_reward = float("-inf")
        self.worst_reward = float("inf")

        self.print_start_info(self.num_episodes)

        self.training_loop()

        self.cleanup()

    def _log_episode_data(self, ep, episode_reward, episode_length):
        """log the episode data to wandb"""
        stats_window = int(self.train_cfg.get("stats_window", 100))
        self.recent_rewards.append(episode_reward)

        if len(self.recent_rewards) > stats_window:
            self.recent_rewards.pop(0)

        self.best_reward = max(self.best_reward, episode_reward)
        self.worst_reward = min(self.worst_reward, episode_reward)
        rolling_avg = float(np.mean(self.recent_rewards))
        rolling_std = float(np.std(self.recent_rewards))

        if hasattr(self, 'writer'):
            self.writer.add_scalar("episode/reward", episode_reward, ep)
            self.writer.add_scalar("episode/steps", episode_length, ep)
            if hasattr(self, 'rolling_avg'):
                self.writer.add_scalar("episode/rolling_avg", rolling_avg, ep)
                self.writer.add_scalar("episode/rolling_std", rolling_std, ep)

    def _log_batch_data(self, ep, loss):
            if hasattr(self, 'writer'):
                self.writer.add_scalar("batch/loss", loss, ep)



    def _setup_environment(self):
        """Setup vectorized environment based on configuration """

        env_id = self.env_cfg["id"]   
        render_every = self.env_cfg.get("render_every", 0)
        capture_video = render_every > 0
        video_dir = os.path.join(self.experiment_logger.experiment_dir, "video")     
        self.num_env = self.env_cfg["num_env"]

        if self.num_env < 1:
            raise Exception("num_env must be a positive integer")

        def make_env(index):
            def helper():
                if index == 0 and capture_video:
                    env = gym.make(env_id, render_mode="rgb_array")
                    env = gym.wrappers.RecordVideo(
                        env,
                        video_folder=video_dir, 
                        name_prefix=self.experiment_logger.experiment_name,
                        episode_trigger=lambda x: x % render_every == 0 and x > 0
                    )
                else:
                    env = gym.make(env_id)
                # Record statistics (reward, length, etc.)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env
            return helper
        
        self.env = SyncVectorEnv(
            [make_env(i)
            for i in range(self.num_env) ]
        )

    def _update_episode_info(self, info: Dict[str, Any]):
        """Print episode progress and log data"""
        if "episode" in info:
            mask = info["_episode"]
            rewards = info["episode"]["r"][mask]
            lengths = info["episode"]["l"][mask]
            step = self.steps_done
            for reward, length in zip(rewards, lengths):
                self.episode += 1
                self._print_episode_progress(self.episode, reward, length)
                self._log_episode_data(self.episode, reward, length)

    def _print_episode_progress(self, episode: int, reward: float, length: int):
        """Print episode progress."""
        print(f"Ep {episode:4d} | R {reward:8.2f} | steps {length:5d}")


