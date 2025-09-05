
from typing import Any, Dict, Tuple, List 
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import RecordEpisodeStatistics

import os
from .model import DeepAgent
import numpy as np


class VectorizedTrainer(DeepAgent):
    """
    Base trainer class for the vectorized environments
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
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
                env = gym.make(env_id)
                # Record statistics (reward, length, etc.)
                env = RecordEpisodeStatistics(env)
                if index == 0 and capture_video:
                    env = RecordVideo(
                        env,
                        video_folder=video_dir, 
                        name_prefix=self.experiment_logger.experiment_name,
                        episode_trigger=lambda x: x % render_every == 0
                    )
                return env
            return helper
        
        self.env = SyncVectorEnv(
            [make_env(i)
            for i in range(self.num_env) ]
        )

    def train(self) -> None:
        """Generic training loop for vectorized environments."""
        # setup variables needed for training
        self.num_steps = self.train_cfg.get("num_steps", 1000)
        total_steps = self.train_cfg.get("total_steps", 10e4)
        save_every = self.train_cfg.get("save_every", 100)
        render_every = self.train_cfg.get("render_every", 0)

        num_episodes = total_steps // (self.num_steps * self.num_env)
        self.steps_done = 0

        self.print_start_info(num_episodes)

        next_obs, _ = self.env.reset()
        next_done = np.zeros(self.num_env)

        for ep in range(num_episodes):
            episode_reward, episode_steps, episode_losses = self.run_episode(ep, render_every)
            


            for step in range(self.num_steps):
                # s_t = s_t+1
                obs = next_obs
                done = next_done
                action = self.env.action_space.sample()
                next_obs, reward, next_terminated, next_truncated = self.env.step(action)
                next_done = np.logical_or(next_terminated, next_truncated)
                # here you should append to the buffer            

                # saving model weights
                if save_every and ep > 0 and ep % save_every == 0:
                    checkpoint_path = os.path.join(
                        self.experiment_logger.checkpoints_dir, 
                        f"checkpoint_ep_{ep}.pth"
                    )
                    self.save(checkpoint_path)
            
            self.episode += 1
            self.steps_done += self.num_steps

        self.cleanup()

