
from typing import Any, Dict, Tuple, List 
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import RecordEpisodeStatistics
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
                else:
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
        """training in a vectorized environment."""
        # setup variables needed for training
        self.num_steps = self.train_cfg.get("num_steps", 1000)
        self.total_steps = self.train_cfg.get("total_steps", 10e4)
        self.save_every = self.train_cfg.get("save_every", 100)

        self.num_episodes = self.total_steps // (self.num_steps * self.num_env)
        self.steps_done = 0

        self.print_start_info(self.num_episodes)

        self.training_loop()

        self.cleanup()


