from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np

import os

class BaseAgent(ABC):
    """
    Base class for all RL agents.
    """
    
    @abstractmethod
    def train(self) -> None:
        """Main training loop"""
        pass

    @abstractmethod
    def act(self, state) -> int:
        """Select action given current state"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint"""
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate agent performance"""
        pass

class DeepAgent(BaseAgent):
    """
    Base class with the functionality for deep leraning models
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.env_cfg = config["env"]
        self.train_cfg = config["train"]
        self.model_cfg = config.get("model", {})
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = self.train_cfg.get("gamma", 0.99)
        self.episode = 0

        experiment_name = f"{self.get_algorithm_name()}_{self.env_cfg['id']}"
        self._setup_experiment_logging(experiment_name)

        self._setup_environment()

    @abstractmethod
    def _setup_environment() -> None:
        """Setup environment based on configuration"""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name for logging."""
        pass

    def save_networks_and_optimizers(self, networks: Dict[str, torch.nn.Module], 
                                optimizers: Dict[str, torch.optim.Optimizer],
                                additional_data: Optional[Dict[str, Any]] = None) -> None:
        self.save_agent(self.get_checkpoint_path(), networks, optimizers, additional_data)

    def load_networks_and_optimizers(self, network_names: List[str], optimizer_names: List[str]) -> None:
        self.load_agent(self.get_checkpoint_path(), network_names, optimizer_names)

    def get_checkpoint_path(self, episode: Optional[int] = None) -> str:
        if episode is None:
            episode = self.episode
        return os.path.join(
            self.experiment_logger.checkpoints_dir,
            f"checkpoint_ep_{episode}.pth"
        )
    
    def _setup_experiment_logging(self, experiment_name: str) -> None:
        """Setup experiment logging infrastructure."""
        from games.experiment_logger import ExperimentLogger
        
        self.experiment_logger = ExperimentLogger(
            experiment_name=experiment_name,
            base_dir=self.train_cfg.get("experiments_dir", "experiments")
        )
        
        logdir = os.path.join(self.experiment_logger.experiment_dir, "tensorboard")
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        
        self.experiment_logger.log_config(self.config)

    def _apply_gradient_clipping(self, network: nn.Module) -> None:
        """Apply gradient clipping to prevent exploding gradients."""
        grad_clip = self.train_cfg.get("gradient_clip", 10.0)
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)

    def cleanup(self) -> None:
        if hasattr(self, 'writer'):
            self.writer.close()
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'game'):
            self.game.close()

    def print_start_info(self, num_episodes) -> None:    
        print(f"Training {self.get_algorithm_name()} on {self.env_cfg['id']}")
        print(f"Device: {self.device} | Episodes: {num_episodes}")
        if hasattr(self, 'writer'):
            print(f"Log dir: {self.writer.log_dir}")
        print("=" * 60)

    def _setup_optimizer(self, network: nn.Module, lr: Optional[float] = None) -> torch.optim.Optimizer:
        """Setup optimizer for the given network."""
        if lr is None:
            lr = self.train_cfg.get("learning_rate", 2.5e-4)
            
        if self.train_cfg.get("use_rmsprop", False):
            alpha = self.train_cfg.get("rmsprop_alpha", 0.95)
            momentum = self.train_cfg.get("rmsprop_momentum", 0.95)
            eps = self.train_cfg.get("rmsprop_eps", 0.01)
            
            return optim.RMSprop(
                network.parameters(),
                lr=lr,
                alpha=alpha,
                momentum=momentum,
                eps=eps
            )
        else:
            return optim.Adam(network.parameters(), lr=lr)    
        
    def _to_tensor(self, data, dtype=None) -> torch.Tensor:
        """Convert numpy array or list to tensor on correct device."""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)
            
        if dtype is not None:
            tensor = tensor.to(dtype)
            
        return tensor.to(self.device)


        