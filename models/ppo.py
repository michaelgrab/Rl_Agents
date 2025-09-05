from models.actor_critic.a2c import A2CModel
from typing import Dict, Any, Tuple, List
from models.vectorized_trainer import VectorizedTrainer

import numpy as np

class PPOAgent(VectorizedTrainer):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.run_episode
    
    def get_algorithm_name(self):
        return "ppo"
    
    def run_episode(self, episode, render_every) -> Tuple[float, int, List[float]]:
        next_obs, _ = self.env.reset()
        episode_losses = []
        next_done = np.zeros(self.num_env)

        for update in range():
            pass
    
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
