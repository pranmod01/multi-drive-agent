from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import numpy as np


class BaseEnvironment(ABC):
    """
    Abstract base class for all environments in the multi-drive agent framework.

    This interface defines the contract that all environments must implement,
    ensuring compatibility with the agent and drive systems.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the environment.

        Args:
            config: Configuration dictionary for environment setup
        """
        self.config = config or {}
        self._step_count = 0
        self._episode_count = 0

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take in the environment

        Returns:
            observation: New state observation
            reward: Reward signal (extrinsic)
            done: Whether episode is complete
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def get_state_features(self) -> Dict[str, Any]:
        """
        Get current state features for drive calculations.

        Returns:
            Dictionary containing state features such as:
            - novelty_features: Features for curiosity calculation
            - risk_features: Features for safety assessment
            - position: Agent position (if applicable)
            - visited_states: State visitation counts
        """
        pass

    @abstractmethod
    def compute_risk_score(self) -> float:
        """
        Compute the current environmental risk score.

        This is used by the meta-controller to modulate drive weights.

        Returns:
            Risk score in [0, 1], where 0 is safe and 1 is highly risky
        """
        pass

    @property
    def observation_space(self):
        """Return the observation space specification."""
        raise NotImplementedError

    @property
    def action_space(self):
        """Return the action space specification."""
        raise NotImplementedError

    def render(self, mode: str = 'human'):
        """Render the environment (optional)."""
        pass

    def close(self):
        """Clean up environment resources."""
        pass
