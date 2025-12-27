import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque


class MetricsTracker:
    """
    Track and aggregate metrics for multi-drive agent experiments.

    Tracks both per-episode and cumulative statistics for:
    - Drive signals (curiosity, safety, etc.)
    - Exploration metrics (coverage, novelty, etc.)
    - Performance metrics (rewards, episode length, etc.)
    - Meta-controller behavior (drive weights over time)
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Number of episodes for rolling statistics
        """
        self.window_size = window_size

        # Episode metrics
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_discoveries = deque(maxlen=window_size)

        # Drive-specific metrics
        self.curiosity_signals = defaultdict(list)
        self.safety_signals = defaultdict(list)
        self.drive_weights = defaultdict(lambda: deque(maxlen=window_size))

        # Exploration metrics
        self.state_coverage = []
        self.novelty_scores = deque(maxlen=window_size)
        self.risk_exposure = deque(maxlen=window_size)

        # Cumulative counters
        self.total_steps = 0
        self.total_episodes = 0
        self.unique_states_visited = set()

        # Current episode tracking
        self.current_episode_data = {
            'reward': 0.0,
            'length': 0,
            'curiosity': [],
            'safety': [],
            'drive_weights': defaultdict(list),
            'discoveries': 0,
            'risk_exposure': []
        }

    def step(self, step_data: Dict[str, Any]):
        """
        Record data from a single step.

        Args:
            step_data: Dictionary containing:
                - reward: Step reward
                - curiosity_signal: Curiosity drive value
                - safety_signal: Safety drive value
                - drive_weights: Dict of drive names to weights
                - state: Current state representation
                - risk_score: Current risk score
                - discovered_new: Whether new object was discovered
        """
        self.total_steps += 1
        self.current_episode_data['length'] += 1
        self.current_episode_data['reward'] += step_data.get('reward', 0.0)

        # Track drive signals
        if 'curiosity_signal' in step_data:
            self.current_episode_data['curiosity'].append(step_data['curiosity_signal'])

        if 'safety_signal' in step_data:
            self.current_episode_data['safety'].append(step_data['safety_signal'])

        # Track drive weights
        if 'drive_weights' in step_data:
            for drive_name, weight in step_data['drive_weights'].items():
                self.current_episode_data['drive_weights'][drive_name].append(weight)

        # Track exploration
        if 'state' in step_data:
            state_key = self._state_to_key(step_data['state'])
            self.unique_states_visited.add(state_key)

        # Track risk
        if 'risk_score' in step_data:
            self.current_episode_data['risk_exposure'].append(step_data['risk_score'])

        # Track discoveries
        if step_data.get('discovered_new', False):
            self.current_episode_data['discoveries'] += 1

    def _state_to_key(self, state: Any) -> tuple:
        """Convert state to hashable key."""
        if isinstance(state, np.ndarray):
            # Discretize continuous states for counting
            return tuple(np.round(state, decimals=1))
        elif isinstance(state, dict) and 'position' in state:
            return tuple(state['position'])
        else:
            return hash(str(state))

    def end_episode(self):
        """Mark the end of an episode and aggregate metrics."""
        self.total_episodes += 1

        # Store episode-level metrics
        self.episode_rewards.append(self.current_episode_data['reward'])
        self.episode_lengths.append(self.current_episode_data['length'])
        self.episode_discoveries.append(self.current_episode_data['discoveries'])

        # Store drive signals (mean over episode)
        if self.current_episode_data['curiosity']:
            self.curiosity_signals['mean'].append(
                np.mean(self.current_episode_data['curiosity'])
            )
            self.curiosity_signals['max'].append(
                np.max(self.current_episode_data['curiosity'])
            )

        if self.current_episode_data['safety']:
            self.safety_signals['mean'].append(
                np.mean(self.current_episode_data['safety'])
            )
            self.safety_signals['max'].append(
                np.max(self.current_episode_data['safety'])
            )

        # Store drive weights
        for drive_name, weights in self.current_episode_data['drive_weights'].items():
            if weights:
                self.drive_weights[drive_name].append(np.mean(weights))

        # Store exploration metrics
        if self.current_episode_data['risk_exposure']:
            self.risk_exposure.append(np.mean(self.current_episode_data['risk_exposure']))

        # State coverage
        self.state_coverage.append(len(self.unique_states_visited))

        # Reset current episode
        self.current_episode_data = {
            'reward': 0.0,
            'length': 0,
            'curiosity': [],
            'safety': [],
            'drive_weights': defaultdict(list),
            'discoveries': 0,
            'risk_exposure': []
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with current metrics summary
        """
        summary = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'unique_states': len(self.unique_states_visited),
        }

        # Episode statistics (recent window)
        if self.episode_rewards:
            summary['reward_mean'] = np.mean(self.episode_rewards)
            summary['reward_std'] = np.std(self.episode_rewards)
            summary['reward_max'] = np.max(self.episode_rewards)

        if self.episode_lengths:
            summary['length_mean'] = np.mean(self.episode_lengths)

        if self.episode_discoveries:
            summary['discoveries_mean'] = np.mean(self.episode_discoveries)

        # Drive statistics
        if self.curiosity_signals['mean']:
            summary['curiosity_mean'] = np.mean(list(self.curiosity_signals['mean']))

        if self.safety_signals['mean']:
            summary['safety_mean'] = np.mean(list(self.safety_signals['mean']))

        # Drive weight statistics
        for drive_name, weights in self.drive_weights.items():
            if weights:
                summary[f'{drive_name}_weight_mean'] = np.mean(weights)

        # Exploration metrics
        if self.risk_exposure:
            summary['risk_exposure_mean'] = np.mean(self.risk_exposure)

        return summary

    def get_episode_history(self, metric: str, episodes: Optional[int] = None) -> List[float]:
        """
        Get historical values for a specific metric.

        Args:
            metric: Name of metric ('reward', 'length', 'discoveries', etc.)
            episodes: Number of recent episodes to return (None for all)

        Returns:
            List of metric values
        """
        if metric == 'reward':
            data = list(self.episode_rewards)
        elif metric == 'length':
            data = list(self.episode_lengths)
        elif metric == 'discoveries':
            data = list(self.episode_discoveries)
        elif metric == 'curiosity':
            data = self.curiosity_signals.get('mean', [])
        elif metric == 'safety':
            data = self.safety_signals.get('mean', [])
        elif metric == 'coverage':
            data = self.state_coverage
        elif metric == 'risk_exposure':
            data = list(self.risk_exposure)
        else:
            return []

        if episodes is not None:
            return data[-episodes:]
        return data

    def reset(self):
        """Reset all metrics."""
        self.__init__(window_size=self.window_size)
