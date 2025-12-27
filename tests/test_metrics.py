import pytest
import numpy as np
from multi_drive_agent.utils.metrics import MetricsTracker


class TestMetricsTracker:
    """Tests for the MetricsTracker."""

    def test_initialization(self):
        """Test metrics tracker initializes correctly."""
        tracker = MetricsTracker(window_size=50)
        assert tracker.window_size == 50
        assert tracker.total_steps == 0
        assert tracker.total_episodes == 0

    def test_step_tracking(self):
        """Test step data is tracked."""
        tracker = MetricsTracker()

        step_data = {
            'reward': 1.0,
            'curiosity_signal': 0.5,
            'safety_signal': 0.3,
            'risk_score': 0.2
        }

        tracker.step(step_data)
        assert tracker.total_steps == 1
        assert tracker.current_episode_data['reward'] == 1.0

    def test_episode_tracking(self):
        """Test episode completion is tracked."""
        tracker = MetricsTracker()

        for _ in range(10):
            tracker.step({'reward': 1.0})

        tracker.end_episode()
        assert tracker.total_episodes == 1
        assert len(tracker.episode_rewards) == 1
        assert tracker.episode_rewards[0] == 10.0

    def test_drive_weights_tracking(self):
        """Test drive weights are tracked."""
        tracker = MetricsTracker()

        step_data = {
            'drive_weights': {
                'curiosity': 0.7,
                'safety': 0.3
            }
        }

        tracker.step(step_data)
        tracker.end_episode()

        summary = tracker.get_summary()
        assert 'curiosity_weight_mean' in summary
        assert 'safety_weight_mean' in summary

    def test_get_summary(self):
        """Test summary generation."""
        tracker = MetricsTracker()

        for i in range(5):
            for _ in range(10):
                tracker.step({
                    'reward': 1.0,
                    'curiosity_signal': 0.5,
                    'safety_signal': 0.3
                })
            tracker.end_episode()

        summary = tracker.get_summary()
        assert 'total_episodes' in summary
        assert summary['total_episodes'] == 5
        assert 'reward_mean' in summary

    def test_episode_history(self):
        """Test getting episode history."""
        tracker = MetricsTracker()

        for i in range(10):
            tracker.step({'reward': float(i)})
            tracker.end_episode()

        history = tracker.get_episode_history('reward', episodes=5)
        assert len(history) == 5

    def test_reset(self):
        """Test metrics reset."""
        tracker = MetricsTracker()

        tracker.step({'reward': 1.0})
        tracker.end_episode()

        tracker.reset()
        assert tracker.total_steps == 0
        assert tracker.total_episodes == 0
