import pytest
import numpy as np
from multi_drive_agent.environments.sandbox_environment import SandboxEnvironment


class TestSandboxEnvironment:
    """Tests for the SandboxEnvironment."""

    def test_initialization(self):
        """Test environment initializes correctly."""
        env = SandboxEnvironment()
        assert env.grid_size == 20
        assert env.num_objects == 10
        assert env.num_hazards == 5
        assert env.max_steps == 500

    def test_custom_config(self):
        """Test environment with custom configuration."""
        config = {
            'grid_size': 10,
            'num_objects': 5,
            'num_hazards': 2,
            'max_steps': 100
        }
        env = SandboxEnvironment(config)
        assert env.grid_size == 10
        assert env.num_objects == 5

    def test_reset(self):
        """Test environment reset."""
        env = SandboxEnvironment()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (16,)
        assert env._step_count == 0

    def test_step(self):
        """Test environment step."""
        env = SandboxEnvironment()
        env.reset()

        obs, reward, done, info = env.step(0)  # Move up
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_action_space(self):
        """Test action space is correct."""
        env = SandboxEnvironment()
        action_space = env.action_space
        assert action_space['type'] == 'discrete'
        assert action_space['n'] == 5

    def test_observation_space(self):
        """Test observation space is correct."""
        env = SandboxEnvironment()
        obs_space = env.observation_space
        assert obs_space['shape'] == (16,)

    def test_movement_actions(self):
        """Test movement actions change position."""
        env = SandboxEnvironment()
        env.reset()
        initial_pos = env.agent_pos.copy()

        # Move right
        env.step(1)
        assert not np.array_equal(env.agent_pos, initial_pos)

    def test_interaction_action(self):
        """Test interaction with objects."""
        env = SandboxEnvironment()
        env.reset()

        # Place agent near an object
        if env.objects:
            env.agent_pos = env.objects[0]['pos'].copy()
            obs, reward, done, info = env.step(4)  # Interact

            # Should discover object if close enough
            assert isinstance(reward, float)

    def test_risk_score(self):
        """Test risk score computation."""
        env = SandboxEnvironment()
        env.reset()

        risk = env.compute_risk_score()
        assert 0.0 <= risk <= 1.0

    def test_state_features(self):
        """Test state features extraction."""
        env = SandboxEnvironment()
        env.reset()

        features = env.get_state_features()
        assert 'position' in features
        assert 'visit_count' in features
        assert 'novelty_score' in features
        assert 'nearby_novel_objects' in features

    def test_episode_termination(self):
        """Test episode terminates at max steps."""
        config = {'max_steps': 10}
        env = SandboxEnvironment(config)
        env.reset()

        done = False
        steps = 0
        while not done and steps < 20:
            obs, reward, done, info = env.step(0)
            steps += 1

        assert done
        assert steps == 10

    def test_visited_states_tracking(self):
        """Test that visited states are tracked."""
        env = SandboxEnvironment()
        env.reset()

        env.step(0)
        env.step(1)

        assert len(env.visited_states) > 0
