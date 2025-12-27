import pytest
import tempfile
import os
from multi_drive_agent.utils.config import Config, load_config


class TestConfig:
    """Tests for the Config class."""

    def test_initialization(self):
        """Test config initializes correctly."""
        config = Config({'key': 'value'})
        assert config.get('key') == 'value'

    def test_dot_notation_get(self):
        """Test getting values with dot notation."""
        config = Config({
            'agent': {
                'learning_rate': 0.001,
                'gamma': 0.99
            }
        })

        assert config.get('agent.learning_rate') == 0.001
        assert config.get('agent.gamma') == 0.99

    def test_dot_notation_set(self):
        """Test setting values with dot notation."""
        config = Config()

        config.set('agent.learning_rate', 0.01)
        assert config.get('agent.learning_rate') == 0.01

    def test_default_value(self):
        """Test default value when key not found."""
        config = Config()

        assert config.get('nonexistent', 'default') == 'default'

    def test_update(self):
        """Test updating configuration."""
        config = Config({'a': 1, 'b': {'c': 2}})

        config.update({'b': {'c': 3, 'd': 4}})

        assert config.get('a') == 1
        assert config.get('b.c') == 3
        assert config.get('b.d') == 4

    def test_to_dict(self):
        """Test converting to dictionary."""
        config_dict = {'a': 1, 'b': {'c': 2}}
        config = Config(config_dict)

        result = config.to_dict()
        assert result == config_dict

    def test_from_dict(self):
        """Test creating from dictionary."""
        config_dict = {'a': 1, 'b': 2}
        config = Config.from_dict(config_dict)

        assert config.get('a') == 1

    def test_save_and_load(self):
        """Test saving and loading YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'test_config.yaml')

            config = Config({'test': {'value': 42}})
            config.save(config_path)

            loaded = Config.from_yaml(config_path)
            assert loaded.get('test.value') == 42

    def test_dictionary_access(self):
        """Test dictionary-style access."""
        config = Config({'key': 'value'})

        assert config['key'] == 'value'
        config['new_key'] = 'new_value'
        assert config['new_key'] == 'new_value'
