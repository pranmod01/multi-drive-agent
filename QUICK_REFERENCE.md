# Quick Reference Guide

## Installation & Setup

```bash
# Clone and setup
cd multi-drive-agent
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# Verify installation
python3 examples/test_environment.py
python3 -m pytest tests/
```

## Project File Organization

### Core Implementation Files

| File | Purpose | Status |
|------|---------|--------|
| `environments/base_environment.py` | Environment interface | ✅ Complete |
| `environments/sandbox_environment.py` | Sandbox test environment | ✅ Complete |
| `utils/metrics.py` | Metrics tracking | ✅ Complete |
| `utils/logger.py` | Experiment logging | ✅ Complete |
| `utils/config.py` | Configuration management | ✅ Complete |
| `agents/` | Agent implementations | 📋 Phase 1 |
| `drives/` | Drive modules | 📋 Phase 2+ |
| `meta_controller/` | Drive arbitration | 📋 Phase 5 |

### Configuration Files

- `configs/default_config.yaml` - Default experiment configuration
- Edit this file to change experiment parameters

### Documentation Files

- `README.md` - Project overview
- `ROADMAP.md` - Development timeline
- `SETUP_SUMMARY.md` - Detailed setup info
- `QUICK_REFERENCE.md` - This file

## Common Tasks

### Running the Example

```bash
python3 examples/test_environment.py
```

This will run 3 episodes with random actions in the sandbox environment.

### Running Tests

```bash
# All tests
python3 -m pytest tests/

# Specific test file
python3 -m pytest tests/test_environment.py

# Verbose output
python3 -m pytest tests/ -v

# With coverage
python3 -m pytest tests/ --cov=multi_drive_agent
```

### Creating a New Environment

```python
from multi_drive_agent.environments import BaseEnvironment
import numpy as np

class MyEnvironment(BaseEnvironment):
    def reset(self):
        # Return initial observation
        return np.zeros(10)

    def step(self, action):
        # Return (observation, reward, done, info)
        obs = np.zeros(10)
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def get_state_features(self):
        # Return features dict for drives
        return {
            'position': np.zeros(2),
            'novelty_score': 0.0,
        }

    def compute_risk_score(self):
        # Return risk in [0, 1]
        return 0.0

    @property
    def observation_space(self):
        return {'shape': (10,), 'dtype': np.float32}

    @property
    def action_space(self):
        return {'n': 4, 'type': 'discrete'}
```

### Using Metrics Tracker

```python
from multi_drive_agent.utils import MetricsTracker

tracker = MetricsTracker(window_size=100)

# During training
for episode in range(num_episodes):
    # ... episode loop ...
    for step in range(max_steps):
        # Take action, get reward, etc.

        step_data = {
            'reward': reward,
            'curiosity_signal': curiosity_value,
            'safety_signal': safety_value,
            'drive_weights': {'curiosity': 0.7, 'safety': 0.3},
            'state': state,
            'risk_score': risk,
        }
        tracker.step(step_data)

    tracker.end_episode()

    # Get summary every 10 episodes
    if episode % 10 == 0:
        summary = tracker.get_summary()
        print(f"Episode {episode}: {summary}")
```

### Using Logger

```python
from multi_drive_agent.utils import ExperimentLogger

logger = ExperimentLogger(
    experiment_name='my_experiment',
    log_dir='logs',
    use_tensorboard=True
)

# Log hyperparameters
config = {'learning_rate': 0.001, 'gamma': 0.99}
logger.log_hyperparameters(config)

# Log metrics during training
metrics = {'reward': 10.5, 'loss': 0.23}
logger.log_metrics(metrics, step=100)

# Info messages
logger.info("Training started")

# Close when done
logger.close()
```

### Loading Configuration

```python
from multi_drive_agent.utils import load_config

# Load default config
config = load_config()

# Load custom config
config = load_config('configs/my_config.yaml')

# Load with overrides
overrides = {'agent.learning_rate': 0.01}
config = load_config(overrides=overrides)

# Access values
lr = config.get('agent.learning_rate')
grid_size = config.get('environment.grid_size', default=20)

# Modify values
config.set('agent.gamma', 0.95)

# Save modified config
config.save('configs/modified_config.yaml')
```

## Sandbox Environment Details

### State Representation

The sandbox environment provides a 16-dimensional observation vector:

| Indices | Description |
|---------|-------------|
| 0-1 | Agent position (normalized) |
| 2-3 | Direction to nearest undiscovered object |
| 4 | Distance to nearest undiscovered object |
| 5 | Nearest hazard severity |
| 6 | Distance to nearest hazard |
| 7 | Visit count for current position |
| 8 | Discovery progress (discovered/total) |
| 9-15 | Reserved for future use |

### Actions

| Action | Effect |
|--------|--------|
| 0 | Move up |
| 1 | Move right |
| 2 | Move down |
| 3 | Move left |
| 4 | Interact with nearby object |

### Configuration Options

```yaml
environment:
  grid_size: 20        # Size of grid (20x20)
  num_objects: 10      # Number of explorable objects
  num_hazards: 5       # Number of hazard zones
  max_steps: 500       # Max steps per episode
```

### State Features for Drives

```python
features = env.get_state_features()
# Returns:
# {
#   'position': np.array([x, y]),
#   'visit_count': int,
#   'novelty_score': float,  # 1/(1 + visit_count)
#   'nearby_novel_objects': int,
#   'total_discovered': int,
#   'state_embedding': np.array (16-dim),
#   'visited_states': dict
# }
```

### Risk Score

```python
risk = env.compute_risk_score()
# Returns: float in [0, 1]
# - 0.0 = completely safe
# - 1.0 = maximum danger
# Based on proximity to hazards
```

## Development Workflow

### Phase 1: RL Baseline (Next Steps)

1. Install RL dependencies:
   ```bash
   python3 -m pip install stable-baselines3 torch
   ```

2. Create agent interface in `multi_drive_agent/agents/base_agent.py`

3. Implement PPO wrapper in `multi_drive_agent/agents/ppo_agent.py`

4. Create training script in `experiments/train_baseline.py`

5. Test training:
   ```bash
   python3 experiments/train_baseline.py
   ```

### Adding a New Drive

1. Create drive file: `multi_drive_agent/drives/my_drive.py`

2. Implement drive interface:
   ```python
   from abc import ABC, abstractmethod

   class BaseDrive(ABC):
       @abstractmethod
       def compute_reward(self, state_features):
           pass
   ```

3. Add to configuration: `configs/default_config.yaml`

4. Add tests: `tests/test_my_drive.py`

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure package is installed
python3 -m pip install -e .

# Verify installation
python3 -c "import multi_drive_agent; print('OK')"
```

### Tests Failing

```bash
# Run with verbose output
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_environment.py::TestSandboxEnvironment::test_reset -v
```

### Environment Not Rendering

The sandbox environment uses text-based rendering:
```python
env.render(mode='human')
```

For visual rendering, you'll need to implement a custom renderer.

## Useful Commands

```bash
# Check code style
python3 -m pylint multi_drive_agent/

# Format code
python3 -m black multi_drive_agent/

# Type checking
python3 -m mypy multi_drive_agent/

# Generate documentation
python3 -m pydoc multi_drive_agent.environments.sandbox_environment
```

## Resources

- **Stable Baselines3 Docs:** https://stable-baselines3.readthedocs.io/
- **Gymnasium Docs:** https://gymnasium.farama.org/
- **PyTorch Docs:** https://pytorch.org/docs/

## Getting Help

1. Check the documentation files (README, ROADMAP, SETUP_SUMMARY)
2. Review example scripts in `examples/`
3. Look at test files in `tests/` for usage examples
4. Read docstrings in the code
