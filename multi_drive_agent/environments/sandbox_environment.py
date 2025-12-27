import numpy as np
from typing import Dict, Tuple, Any, Optional
from .base_environment import BaseEnvironment


class SandboxEnvironment(BaseEnvironment):
    """
    Free-form sandbox environment for testing multi-drive agents.

    This environment provides:
    - A 2D grid world with various objects and hazards
    - Novelty-based exploration opportunities
    - Risk zones that trigger safety concerns
    - Rich state features for drive computation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sandbox environment.

        Config options:
            grid_size: Size of the grid world (default: 20x20)
            num_objects: Number of explorable objects (default: 10)
            num_hazards: Number of risky zones (default: 5)
            max_steps: Maximum steps per episode (default: 500)
        """
        super().__init__(config)

        self.grid_size = self.config.get('grid_size', 20)
        self.num_objects = self.config.get('num_objects', 10)
        self.num_hazards = self.config.get('num_hazards', 5)
        self.max_steps = self.config.get('max_steps', 500)

        # Action space: 0=up, 1=right, 2=down, 3=left, 4=interact
        self._action_space_n = 5

        # State tracking
        self.agent_pos = np.array([0, 0])
        self.objects = []
        self.hazards = []
        self.visited_states = {}
        self.discovered_objects = set()

        # Initialize environment
        self._initialize_world()

    def _initialize_world(self):
        """Initialize objects and hazards in the world."""
        # Place objects randomly
        self.objects = []
        for i in range(self.num_objects):
            pos = self._random_position()
            self.objects.append({
                'id': i,
                'pos': pos,
                'type': np.random.choice(['novel', 'common']),
                'discovered': False
            })

        # Place hazards randomly
        self.hazards = []
        for i in range(self.num_hazards):
            pos = self._random_position()
            radius = np.random.uniform(1.5, 3.0)
            self.hazards.append({
                'pos': pos,
                'radius': radius,
                'severity': np.random.uniform(0.3, 1.0)
            })

    def _random_position(self) -> np.ndarray:
        """Generate a random position in the grid."""
        return np.random.randint(0, self.grid_size, size=2)

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self._step_count = 0
        self.visited_states = {}
        self.discovered_objects = set()

        for obj in self.objects:
            obj['discovered'] = False

        self._initialize_world()

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Actions:
            0: Move up
            1: Move right
            2: Move down
            3: Move left
            4: Interact with nearby object
        """
        self._step_count += 1

        # Execute action
        reward = 0.0
        info = {
            'discovered_new': False,
            'in_hazard': False,
            'collision': False
        }

        if action < 4:  # Movement actions
            new_pos = self.agent_pos.copy()
            if action == 0:  # Up
                new_pos[1] = min(new_pos[1] + 1, self.grid_size - 1)
            elif action == 1:  # Right
                new_pos[0] = min(new_pos[0] + 1, self.grid_size - 1)
            elif action == 2:  # Down
                new_pos[1] = max(new_pos[1] - 1, 0)
            elif action == 3:  # Left
                new_pos[0] = max(new_pos[0] - 1, 0)

            self.agent_pos = new_pos

        elif action == 4:  # Interact
            # Check for nearby objects
            for obj in self.objects:
                dist = np.linalg.norm(self.agent_pos - obj['pos'])
                if dist < 2.0 and not obj['discovered']:
                    obj['discovered'] = True
                    self.discovered_objects.add(obj['id'])
                    reward += 1.0 if obj['type'] == 'novel' else 0.3
                    info['discovered_new'] = True
                    break

        # Update visited states
        state_key = tuple(self.agent_pos)
        self.visited_states[state_key] = self.visited_states.get(state_key, 0) + 1

        # Check if in hazard zone
        for hazard in self.hazards:
            dist = np.linalg.norm(self.agent_pos - hazard['pos'])
            if dist < hazard['radius']:
                info['in_hazard'] = True
                reward -= 0.1 * hazard['severity']

        # Episode termination
        done = self._step_count >= self.max_steps

        if done:
            self._episode_count += 1

        observation = self._get_observation()

        return observation, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Observation includes:
        - Agent position (normalized)
        - Nearby object features
        - Hazard proximity
        - Visit counts for current region
        """
        obs = np.zeros(16)

        # Agent position (normalized)
        obs[0:2] = self.agent_pos / self.grid_size

        # Nearest undiscovered object
        min_dist = float('inf')
        nearest_obj_vec = np.zeros(2)
        for obj in self.objects:
            if not obj['discovered']:
                dist = np.linalg.norm(self.agent_pos - obj['pos'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_obj_vec = (obj['pos'] - self.agent_pos) / self.grid_size
        obs[2:4] = nearest_obj_vec
        obs[4] = min(min_dist / self.grid_size, 1.0)

        # Hazard proximity
        min_hazard_dist = float('inf')
        for hazard in self.hazards:
            dist = np.linalg.norm(self.agent_pos - hazard['pos'])
            if dist < min_hazard_dist:
                min_hazard_dist = dist
                obs[5] = hazard['severity']
        obs[6] = min(min_hazard_dist / self.grid_size, 1.0)

        # Visit count for current position
        state_key = tuple(self.agent_pos)
        obs[7] = min(self.visited_states.get(state_key, 0) / 10.0, 1.0)

        # Discovery progress
        obs[8] = len(self.discovered_objects) / max(self.num_objects, 1)

        return obs

    def get_state_features(self) -> Dict[str, Any]:
        """Get state features for drive calculations."""
        state_key = tuple(self.agent_pos)
        visit_count = self.visited_states.get(state_key, 0)

        # Novelty features
        novelty_score = 1.0 / (1.0 + visit_count)

        # Nearby novel objects
        nearby_novel = 0
        for obj in self.objects:
            if not obj['discovered']:
                dist = np.linalg.norm(self.agent_pos - obj['pos'])
                if dist < 5.0:
                    nearby_novel += 1

        return {
            'position': self.agent_pos.copy(),
            'visit_count': visit_count,
            'novelty_score': novelty_score,
            'nearby_novel_objects': nearby_novel,
            'total_discovered': len(self.discovered_objects),
            'state_embedding': self._get_observation(),
            'visited_states': dict(self.visited_states)
        }

    def compute_risk_score(self) -> float:
        """Compute current risk score based on hazard proximity."""
        risk_score = 0.0

        for hazard in self.hazards:
            dist = np.linalg.norm(self.agent_pos - hazard['pos'])
            if dist < hazard['radius']:
                # Risk increases as we get closer to hazard center
                proximity_factor = 1.0 - (dist / hazard['radius'])
                risk_score = max(risk_score, proximity_factor * hazard['severity'])

        return np.clip(risk_score, 0.0, 1.0)

    @property
    def observation_space(self):
        """Return observation space shape."""
        return {'shape': (16,), 'dtype': np.float32}

    @property
    def action_space(self):
        """Return action space size."""
        return {'n': 5, 'type': 'discrete'}

    def render(self, mode: str = 'human'):
        """Simple text-based rendering."""
        if mode == 'human':
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = '.'

            # Place objects
            for obj in self.objects:
                x, y = obj['pos']
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[y, x] = 'N' if obj['type'] == 'novel' and not obj['discovered'] else 'o'

            # Place hazards
            for hazard in self.hazards:
                x, y = hazard['pos'].astype(int)
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[y, x] = 'X'

            # Place agent
            x, y = self.agent_pos
            grid[y, x] = 'A'

            print('\n' + '=' * (self.grid_size + 2))
            for row in reversed(grid):
                print('|' + ''.join(row) + '|')
            print('=' * (self.grid_size + 2))
            print(f'Step: {self._step_count}, Discovered: {len(self.discovered_objects)}/{self.num_objects}')
            print(f'Risk: {self.compute_risk_score():.2f}')
