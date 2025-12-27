"""
Simple script to test the sandbox environment.
"""
import numpy as np
from multi_drive_agent.environments import SandboxEnvironment


def main():
    """Run a simple test of the sandbox environment."""
    print("Testing Sandbox Environment")
    print("=" * 50)

    # Create environment
    config = {
        'grid_size': 15,
        'num_objects': 5,
        'num_hazards': 3,
        'max_steps': 100
    }
    env = SandboxEnvironment(config)

    # Run a few episodes
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        print("-" * 50)

        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Random action
            action = np.random.randint(0, 5)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            # Print step info
            if info.get('discovered_new'):
                print(f"  Step {steps}: Discovered new object! Reward: {reward:.2f}")

            if info.get('in_hazard'):
                features = env.get_state_features()
                risk = env.compute_risk_score()
                print(f"  Step {steps}: In hazard zone! Risk: {risk:.2f}")

        # Episode summary
        features = env.get_state_features()
        print(f"\nEpisode Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Objects Discovered: {features['total_discovered']}/{env.num_objects}")
        print(f"  Unique States Visited: {len(features['visited_states'])}")

        # Render final state
        print("\nFinal State:")
        env.render()

    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    main()
