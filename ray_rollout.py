import ray
from ray.rllib.algorithms.sac import SAC
from race import RacingEnv

# Load the latest checkpoint (adjust the path as needed)
checkpoint_path = "path/to/latest/checkpoint"

# Load the trained SAC model
agent = SAC.from_checkpoint(checkpoint_path)

# Initialize the environment for testing
env = RacingEnv({})

# Run several episodes to evaluate
for episode in range(5):  # Test over 5 episodes
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.compute_single_action(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # Optional: render for visualization
    print(f"Episode {episode + 1} reward: {total_reward}")
