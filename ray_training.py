import ray
from ray import tune
from ray.rllib.algorithms.sac import SAC
from race import RacingEnv  # Ensure RacingEnv is properly defined in race.py

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define environment configuration specific to an F1 racing track
env_config = {
    "export_frames": False,     # Set to True to export frames for visualization
    "max_steps": 2000,          # Max steps per episode (adjust for lap completion length)
    "reward_mode": "time",      # Reward mode that prioritizes lap time reduction
}

# SAC training configuration
config = {
    "env": RacingEnv,              # Environment class to use
    "env_config": env_config,      # Custom environment settings
    "num_workers": 4,              # Number of parallel workers for faster training
    "framework": "torch",          # Deep learning framework to use (PyTorch)
    "train_batch_size": 512,       # Batch size for each training iteration
    "gamma": 0.99,                 # Discount factor for reward calculation
    "learning_starts": 1000,       # Number of steps before starting training
    "optimization": {
        "actor_learning_rate": 3e-4,     # Learning rate for actor network
        "critic_learning_rate": 3e-4,    # Learning rate for critic network
        "entropy_learning_rate": 3e-4,   # Learning rate for entropy parameter
    },
    "num_gpus": 1,                # Use GPU if available for training
    "rollout_fragment_length": 200,  # Length of rollout fragments (for larger action spaces)
    "horizon": 2000,              # Max steps per episode (similar to max_steps in env_config)
    "checkpoint_freq": 100,       # Frequency of saving checkpoints
    "local_dir": "ray_results",   # Directory for logs and checkpoints
}

# Run the SAC algorithm with the configuration
tune.run(
    SAC,
    config=config,
    stop={"training_iteration": 1000},  # Define stopping criteria
    checkpoint_at_end=True
)

# Shutdown Ray after training
ray.shutdown()
