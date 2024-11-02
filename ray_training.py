import ray
import pandas as pd
from ray import tune
from ray.rllib.algorithms.sac import SAC
from race import RacingEnv  # Ensure RacingEnv is defined correctly in race.py

# Load the CSV file data
csv_file_path = 'Monza.csv'  # Ensure this is the correct path to Monza.csv
track_data = pd.read_csv(csv_file_path, header=None).values  # Load as numpy array for easy integration

# Initialize Ray with minimized background processing and logging
ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

# Define the environment configuration specific to your racing setup
env_config = {
    "export_frames": False,     # Disable frame export unless needed
    "max_steps": 1000,          # Max steps per episode (adjust if needed for lap length)
    "reward_mode": "time",      # This should match your reward design in `car.py`
    "track_data": track_data    # Pass the track data into the environment
}

# Define SAC training configuration with essential parameters
config = {
    "env": RacingEnv,              # Use the RacingEnv class
    "env_config": env_config,      # Pass environment-specific settings
    "num_workers": 0,              # Set to 0 to reduce multi-processing issues on Windows
    "framework": "torch",          # Use PyTorch for model training
    "train_batch_size": 512,       # Batch size for each training step
    "gamma": 0.99,                 # Discount factor for future rewards
    "learning_starts": 1000,       # Steps before learning begins
    "optimization": {
        "actor_learning_rate": 3e-4,     # Learning rate for the actor network
        "critic_learning_rate": 3e-4,    # Learning rate for the critic network
        "entropy_learning_rate": 3e-4,   # Learning rate for entropy parameter
    },
    "num_gpus": 0,                # Set to 0 if you don’t have a GPU, or adjust if available
    "rollout_fragment_length": 200,  # Fragment length for rollouts
    "horizon": 1000,              # Episode horizon (matches max_steps in env_config)
    "checkpoint_freq": 100,       # Save a checkpoint every 100 iterations
    "local_dir": "ray_results",   # Directory to save logs and checkpoints
}

# Run SAC training using the defined configuration
tune.run(
    SAC,
    config=config,
    stop={"training_iteration": 1000},  # Define stopping criteria
    checkpoint_at_end=True
)

# Shutdown Ray after training
ray.shutdown()
