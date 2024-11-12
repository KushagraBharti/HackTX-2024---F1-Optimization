from ray import tune
from race import Race
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'Monza_track_extra_wide_contour.png')

tune.run(
    "SAC",  # Reinforcement learning agent
    name="TrainingAgata",  # Session name for organizing results
    checkpoint_freq=1,  # Save checkpoint every 100 iterations
    checkpoint_at_end=True,  # Save final checkpoint when training completes
    reuse_actors=True,  # Reuse workers to save on setup time
    storage_path=r"C:\\Users\\kusha\\OneDrive\Documents\\CS Projects\\HackTX-2024---F1-Optimization\\ray_results",  # Updated with absolute path
    config={
        "env": Race,  # Custom environment to train in
        "seed": 204060,  # Random seed for reproducibility
        "api_stack": {
            "enable_rl_module_and_learner": False,
            "enable_env_runner_and_connector_v2": False
        },
        "num_workers": 8,  # Adjusted based on laptop specs
        "num_cpus_per_worker": 2,
        "num_gpus": 1,  # GPU usage
        "env_config": {
            "export_frames": False,
            "export_states": False,
        },
    },
    stop={
        "training_iteration": 3,
    },
)
