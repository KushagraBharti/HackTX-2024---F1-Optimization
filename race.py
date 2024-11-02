import gymnasium as gym
import numpy as np
import pygame
from car import Car
from track import Track

# Define screen and track dimensions
TRACK_SIZE = (2463, 1244)
MAIN_WINDOW_SIZE = (int(TRACK_SIZE[0] / 1.3), int(TRACK_SIZE[1] / 1.3))

# Define colors for easier debugging
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Race(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env_config={}):
        # Parse environment config
        self.parse_env_config(env_config)

        # Initialize Pygame and display screen
        pygame.init()
        self.win = pygame.display.set_mode(MAIN_WINDOW_SIZE)
        pygame.display.set_caption("F1 Racing Environment")

        # Set up Track and Car
        self.track = Track(self, self.env_config['csv_file'])
        self.car = Car(self, self.track)

        # Define action and observation space
        # Action space: [steering, acceleration]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [sensor distances, velocity]
        # Sensor distances (normalized) + car velocity
        obs_space_dim = self.car.N_SENSORS + 1  # Number of sensors + velocity
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_space_dim,), dtype=np.float32
        )

        # Initialize environment state
        self.reset()

    def parse_env_config(self, env_config):
        """Parse environment configuration settings."""
        # Set default values and update with env_config
        default_config = {
            'csv_file': 'Austin.csv',
            'max_steps': 1000,
            'reward_mode': 'level',
            'gui': True,
            'camera_mode': 'fixed',
            'rule_keep_on_screen': True,
            'rule_collision': True,
            'rule_max_steps': True,
        }
        self.env_config = {**default_config, **env_config}

    def reset(self):
        """Reset the environment and initialize the car state."""
        self.car.reset_game_state()  # Reset car position and state
        self.current_step = 0
        self.done = False
        self.reward = 0

        # Initial observations
        self.car.update_echo_vectors()
        distances = self.car.echo_collision_distances_interp
        velocity = np.interp(self.car.net_velocity.y, [0, self.car.MAX_VELOCITY], [-1, 1])
        observations = np.concatenate((distances, [velocity]))

        return observations

    def step(self, action):
        """Advance the environment by one time step based on the action taken."""
        if self.done:
            raise Exception("Environment is already done. Please call reset() before step().")

        # Move the car based on the action (steering, acceleration)
        self.car.move(action)
        self.current_step += 1

        # Check for collisions and goal checkpoints
        if self.env_config['rule_collision']:
            self.car.check_collision_track()
            self.car.check_collision_goal()

        # Check if max steps have been reached
        if self.env_config['rule_max_steps'] and self.current_step >= self.env_config['max_steps']:
            self.done = True

        # Calculate observations, reward, and done flag
        distances = self.car.echo_collision_distances_interp
        velocity = np.interp(self.car.net_velocity.y, [0, self.car.MAX_VELOCITY], [-1, 1])
        observations = np.concatenate((distances, [velocity]))

        # Update reward based on current level or laptime
        if self.env_config['reward_mode'] == 'level':
            self.car.update_reward_level()
        else:
            self.car.update_reward_laptime()
        
        reward = self.car.reward_step
        done = self.done or self.car.done
        info = {
            "lap_count": self.car.n_lap,
            "position": (self.car.position.x, self.car.position.y),
            "angle": self.car.angle,
            "velocity": self.car.net_velocity.y
        }

        return observations, reward, done, info

    def render(self, mode='human'):
        """Render the environment using Pygame."""
        if not self.env_config['gui']:
            return  # No rendering if GUI is disabled

        # Fill background
        self.win.fill(BLACK)
        
        # Render track and car
        self.track.render(self.win)
        if self.car.visible:
            self._draw_car()

        pygame.display.flip()

    def _draw_car(self):
        """Render the car on the track."""
        car_img = pygame.Surface((20, 10))
        car_img.fill(WHITE)
        rotated_car = pygame.transform.rotate(car_img, -self.car.angle)
        car_rect = rotated_car.get_rect(center=(self.car.position.x, self.car.position.y))
        self.win.blit(rotated_car, car_rect.topleft)

    def close(self):
        """Cleanup resources on close."""
        pygame.quit()
