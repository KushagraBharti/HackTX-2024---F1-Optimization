import gym
import numpy as np
from gym import spaces
from car import Car

class RacingEnv(gym.Env):
    def __init__(self, config):
        super(RacingEnv, self).__init__()

        # Define action and observation spaces
        self.track_data = config.get("track_data", None)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)  # steering, throttle
        self.observation_space = spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)  # Example: 4 sensor readings, speed, and angle

        # Initialize the car
        self.car = Car()
        self.max_steps = config.get("max_steps", 1000)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.car.reset()
        return self._get_observation()

    def step(self, action):
        self.car.apply_action(action)
        self.current_step += 1

        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps or self.car.has_crashed()

        return observation, reward, done, {}

    def _get_observation(self):
        return np.concatenate((self.car.get_sensor_readings(), [self.car.velocity, self.car.angle]))

    def _calculate_reward(self):
        reward = self.car.update_reward_level()
        return reward

    def render(self, mode='human'):
        pass  # Optional visualization code

