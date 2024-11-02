import numpy as np
import pygame
import utils

# Constants for track and car
TRACK_SIZE = (2463, 1244)
MAIN_WINDOW_SIZE = tuple(int(x / 1.3) for x in TRACK_SIZE)
START_POS = (1501, 870)

class Car:
    # Maximum constraints for car movement
    MAX_ACCELERATION = 1.0
    MAX_VELOCITY = 8.0
    ROTATION_VELOCITY = 4.0
    N_SENSORS = 7

    def __init__(self, game, track):
        self.game = game
        self.track = track
        self.visible = True
        self.reset_game_state()  # Reset initial car state

    def reset_game_state(self, x=START_POS[0], y=START_POS[1], ang=-92, vel_x=0, vel_y=0, level=0):
        # Initialize car position, velocity, angle, and state variables
        self.position = pygame.math.Vector2(x, y)
        self.net_velocity = pygame.math.Vector2(vel_x, vel_y)
        self.angle = ang
        self.level = level
        self.level_previous = level
        self.framecount_goal = 0
        self.framecount_total = 0
        self.n_lap = 0
        self.reward_step = 0
        self.reward_total = 0
        self.done = False
        self.action = np.array([0, 0])
        self.update_goal_vectors()
        self.update_echo_vectors()

    def update_state(self, car_state):
        # Update car's state variables
        self.position, self.angle, self.net_velocity.y, self.action_state = car_state

    def update_goal_vectors(self):
        # Update vectors for current, next, and previous goals
        self.goal_vector = self.track.get_goal_line(self.level)
        self.goal_vector_next = self.track.get_goal_line((self.level + 1) % self.track.n_goals)
        self.goal_vector_last = self.track.get_goal_line((self.level - 1) % self.track.n_goals)

    def update_echo_vectors(self):
        # Generate echo vectors for collision detection with N_SENSORS
        n = self.N_SENSORS if self.N_SENSORS % 2 else self.N_SENSORS - 1
        self.echo_vectors = np.zeros((n, 4))
        for i, ang in enumerate(np.linspace(-60, 60, n)):
            rotated_vector = pygame.math.Vector2(0, -1500).rotate(self.angle + ang)
            self.echo_vectors[i] = [*self.position, *(self.position + rotated_vector)]

    def rotate(self, steering):
        # Rotate the car based on the steering input and velocity
        if self.net_velocity.y != 0.0:
            rotation_factor = min(self.MAX_VELOCITY / 4 * self.net_velocity.y,
                                  self.MAX_VELOCITY / np.sqrt(self.net_velocity.y + self.MAX_VELOCITY / 2))
            self.angle += steering * rotation_factor
            self.angle = (self.angle + 180) % 360 - 180  # Normalize to [-180, 180]

    def accelerate(self, accelerate):
        # Update net velocity based on acceleration, clamped by MAX_VELOCITY
        self.net_velocity += (0.0, accelerate)
        self.net_velocity.y = max(0, min(self.net_velocity.y, self.MAX_VELOCITY))
        self.velocity = self.net_velocity.rotate(self.angle)

    def move(self, action):
        # Execute rotation and acceleration, then update car position
        self.rotate(action[0])
        self.accelerate(action[1])
        
        dx, dy = self.velocity.x, self.velocity.y
        self.movement_vector = [*self.position, self.position.x + dx, self.position.y + dy]
        
        if self.game.camera_mode == 'centered':
            self.track.move_env(dx, dy)
            self.movement_vector = [MAIN_WINDOW_SIZE[0] / 2, MAIN_WINDOW_SIZE[1] / 2,
                                    MAIN_WINDOW_SIZE[0] / 2 + dx, MAIN_WINDOW_SIZE[1] / 2 + dy]
        elif self.game.camera_mode == 'fixed':
            self.position.x -= dx
            self.position.y -= dy

        if self.game.rule_keep_on_screen:
            self.position.x = max(0, min(self.position.x, MAIN_WINDOW_SIZE[0]))
            self.position.y = max(0, min(self.position.y, MAIN_WINDOW_SIZE[1]))

    def check_collision_goal(self):
        # Check if car crosses goal line (forward or backward)
        result_last = utils.line_intersect(*self.movement_vector, *self.goal_vector_last)
        result_next = utils.line_intersect(*self.movement_vector, *self.goal_vector_next)
        if result_last is not None:
            self.level = (self.level - 1) % self.track.n_goals
            self.update_goal_vectors()
        elif result_next is not None:
            self.level = (self.level + 1) % self.track.n_goals
            self.update_goal_vectors()

    def check_collision_track(self):
        # Check collision with track boundaries
        for line in self.track.get_track_segments():
            if utils.line_intersect(*self.movement_vector, *line):
                self.game.set_done()
                break

    def check_collision_echo(self):
        # Set max sensor distance for obstacle detection
        max_distance = 5000
        points = np.full((self.N_SENSORS, 2), self.position.x)
        points[:, 1] = self.position.y
        distances = np.full((self.N_SENSORS), max_distance)

        for i in range(self.N_SENSORS):
            line1 = self.echo_vectors[i]
            closest_point, min_distance = None, max_distance
            for line2 in self.track.get_track_segments():
                result = utils.line_intersect(*line1, *line2)
                if result:
                    distance = np.linalg.norm(np.array(result) - self.position)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = result
            if closest_point is not None:
                points[i] = closest_point
                distances[i] = min_distance

        self.echo_collision_points = points
        self.echo_collision_distances_interp = np.interp(distances, [0, 1000], [-1, 1])

    def update_reward_level(self):
        # Reward based on reaching checkpoints
        previous_reward = self.reward_total
        if self.level == 0 and self.level_previous == self.track.n_goals - 1:
            self.n_lap += 1
        elif self.level == self.track.n_goals - 1 and self.level_previous == 0:
            self.n_lap -= 1
        self.reward_total = self.n_lap * self.track.n_goals + self.level
        self.reward_step = self.reward_total - previous_reward

    def update_reward_laptime(self):
        # Reward based on time to reach checkpoints
        if self.level == 0 and self.level_previous == self.track.n_goals - 1:
            self.n_lap += 1
        elif self.level == self.track.n_goals - 1 and self.level_previous == 0:
            self.n_lap -= 1
        if (self.level - self.level_previous == 1) or (self.level == 0 and self.level_previous == self.track.n_goals - 1):
            self.reward_step = max(1, 100 - self.framecount_goal) / 100
            self.reward_total += self.reward_step
            self.framecount_goal = 0
        elif (self.level - self.level_previous == -1) or (self.level == self.track.n_goals - 1 and self.level_previous == 0):
            self.reward_step = -1
            self.reward_total += self.reward_step
            self.framecount_goal = 0
