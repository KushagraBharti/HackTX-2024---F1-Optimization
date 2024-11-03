import numpy as np
import utils
import pygame


TRACK_SIZE = (2463, 1244)
MAIN_WINDOW_SIZE = tuple(int(x / 1.3) for x in TRACK_SIZE)
START_POS = (1501, 870)


class Car:
    MAX_ACCELERATION = 1
    MAX_VELOCITY = 8   # [m/frame]
    ROTATION_VELOCITY = 4   # [deg/frame]
    N_SENSORS = 7

    def __init__(self, game, track):
        self.game = game
        self.track = track
        self.visible = True
        self.reset_game_state()

    def reset_game_state(self, x=START_POS[0], y=START_POS[1], ang=-92, vel_x=0, vel_y=0, level=0):
        self.position = pygame.math.Vector2(x, y)
        self.net_velocity = pygame.math.Vector2(vel_x, vel_y)
        self.velocity = pygame.math.Vector2(vel_x, vel_y)
        self.angle = ang
        self.action_state = 1   # 1 if alive, 0 if dead

        self.update_state(np.array([self.position, self.angle, self.net_velocity.y, self.action_state], dtype=object))

        self.level = level
        self.level_previous = level
        # framecount_goal: since last goal
        self.framecount_goal = 0
        # framecount_total: since reset
        self.framecount_total = 0
        # reward: since last frame
        self.n_lap = 0
        self.reward_step = 0
        self.reward_total = 0
        self.done = False
        self.action = np.array([0, 0])
        self.update_echo_vectors()
        self.update_goal_vectors()
        self.check_collision_echo()

    def update_state(self, car_state):
        self.position, self.angle, self.net_velocity.y, self.action_state = car_state

    def update_reward_level(self):  # static
        reward_total_previous = self.reward_total
        if self.level == 0 and self.level_previous == self.track.n_goals-1:
            self.n_lap += 1
            print(f"Completed lap {self.n_lap}.")
        if self.level == self.track.n_goals-1 and self.level_previous == 0:
            self.n_lap -= 1
        self.reward_total = self.n_lap * self.track.n_goals + self.level
        self.reward_step = self.reward_total - reward_total_previous
        print(f"Reward updated to {self.reward_total} at level {self.level}.")

    def update_reward_laptime(self):
        if self.level == 0 and self.level_previous == self.track.n_goals-1:
            self.n_lap += 1
            print(f"Completed lap {self.n_lap}.")
        if self.level == self.track.n_goals-1 and self.level_previous == 0:
            self.n_lap -= 1
        if (self.level - self.level_previous == 1) or (self.level == 0 and self.level_previous == self.track.n_goals-1):
            self.reward_step = max(1, max(0, 100-self.framecount_goal)) / 100
            self.reward_total += self.reward_step
            print(f"Progress reward gained: {self.reward_step}. Total reward: {self.reward_total}")
            self.framecount_goal = 0
        if (self.level - self.level_previous == -1) or (self.level == self.track.n_goals-1 and self.level_previous == 0):
            self.reward_step = - 1
            self.reward_total += self.reward_step
            print(f"Penalty applied: {self.reward_step}. Total reward: {self.reward_total}")
            self.framecount_goal = 0

    def update_goal_vectors(self):
        self.goal_vector = self.track.get_goal_line(self.level)
        self.goal_vector_next = self.track.get_goal_line(self.level)
        self.goal_vector_last = self.track.get_goal_line(self.level-1)

    def update_echo_vectors(self):
        n = self.N_SENSORS
        if n % 2 == 0: n = max(n-1, 3)  # make sure that n>=3 and odd
        matrix = np.zeros((n, 4))
        matrix[:, 0], matrix[:, 1] = self.position
        for idx, ang in enumerate(np.linspace(-60, 60, n)):
            matrix[idx, 2], matrix[idx, 3] = self.position + pygame.math.Vector2(0.0, -1500.0).rotate(self.angle+ang)
        self.echo_vectors = matrix

    def rotate(self, steering):  # input: action0
        if self.net_velocity.y != 0.0:   
            self.angle += steering * min(self.MAX_VELOCITY/4*self.net_velocity.y, (self.MAX_VELOCITY/np.sqrt(self.net_velocity.y+self.MAX_VELOCITY/2)))
        if self.angle > 180:
            self.angle -= 2 * 180
        if self.angle < -180:
            self.angle += 2 * 180

    def accelerate(self, accelerate):  # input: action1
        self.net_velocity += (0.0, accelerate)
        # self.net_velocity.y = max(-self.MAX_VELOCITY, min(self.net_velocity.y, self.MAX_VELOCITY))
        self.net_velocity.y = max(0, min(self.net_velocity.y, self.MAX_VELOCITY))
        self.velocity = self.net_velocity.rotate(self.angle)

    def update_observations(self):
        # ─── OBSERVATION 8: VELOCITY ─────────────────────────────────────
        vel = self.net_velocity.y
        self.vel_interp = np.interp(vel, [0, self.MAX_VELOCITY], [-1, 1])

    def move(self, action):
        # first apply rotation!
        self.rotate(action[0])
        self.accelerate(action[1])

        # ─── MOVE ───────────────────────────────────────────────────
        d_x, d_y = self.velocity.x, self.velocity.y
        x_from, y_from = self.position.x, self.position.y

        # ─── CENTERED MODE ───────────────────────────────────────────────
        if self.game.camera_mode == 'centered':
            self.track.move_env(d_x,d_y)
            self.movement_vector = [MAIN_WINDOW_SIZE[0]/2, MAIN_WINDOW_SIZE[1]/2, MAIN_WINDOW_SIZE[0]/2 + d_x, [1]/2 + d_y]
        # ─── FIXED MODE ─────────────────────────────────────────────────
        if self.game.camera_mode == 'fixed':
            self.position.x -= d_x
            self.position.y -= d_y
            self.movement_vector = [x_from, y_from, self.position.x, self.position.y]

        # ─── KEEP ON SCREEN ──────────────────────────────────────────────
        # rocket cannot leave fixed screen area
        if self.game.rule_keep_on_screen:
            if self.position.x > MAIN_WINDOW_SIZE[0]:
                self.position.x = MAIN_WINDOW_SIZE[0]
            elif self.position.x < 0:
                self.position.x = 0            
            if self.position.y > MAIN_WINDOW_SIZE[1]:
                self.position.y = MAIN_WINDOW_SIZE[1]
            elif self.position.y < 0:
                self.position.y = 0

    def check_collision_goal(self):
        result_last = utils.line_intersect(*self.movement_vector, *self.goal_vector_last)
        result_next = utils.line_intersect(*self.movement_vector, *self.goal_vector_next)
        if result_last is not None:
            self.level -= 1
            if self.level == -1:
                self.level = self.track.n_goals - 1
            self.update_goal_vectors()
        elif result_next is not None:
            self.level += 1
            if self.level == self.track.n_goals:
                self.level = 0
            self.update_goal_vectors()

    def check_collision_track(self):
        for line in self.track.level_collision_vectors:
            result = utils.line_intersect(*self.movement_vector, *line)
            if result is not None:
                self.game.set_done()
                break

    def check_collision_echo(self):
        # max_distance: Distance value maps to observation=1 if distance >= max_distance
        max_distance = 5000
        points = np.full((self.N_SENSORS, 2), self.position.x) # points for visualiziation
        points[:,1] = self.position.y
        distances = np.full((self.N_SENSORS), max_distance) # distances for observation
        n = self.track.level_collision_vectors.shape[0]
        for i in range(self.N_SENSORS):
            found = False
            line1 = self.echo_vectors[i, :]
            points_candidates = np.zeros((n,2))
            distances_candidates = np.full((n), max_distance)
            for j, line2 in enumerate(self.track.level_collision_vectors):
                result = utils.line_intersect(*line1, *line2)
                if result is not None:
                    found = True
                    points_candidates[j,:] = result
                    distances_candidates[j] = np.sqrt((self.position.x-result[0])**2+(self.position.y-result[1])**2)
            if found: # make sure one intersection is found
                argmin = np.argmin(distances_candidates)  # index of closest intersection 
                points[i, :] = points_candidates[argmin]
                distances[i] = distances_candidates[argmin]

        self.echo_collision_points = points
        # ─── NORMALIZE DISTANCES ─────────────────────────────────────────
        # linear mapping from 0,1000 to -1,1
        # distance 0 becomes -1, distance 1000 becomes +1
        # values always in range [-1,1]
        self.echo_collision_distances_interp = np.interp(distances, [0, 1000], [-1, 1])

    def heuristic_agent(self):
        """
        Heuristic agent that leverages existing echo vector distances to navigate the track.
        """
        # Ensure the echo vectors and collision points are up-to-date
        self.update_echo_vectors()
        self.check_collision_echo()  # This updates echo distances

        # Parameters for movement
        steering = 0
        acceleration = 0.5  # Moderate acceleration

        # Sensor readings
        front_distance = self.echo_collision_distances_interp[self.N_SENSORS // 2]  # center sensor
        left_distance = self.echo_collision_distances_interp[0]
        right_distance = self.echo_collision_distances_interp[-1]

        # Thresholds for obstacle distance
        safe_distance = 0.09  # Consider distances greater than this as "safe"
        close_distance = 0.05  # Trigger avoidance if below this

        # Heuristic navigation logic
        if front_distance < close_distance:
            # Very close obstacle, decide to turn left or right based on which side is more open
            print("Close obstacle detected. Turning to avoid collision.")
            acceleration = -0.9  # Brake to avoid collision
            steering = 0.7 if left_distance > right_distance else -0.7
            acceleration = 0.2  # Accelerate after avoiding collision

        elif front_distance < safe_distance:
            # Moderate obstacle distance; adjust to avoid heading straight into it
            print("Adjusting to avoid moderate obstacle.")
            if left_distance < right_distance:
                steering = 0.5  # Steer left
            else:
                steering = -0.5  # Steer right
            acceleration = 0.4  # Slightly reduce speed

        else:
            # No immediate obstacle, steer towards the longest open path
            print("Clear path detected. Accelerating.")
            if left_distance < safe_distance and right_distance >= safe_distance:
                steering = -0.2  # Small right turn to keep distance from left boundary
            elif right_distance < safe_distance and left_distance >= safe_distance:
                steering = 0.2  # Small left turn to keep distance from right boundary
            else:
                steering = 0  # Keep straight if balanced
            acceleration = 0.8  # Increase speed when the path is clear

        # Move with the chosen steering and acceleration
        self.move([steering, acceleration])

    def echo_heuristic_agent(self):
        """
        Heuristic agent that steers towards the longest echo vector at a very slow speed
        and with a high turning response.
        """
        # Update echo vectors and calculate distances to boundaries
        self.update_echo_vectors()
        self.check_collision_echo()  # This updates self.echo_collision_distances_interp with sensor distances

        # Find the index of the longest echo vector (the sensor with the maximum distance)
        longest_index = np.argmax(self.echo_collision_distances_interp)
        longest_distance = self.echo_collision_distances_interp[longest_index]
        
        # Steering logic to follow the longest vector
        # Calculate the angle to the longest echo vector based on its index
        # Assume sensors are spaced evenly across a 120-degree field of view
        angle_offset = np.linspace(-60, 60, self.N_SENSORS)[longest_index]
        
        # Adjust steering to point towards the longest vector direction
        # Increase steering angle sensitivity by a factor to make sharper turns
        steering = (angle_offset / 60.0) * 9.0  # Multiplied by 2 for sharper turns

        # Use very slow acceleration to maintain a controlled, slow pace
        # Keep acceleration low regardless of the distance to maintain very slow speed
        acceleration = 0.05  # Extremely low speed for precision

        print(f"Following longest vector at index {longest_index} with distance {longest_distance:.2f}. Steering: {steering:.2f}, Acceleration: {acceleration:.2f}")

        # Check for collisions after the move
        self.check_collision_track()  # This sets `self.done` if a collision occurs

        # Calculate reward
        if self.game.reward_mode == 'level':
            self.update_reward_level()  # Updates reward based on checkpoints passed
        elif self.game.reward_mode == 'laptime':
            self.update_reward_laptime()
            
        # Execute the move command with calculated steering and acceleration
        self.move([steering, acceleration])


