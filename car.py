import numpy as np
import pygame
import utils


# track and window sizes, starting position
TRACK_SIZE = (2463, 1244)
MAIN_WINDOW_SIZE = tuple(int(x / 1.3) for x in TRACK_SIZE)
START_POS = (1501, 870)



class Car:
    # set max acceleration, velocity, and rotation speed
    MAX_ACCELERATION = 1
    MAX_VELOCITY = 8  # max speed [m/frame]
    ROTATION_VELOCITY = 4  # rotation speed [deg/frame]
    N_SENSORS = 7  # sensor count for echo vectors



    def __init__(self, game, track):
        # init game and track references
        self.game = game
        self.track = track
        self.visible = True  # flag for rendering car visibility
        self.reset_game_state()  # set initial car state

    def reset_game_state(self, x=START_POS[0], y=START_POS[1], ang=-92, vel_x=0, vel_y=0, level=0):
        # reset position, velocity, angle
        self.position = pygame.math.Vector2(x, y)
        self.net_velocity = pygame.math.Vector2(vel_x, vel_y)
        self.velocity = pygame.math.Vector2(0, 0)

        self.angle = ang
        self.action_state = 1  # car is "alive" at reset

        # reset game tracking vars
        self.update_state(np.array([self.position, self.angle, self.net_velocity.y, self.action_state], dtype=object))
        self.level = level  # start level
        self.level_previous = level  # track previous level
        self.framecount_goal = 0  # frames since last checkpoint
        self.framecount_total = 0  # total frames since reset
        self.n_lap = 0  # track lap count
        self.reward_step = 0  # reward since last action
        self.reward_total = 0  # total reward
        self.done = False  # if game is done
        self.action = np.array([0, 0])  # default action (no rotation, no accel)

        # initialize goal and sensor vectors
        self.update_echo_vectors()
        self.update_goal_vectors()
        self.check_collision_echo()


#weeeeeeewoooooooooooweeeeeeeeeeewooooooooweeeeeeeewooooooo i wanna kms :) weeeee


    def update_state(self, car_state):
        # update car state (position, angle, velocity, action state)
        self.position, self.angle, self.net_velocity.y, self.action_state = car_state





    def update_reward_level(self):
        # update total reward for checkpoint progress
        reward_total_previous = self.reward_total
        if self.level == 0 and self.level_previous == self.track.n_goals - 1:
            self.n_lap += 1  # lap completed
        if self.level == self.track.n_goals - 1 and self.level_previous == 0:
            self.n_lap -= 1  # moving back a lap
        self.reward_total = self.n_lap * self.track.n_goals + self.level
        self.reward_step = self.reward_total - reward_total_previous  # change in reward since last check




    def update_reward_laptime(self):
        # add reward based on time taken since last checkpoint
        if self.level == 0 and self.level_previous == self.track.n_goals - 1:
            self.n_lap += 1
        if self.level == self.track.n_goals - 1 and self.level_previous == 0:
            self.n_lap -= 1
        if (self.level - self.level_previous == 1) or (self.level == 0 and self.level_previous == self.track.n_goals - 1):
            self.reward_step = max(1, max(0, 100 - self.framecount_goal)) / 100
            self.reward_total += self.reward_step  # add time-based reward
            self.framecount_goal = 0  # reset frame counter on checkpoint
        if (self.level - self.level_previous == -1) or (self.level == self.track.n_goals - 1 and self.level_previous == 0):
            self.reward_step = -1  # penalty for going backward
            self.reward_total += self.reward_step
            self.framecount_goal = 0

   
   
   
    def update_goal_vectors(self):
        # update vectors to current, next, and last goal
        self.goal_vector = self.track.get_goal_line(self.level)
        self.goal_vector_next = self.track.get_goal_line(self.level + 1)
        self.goal_vector_last = self.track.get_goal_line(self.level - 1)

    
    
    def update_echo_vectors(self):
        # create echo vectors based on car position and angle
        n = self.N_SENSORS
        if n % 2 == 0: 
            n = max(n - 1, 3)  # ensure odd sensor count, at least 3
        matrix = np.zeros((n, 4))
        matrix[:, 0], matrix[:, 1] = self.position  # set position for each sensor
        for idx, ang in enumerate(np.linspace(-60, 60, n)):
            # rotate each vector according to angle for echo distance
            matrix[idx, 2], matrix[idx, 3] = self.position + pygame.math.Vector2(0.0, -1500.0).rotate(self.angle + ang)
        self.echo_vectors = matrix

    
    
    def rotate(self, steering):  # input: action0
        # rotate car based on velocity and steering input
        if self.net_velocity.y != 0.0:
            self.angle += steering * min(
                self.MAX_VELOCITY / 4 * self.net_velocity.y,
                (self.MAX_VELOCITY / np.sqrt(self.net_velocity.y + self.MAX_VELOCITY / 2))
    )

        # keep angle within -180 to 180 range
        if self.angle > 180:
            self.angle -= 360
        elif self.angle < -180:
            self.angle += 360

    
    def accelerate(self, accelerate):  # input: action1
        # update velocity with acceleration input, limit to max speed
        self.net_velocity += (0.0, accelerate)
        self.net_velocity.y = max(0, min(self.net_velocity.y, self.MAX_VELOCITY))
        self.velocity = self.net_velocity.rotate(self.angle)  # apply rotation to net velocity

   
   
    def update_observations(self):
        # normalize velocity for observations (range [-1, 1])
        vel = self.net_velocity.y
        self.vel_interp = np.interp(vel, [0, self.MAX_VELOCITY], [-1, 1])

    
    
    def move(self, action):
        # apply rotation and acceleration based on action
        self.rotate(action[0])
        self.accelerate(action[1])

        # get x, y movement based on camera mode
        d_x, d_y = self.velocity.x, self.velocity.y
        x_from, y_from = self.position.x, self.position.y

        if self.game.camera_mode == 'centered':
            # in centered mode, move track, not car
            self.track.move_env(d_x, d_y)
            self.movement_vector = [MAIN_WINDOW_SIZE[0] / 2, MAIN_WINDOW_SIZE[1] / 2, 
                                    MAIN_WINDOW_SIZE[0] / 2 + d_x, MAIN_WINDOW_SIZE[1] / 2 + d_y]
        elif self.game.camera_mode == 'fixed':
            # in fixed mode, move car position
            self.position.x -= d_x
            self.position.y -= d_y
            self.movement_vector = [x_from, y_from, self.position.x, self.position.y]

        # if keeping car on screen, limit position within window size
        if self.game.rule_keep_on_screen:
            self.position.x = max(0, min(self.position.x, MAIN_WINDOW_SIZE[0]))
            self.position.y = max(0, min(self.position.y, MAIN_WINDOW_SIZE[1]))

    
    
    def check_collision_goal(self):
        # check if crossing goal line (forward or backward)
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
        # check for collision with track boundaries
        for line in self.track.level_collision_vectors:
            result = utils.line_intersect(*self.movement_vector, *line)
            if result is not None:
                self.game.set_done()  # end game on collision
                break

   
   
    def check_collision_echo(self):
        # set max distance for sensors
        max_distance = 5000
        points = np.full((self.N_SENSORS, 2), self.position.x)  # points for visualization
        points[:, 1] = self.position.y
        distances = np.full((self.N_SENSORS), max_distance)  # distances for observations
        n = self.track.level_collision_vectors.shape[0]
        for i in range(self.N_SENSORS):
            found = False
            line1 = self.echo_vectors[i, :]
            points_candidates = np.zeros((n, 2))
            distances_candidates = np.full((n), max_distance)
            for j, line2 in enumerate(self.track.level_collision_vectors):
                # get intersection point for sensor line with track
                result = utils.line_intersect(*line1, *line2)
                if result is not None:
                    found = True
                    points_candidates[j, :] = result
                    distances_candidates[j] = np.sqrt((self.position.x - result[0]) ** 2 + 
                                                      (self.position.y - result[1]) ** 2)
            if found:
                # select closest intersection point
                argmin = np.argmin(distances_candidates)
                points[i, :] = points_candidates[argmin]
                distances[i] = distances_candidates[argmin]

        # set final collision points and normalize distances
        self.echo_collision_points = points
        self.echo_collision_distances_interp = np.interp(distances, [0, 1000], [-1, 1])
