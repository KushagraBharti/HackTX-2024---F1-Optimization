import numpy as np
import matplotlib.pyplot as plt
import pygame
import csv
from scipy.interpolate import CubicSpline

# Constants for screen dimensions
TRACK_SIZE = (2463, 1244)
MAIN_WINDOW_SIZE = (int(TRACK_SIZE[0] / 1.3), int(TRACK_SIZE[1] / 1.3))

class Track:
    def __init__(self, game, csv_file):
        self.game = game
        self.csv_file = csv_file
        self.load_track_data()
        self.generate_goal_checkpoints()
        self.n_goals = len(self.goals)

    def load_track_data(self):
        # Load centerline and boundary data from CSV
        self.x_m, self.y_m, self.w_tr_right_m, self.w_tr_left_m = [], [], [], []
        
        with open(self.csv_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.x_m.append(float(row[0]))
                self.y_m.append(float(row[1]))
                self.w_tr_right_m.append(float(row[2]))
                self.w_tr_left_m.append(float(row[3]))

        # Smooth the centerline and boundary points using cubic splines
        self.centerline = self.smooth_track(self.x_m, self.y_m)
        self.right_boundary = self.smooth_track(
            [x + r for x, r in zip(self.x_m, self.w_tr_right_m)], 
            self.y_m
        )
        self.left_boundary = self.smooth_track(
            [x - l for x, l in zip(self.x_m, self.w_tr_left_m)], 
            self.y_m
        )

    def smooth_track(self, x, y):
        """Smooth the given track points using cubic splines."""
        # Create spline interpolations for both x and y
        t = np.arange(len(x))
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        
        # Generate intermediate points for smooth curves
        t_fine = np.linspace(0, len(x) - 1, num=500)
        x_smooth = cs_x(t_fine)
        y_smooth = cs_y(t_fine)
        return [(x_smooth[i], y_smooth[i]) for i in range(len(x_smooth))]

    def generate_goal_checkpoints(self):
        """Define goals at regular intervals or based on track curves."""
        goal_interval = max(1, len(self.centerline) // 100)
        self.goals = [self.centerline[i] for i in range(0, len(self.centerline), goal_interval)]
        # Define the starting line as the first goal checkpoint
        self.starting_line = self.goals[0]

    def render(self, window):
        """Render the track and additional markers in pygame."""
        # Draw the right and left boundaries with distinct colors
        pygame.draw.lines(window, (255, 0, 0), False, [(int(x), int(y)) for x, y in self.right_boundary], 2)
        pygame.draw.lines(window, (0, 255, 0), False, [(int(x), int(y)) for x, y in self.left_boundary], 2)
        
        # Draw the centerline
        pygame.draw.lines(window, (0, 0, 255), False, [(int(x), int(y)) for x, y in self.centerline], 1)
        
        # Draw goals and starting line
        for goal in self.goals:
            pygame.draw.circle(window, (255, 255, 0), (int(goal[0]), int(goal[1])), 5)  # Goals in yellow
        pygame.draw.circle(window, (255, 255, 255), (int(self.starting_line[0]), int(self.starting_line[1])), 8)  # Starting line in white

    def is_within_boundaries(self, car_position):
        """Check if the car is within the boundaries."""
        car_x, car_y = car_position
        for i in range(len(self.x_m) - 1):
            # Boundary segments as lines between consecutive boundary points
            left_point1 = (self.left_boundary[i][0], self.left_boundary[i][1])
            left_point2 = (self.left_boundary[i + 1][0], self.left_boundary[i + 1][1])
            right_point1 = (self.right_boundary[i][0], self.right_boundary[i][1])
            right_point2 = (self.right_boundary[i + 1][0], self.right_boundary[i + 1][1])

            if self._point_within_boundary_segment(car_x, car_y, left_point1, left_point2, right_point1, right_point2):
                return True
        return False

    def _point_within_boundary_segment(self, px, py, lp1, lp2, rp1, rp2):
        """Helper function to check if a point is within a boundary segment."""
        # Vector cross products to check boundary inclusion
        left_cross = (lp2[0] - lp1[0]) * (py - lp1[1]) - (lp2[1] - lp1[1]) * (px - lp1[0])
        right_cross = (rp2[0] - rp1[0]) * (py - rp1[1]) - (rp2[1] - rp1[1]) * (px - rp1[0])
        return left_cross >= 0 and right_cross <= 0

    def get_track_segments(self):
        """Get each track segment for potential pathfinding purposes."""
        track_segments = []
        for i in range(len(self.centerline) - 1):
            segment = (self.centerline[i], self.centerline[i + 1])
            track_segments.append(segment)
        return track_segments

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode(MAIN_WINDOW_SIZE)
    clock = pygame.time.Clock()
    track = Track(None, "environment/Austin.csv")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # Clear screen
        track.render(screen)  # Draw track
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
