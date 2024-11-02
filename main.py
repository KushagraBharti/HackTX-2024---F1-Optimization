import pygame
import numpy as np
from car import Car
from track import Track
from race import Race

# Screen constants
TRACK_SIZE = (2463, 1244)
MAIN_WINDOW_SIZE = (int(TRACK_SIZE[0] / 1.3), int(TRACK_SIZE[1] / 1.3))

# Environment configuration
env_config = {
    'camera_mode': 'fixed',  # Options: 'fixed', 'centered'
    'rule_keep_on_screen': True,
    'max_steps': 1000
}

# Initialize Pygame and screen
pygame.init()
screen = pygame.display.set_mode(MAIN_WINDOW_SIZE)
pygame.display.set_caption("F1 Track Environment")

# Load track from CSV and initialize car and environment
csv_file = "Austin.csv"  # Change to the track file you're testing
track = Track(None, csv_file)
car = Car(None, track)
env = Race(env_config)  # If Race is a Gym environment

# Game loop
clock = pygame.time.Clock()
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Simulate a basic action - here it's a stationary action; customize for movement
    action = np.array([0, 0])  # [steering, acceleration] - set values manually if testing movement
    
    # Step the environment
    obs, reward, done, _ = env.step(action)

    # Clear the screen and render the environment
    screen.fill((0, 0, 0))  # Fill background with black
    track.render(screen)     # Render the track
    # car.render(screen)     # If you have a render function in Car, you can also add this
    
    pygame.display.flip()  # Update display
    
    # Control FPS
    clock.tick(60)
    
    # Reset environment on episode end
    if done:
        env.reset()

pygame.quit()
