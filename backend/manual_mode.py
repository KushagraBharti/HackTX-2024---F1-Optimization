import numpy as np
import pygame
from race import Race

run = True
heuristic_mode = False

def event_to_action(eventlist):
    global run, heuristic_mode
    for event in eventlist:
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:  # Only check for keydown events
            if event.key == pygame.K_r:
                env.reset()
            elif event.key == pygame.K_h:  # Activate heuristic mode on 'H' key press
                heuristic_mode = True
        elif event.type == pygame.KEYUP:  # Only check for keyup events
            if event.key == pygame.K_h:  # Deactivate heuristic mode on 'H' key release
                heuristic_mode = False


def pressed_to_action(pressed):
    if pressed[pygame.K_UP]:
        action_acc = 1
    elif pressed[pygame.K_DOWN]:
        action_acc = -1
    else:
        action_acc = 0
    
    if pressed[pygame.K_1]:
        action_acc = 0.01

    if pressed[pygame.K_RIGHT]:
        action_steering = 1
    elif pressed[pygame.K_LEFT]:
        action_steering = -1
    else:
        action_steering = 0

    return np.array([action_steering, action_acc])


if __name__ == "__main__":
        
    env_config = {
        'gui': True,
        # 'export_frames': True,
        'export_states': True,
        'export_string': 'human',
        'gui_draw_background': False,
        # 'rule_collision': False,
        # 'max_steps': 2000,
    }

    env = Race(env_config)
    env.render()
    run = True
    while run:
        env.clock.tick(120)
        get_event = pygame.event.get()
        event_to_action(get_event)

        # Use heuristic agent if activated
        if heuristic_mode:
            #env.car.heuristic_agent()
            env.car.echo_heuristic_agent()
        else:
            action = pressed_to_action(pygame.key.get_pressed())
            env.step(action=action)

        env.render()
    pygame.quit()