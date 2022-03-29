from random import randrange, randint
import numpy as np

# Class for the Predator in the predator-prey model
class Predator:

    def __init__(self, w, h):
        self.loc = [randint(int(2*w/3), w), randint(0, h)]
        # Flag that is set to True if the character reached its prey
        self.feasted = False
        # Keeps track of the location trace
        self.trace = [list(self.loc)]
        # Width and Height of the environment
        self.w = w
        self.h = h
        return

    # Get this character's location given the attention level
    def perceive(self, attention):
        blur = 100 - attention
        x = self.loc[0] + blur
        y = self.loc[1] + blur
        return (x, y)

    # Pursues the agent at the given perceived location
    def pursue(self, prey_perceived, prey_real, speed):
        buffer = 10 # If the distance between prey and predator is less than 10 it counts as a contact
        pred_v = np.array(self.loc) # Vector for the predator's location
        prey_real_v = np.array(prey_real) # Vector for the prey's real location
        prey_perceived_v = np.array(prey_perceived) # Vector for the prey's perceived location
        move_v = prey_perceived_v - pred_v
        perceived_dist = np.linalg.norm(move_v)

        # Move predator alongside this vector at a given speed
        d = speed / perceived_dist
        if d > 1:
            d = 1
        new_loc = np.floor((pred_v + d * move_v))

        # Update location
        self.loc = new_loc

        real_dist = np.linalg.norm(prey_real_v - np.array(self.loc))
        # If the prey has been caught, set feasted to True
        if real_dist < buffer:
            self.feasted = True

        # Update trace
        self.trace.append(list(self.loc))
        return
    
    # Displays information about the character
    def __repr__(self):
        display = ['\n======<PREDATOR>======']
        display.append('Did it reach the target? ' + str(self.feasted))
        display.append('Number of steps taken: ' + str(len(self.trace)))
        display.append('Location trace:')
        trace_str = ""
        for loc in self.trace:
            trace_str += ", " + str(loc)
        display.append(trace_str)
        display.append('===============================\n')
        return "\n".join(display)