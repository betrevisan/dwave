from random import randrange, randint, seed
import numpy as np

# Class for the Prey in the predator-prey model
class Prey:

    def __init__(self, w, h):
        seed(3)
        self.loc = [randint(0, int(w/3)), randint(0, h)]
        # Flag that is set to False if the character was reached by some other character (and is no longer alive)
        self.alive = True
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

    # Avoids the agent at the given perceived location
    def avoid(self, pred_perceived, pred_real, speed):
        buffer = 10 # If the distance between prey and predator is less than 10 it counts as a contact
        prey_v = np.array(self.loc) # Vector for the prey's location
        pred_real_v = np.array(pred_real) # Vector for the predator's real location
        pred_perceived_v = np.array(pred_perceived) # Vector for the predator's perceived location
        real_dist = np.linalg.norm(pred_real_v - prey_v)
        move_v = pred_perceived_v - prey_v
        perceived_dist = np.linalg.norm(move_v)

        # If the prey has been caught, set alive to False
        if real_dist < buffer:
            self.alive = False

        # Move prey alongside (in opposite direction) this vector at a given speed
        d = speed / perceived_dist
        if d > 1:
            d = 1
        new_loc = np.floor((prey_v - d * move_v))
        
        # Update location
        self.loc = new_loc
        
        # If the new location is out of range, bounce back
        self.bounce_back()

        # Update trace
        self.trace.append(list(self.loc))
        return
    
    # If the location is out of range, bounces it back into the range
    def bounce_back(self):
        # Fix x-coordinate, if needed
        if self.loc[0] < 0:
            self.loc[0] = 1
        elif self.loc[0] > self.w:
            self.loc[0] = self.w - 1
        
        # Fix y-coordinate, if needed
        if self.loc[1] < 0:
            self.loc[1] = 1
        elif self.loc[1] > self.h:
            self.loc[1] = self.h - 1
        
        return
    
    # Displays information about the character
    def __repr__(self):
        display = ['\n======<PREY>======']
        display.append('Is alive? ' + str(self.alive))
        display.append('Number of steps taken: ' + str(len(self.trace)))
        display.append('Location trace:')
        trace_str = ""
        for loc in self.trace:
            trace_str += ", " + str(loc)
        display.append(trace_str)
        display.append('===============================\n')
        return "\n".join(display)