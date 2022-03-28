from random import randrange, randint
import numpy as np

# Class for the Prey in the predator-prey model
class Prey:

    def __init__(self, w, h):
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
    def avoid(self, perceived_loc, speed):
        # If the prey has been caught, set alive to False
        if perceived_loc[0] == self.loc[0] and perceived_loc[1] == self.loc[1]:
            self.alive = False

        # Vector between the two points
        v = np.linalg.norm(np.array(self.loc)-np.array(perceived_loc))

        # Move prey alongside this vector at a given speed
        d = speed / (np.sqrt(np.sum(np.square(v))))
        new_loc = np.array(self.loc) - d * v

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
            self.loc[0] = self.loc[0] + abs(self.loc[0]) + 10
        elif self.loc[0] > self.w:
            self.loc[0] = self.loc[0] - (self.loc[0] - self.w) - 10
        
        # Fix y-coordinate, if needed
        if self.loc[1] < 0:
            self.loc[1] = self.loc[1] + abs(self.loc[1]) + 10
        elif self.loc[1] > self.h:
            self.loc[1] = self.loc[1] - (self.loc[1] - self.h) - 10
        
        return
        # hi
    
    # Displays information about the character
    def __repr__(self):
        display = ['\n======<PREY>======']
        display.append('Is alive? ' + str(self.alive))
        display.append('Number of steps taken: ' + str(len(self.trace)))
        display.append('Location trace:')
        for loc in self.trace:
            display.append(str(loc))
        display.append('===============================\n')
        return "\n".join(display)