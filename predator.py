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
    def pursue(self, perceived_loc, speed):
        # Vector between the two points
        v = np.linalg.norm(np.array(self.loc)-np.array(perceived_loc))

        # Move predator alongside this vector at a given speed
        d = speed / (sqrt(v[0]**2 + v[1]**2))
        new_loc = np.array(self.loc) + d * v

        # Update location
        self.loc = new_loc

        # If the character has reached its target, set feasted to True
        if perceived_loc[0] == self.loc[0] and perceived_loc[1] == self.loc[1]:
            self.feasted = True
        
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
    
    # Displays information about the character
    def __repr__(self):
        display = ['\n======<PREDATOR>======']
        display.append('Is alive? ' + str(self.alive))
        display.append('Did it reach the target? ' + str(self.feasted))
        display.append('Number of steps taken: ' + str(len(self.trace)))
        display.append('Location trace:')
        for loc in self.trace:
            display.append(str(loc))
        display.append('===============================\n')
        return "\n".join(display)