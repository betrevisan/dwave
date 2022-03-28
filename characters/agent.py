from random import randrange, randint
import numpy as np

# Class for the Agent in the predator-prey model
class Agent:

    def __init__(self, w, h):
        self.loc = [randint(int(w/3), int(2*w/3)), randint(0, h)]
        # Flag that is set to True if the character reached its prey
        self.feasted = False
        # Flag that is set to False if the character was reached by some other character (and is no longer alive)
        self.alive = True
        # Keeps track of the location trace
        self.trace = [list(self.loc)]
        # Keep track of attention levels
        self.attention_trace = []
        # Keeps track of distances
        self.dist_trace = []
        # Width and Height of the environment
        self.w = w
        self.h = h
        # Error metrics
        self.movement_error = []
        return
    
    # Get this agent's location given the attention level
    def perceive(self, attention):
        blur = 100 - attention
        x = self.loc[0] + blur
        y = self.loc[1] + blur
        return (x, y)

    # Move the agent given perceived locations
    def move(self, agent_loc, prey_loc, predator_loc, speed, bias):
        # If the agent has been caught, set alive to False
        if predator_loc[0] == self.loc[0] and predator_loc[1] == self.loc[1]:
            self.alive = False

        # Vector between the agent and prey
        v_prey = np.linalg.norm(np.array(agent_loc)-np.array(prey_loc))
        # Vector between the agent and predator
        v_predator = np.linalg.norm(np.array(agent_loc)-np.array(predator_loc))

        # Superposition of the two vector given a bias on the prey
        v_superposition = ((1 + bias) * v_prey + v_predator) / 2

        # Move agent alongside the superposition vector at a given speed
        d = speed / (np.sqrt(np.sum(np.square(v_superposition))))
        new_loc = np.array(agent_loc) - d * v_superposition

        # If the agent has reached its prey, set feasted to True
        if prey_loc[0] == self.loc[0] and prey_loc[1] == self.loc[1]:
            self.feasted = True

        # Update location
        self.loc = new_loc
        
        # If the new location is out of range, bounce back
        self.bounce_back()

        # Update trace
        self.trace.append(list(self.loc))
        return

    # For testing purposes
    def move_with_full_attention(self, agent_loc, prey_loc, predator_loc, speed, bias):
        # Vector between the agent and prey
        v_prey = np.linalg.norm(np.array(agent_loc)-np.array(prey_loc))
        # Vector between the agent and predator
        v_predator = np.linalg.norm(np.array(agent_loc)-np.array(predator_loc))

        # Superposition of the two vector given a bias on the prey
        v_superposition = ((1 + bias) * v_prey + v_predator) / 2

        # Move agent alongside the superposition vector at a given speed
        d = speed / (np.sqrt(np.sum(np.square(v_superposition))))
        new_loc = np.array(agent_loc) - d * v_superposition

        self.movement_error.append(new_loc - np.array(self.loc))
        
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

    # Add attentions to the attention_trace
    def track_attention(self, attention):
        self.attention_trace.append(attention)
        return

    # Add distance to the dist_trace
    def track_dist(self, dist):
        self.dist_trace.append(dist)
        return

    # Displays information about the character
    def __repr__(self):
        display = ['\n======<AGENT>======']
        display.append('Is alive? ' + str(self.alive))
        display.append('Did it reach the target? ' + str(self.feasted))
        display.append('Number of steps taken: ' + str(len(self.trace)))
        display.append('Location trace:')
        for loc in self.trace:
            display.append(str(loc))
        display.append('Attention trace (agent, prey, predator):')
        for attn in self.attention_trace:
            display.append(str(attn))
        display.append('Distances trace (dist to prey, dist to predator):')
        for dist in self.dist_trace:
            display.append(str(dist))
        ex_sum = 0
        ey_sum = 0
        for e in self.movement_error:
            ex_sum += e[0]
            ey_sum += e[1]
        display.append('Movement Error: ' + str(ex_sum/len(self.movement_error)) + ', ' + str(ey_sum/len(self.movement_error)))
        display.append('===============================\n')
        return "\n".join(display)
