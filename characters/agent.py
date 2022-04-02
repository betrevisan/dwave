from random import randrange, randint, seed
import numpy as np
import math

# Class for the Agent in the predator-prey model
class Agent:

    def __init__(self, w, h):
        seed(1)
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
        # Perceived location of Agent
        self.perceived_agent_trace = []
        # Perceived location of Prey
        self.perceived_prey_trace = []
        # Perceived location of Predator
        self.perceived_predator_trace = []
        return
    
    # Get the target's location given the attention level
    def perceive(self, target, attention):
        blur = 100 - attention
        x = target.loc[0] + blur
        y = target.loc[1] + blur
        return (x, y)

    # Move the agent given perceived locations
    def move(self, agent_perceived, prey_perceived, predator_perceived,
                prey_real, predator_real, speed, bias):
        buffer = 10 # If the distance between prey and predator is less than 10 it counts as a contact
        agent_real_v = np.array(self.loc) # Vector for the agent's real location
        agent_perceived_v = np.array(agent_perceived) # Vector for the agent's perceived location
        prey_real_v = np.array(prey_real) # Vector for the prey's real location
        prey_perceived_v = np.array(prey_perceived) # Vector for the prey's perceived location
        pred_real_v = np.array(predator_real) # Vector for the predator's real location
        pred_perceived_v = np.array(predator_perceived) # Vector for the predator's perceived location

        self.perceived_agent_trace.append(list(agent_perceived))
        self.perceived_prey_trace.append(list(prey_perceived))
        self.perceived_predator_trace.append(list(predator_perceived))

        real_dist_to_pred = np.linalg.norm(pred_real_v - agent_real_v)
        # If the agent has been caught, set alive to False
        if real_dist_to_pred < buffer:
            self.alive = False

        # Reflect the predator's location along the agent's coordinates
        new_pred_perceived_v = agent_perceived_v - (pred_perceived_v - agent_perceived_v)

        # Get point in between predator's and prey's (superposition of pursuit and avoidance)
        super_v = new_pred_perceived_v + bias * (prey_perceived_v - new_pred_perceived_v)
        
        move_v =  super_v - agent_perceived_v
        # Move agent alongside the superposition vector at a given speed
        d = speed / np.linalg.norm(move_v)
        if d > 1:
            d = 1
        new_loc = np.floor((agent_real_v + d * move_v))

        # Update location
        self.loc = new_loc

        real_dist_to_prey = np.linalg.norm(prey_real_v - np.array(self.loc))
        # If the agent has reached its prey, set feasted to True
        if real_dist_to_prey < buffer:
            self.feasted = True
        
        # If the new location is out of range, bounce back
        self.bounce_back()

        # Update trace
        self.trace.append(list(self.loc))

        # Keep track of distances
        self.track_dist([math.dist(prey.loc, agent.loc), math.dist(predator.loc, agent.loc)])
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
        trace_str = ""
        for loc in self.trace:
            trace_str += ", " + str(loc)
        display.append(trace_str)
        display.append('Agent perceived location trace:')
        agent_str = ""
        for loc in self.perceived_agent_trace:
            agent_str += ", " + str(loc)
        display.append(agent_str)
        display.append('Prey perceived location trace:')
        prey_str = ""
        for loc in self.perceived_prey_trace:
            prey_str += ", " + str(loc)
        display.append(prey_str)
        display.append('Predator perceived location trace:')
        predator_str = ""
        for loc in self.perceived_predator_trace:
            predator_str += ", " + str(loc)
        display.append(predator_str)
        display.append('Attention trace (agent, prey, predator):')
        for attn in self.attention_trace:
            display.append(str(attn))
        display.append('Distances trace (dist to prey, dist to predator):')
        for dist in self.dist_trace:
            display.append(str(dist))
        display.append('===============================\n')
        return "\n".join(display)
