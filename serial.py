# This file implements the serial approach to the predator-prey model in quantum computing

from dwave.system import EmbeddingComposite, DWaveSampler
from random import randrange, randint
from numpy import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Number of iterations of the model
ITERATIONS = 10
# Number of reads in the annealer
NUM_READS = 15
# Width and height of the coordinate plane
WIDTH = 500
HEIGHT = 500
# Maximum distance between two points in the plane
MAX_DIST = sqrt(WIDTH**2 + HEIGHT**2)
# For now, speed is always constant
SPEED = 30

# Auxiliar function for calculating the distance between two points
def dist(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Class for the characters in the predator-prey model
class Character:

    def __init__(self, name):
        self.name = name
        # Initialize characters at a random location
        self.loc = [randint(0, WIDTH), randint(0, HEIGHT)]
        # Flag that is set to True if the character reached its target
        self.target_reached = False
        # Flag that is set to False if the character was reached by some other character (and is no longer alive)
        self.alive = True
        # Keeps track of the location trace
        self.trace = [list(self.loc)]
        # Keeps track of attention levels
        self.attention_trace = []
        # Keeps track of distances
        self.dist_trace = []
        # Flag that decribes whether the agent is escaping or pursuing (only used in the agent character)
        self.escaping = False
        return
    
    # Get this agent's location given the attention level
    def perceive(self, attention):
        blur = 100 - attention
        x = self.loc[0] + blur
        y = self.loc[1] + blur
        return (x, y)
    
    # Pursues the target at the given perceived location
    def pursue(self, perceived_loc):
        # If the character has reached its target, set target_reached to True
        if perceived_loc[0] == self.loc[0] and perceived_loc[1] == self.loc[1]:
            self.target_reached = True
        # If target is in the same x cordinate, only change y
        elif perceived_loc[0] == self.loc[0]:
            # Move up towards the target at a given speed
            movey = min([abs(perceived_loc[1] - self.loc[1]), SPEED])

            # If the target is down, move down
            if perceived_loc[1] - self.loc[1] < 0: 
                movey = -movey
            
            # Update y
            self.loc[1] = self.loc[1] + movey
        else:
            slope = (perceived_loc[1] - self.loc[1]) / (perceived_loc[0] - self.loc[1])
            b = self.loc[1] - (slope * self.loc[0])

            # Move right towards the target at a given speed
            movex = min([abs(perceived_loc[0] - self.loc[0]), SPEED])

            # If the target is to the left, move left
            if perceived_loc[0] - self.loc[0] < 0:
                movex = -movex
            
            # Update character's location
            self.loc[0] = self.loc[0] + movex
            self.loc[1] = slope * self.loc[0] + b
        
        # If the new location is out of range, bounce back
        self.bounce_back()

        # Update trace
        self.trace.append(list(self.loc))
        return

    # Avoids the target at the given perceived location
    def avoid(self, perceived_loc):
        # If the character was caught set alive to False
        if perceived_loc[0] == self.loc[0] and perceived_loc[1] == self.loc[1]:
            self.alive = False
        # If they are both in the same x-coordinate, only move y
        elif perceived_loc[0] == self.loc[0]:
            # Move down away from the target at a given speed
            movey = -min([abs(perceived_loc[1] - self.loc[1]), SPEED])

            # If the target is down, move up
            if perceived_loc[1] - self.loc[1] < 0: 
                movey = -movey
            
            # Update y
            self.loc[1] = self.loc[1] + movey
        else:
            slope = (perceived_loc[1] - self.loc[1]) / (perceived_loc[0] - self.loc[1])
            b = self.loc[1] - (slope * self.loc[0])
            
            # Move left away from the target at a given speed
            movex = -min([abs(perceived_loc[0] - self.loc[0]), SPEED])

            # If the target is to the left, move right
            if perceived_loc[0] - self.loc[0] < 0:
                movex = -movex

            # Update character's location
            self.loc[0] = self.loc[0] + movex
            self.loc[1] = slope * self.loc[0] + b
        
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
        elif self.loc[0] > WIDTH:
            self.loc[0] = self.loc[0] - (self.loc[0] - WIDTH) - 10
        
        # Fix y-coordinate, if needed
        if self.loc[1] < 0:
            self.loc[1] = self.loc[1] + abs(self.loc[1]) + 10
        elif self.loc[1] > HEIGHT:
            self.loc[1] = self.loc[1] - (self.loc[1] - HEIGHT) - 10
        
        return

    # Add attention to the attention_trace
    def track_attention(self, attention):
        self.attention_trace.append(attention)
        return

    # Add distance to the dist_trace
    def track_dist(self, dist):
        self.dist_trace.append(dist)
        return

    # Displays information about the character
    def __repr__(self):
        display = ['\n======<' + self.name + '>======']
        display.append('Is alive? ' + str(self.alive))
        display.append('Did it reach the target? ' + str(self.target_reached))
        display.append('Number of steps taken: ' + str(len(self.trace)))
        display.append('Location trace:')
        for loc in self.trace:
            display.append(str(loc))
        display.append('Attention trace:')
        for attn in self.attention_trace:
            display.append(str(attn))
        if self.name == "prey":
            display.append('Distance to agent trace:')
            for d in self.dist_trace:
                display.append(str(d))
        elif self.name == "predator":
            display.append('Distance to agent trace:')
            for d in self.dist_trace:
                display.append(str(d))
        else:
            display.append('Distance to prey and predator (respectively) trace:')
            for d in self.dist_trace:
                display.append(str(d))
        display.append('===============================\n')
        return "\n".join(display)

# Class for the attention allocation model
class AttentionModel:

    def __init__(self):
        self.name = "AttentionModel"
    
    # Updates the QUBO given the distance to the target
    def QUBO(self, dist):
        # Ratio between distance and maximum possible distance
        d = dist/MAX_DIST

        # Attention level dependent on cost
        Q_cost = {('25','25'): -(1 - 0.25),
            ('50','50'): -(1 - 0.5),
            ('75','75'): -(1 - 0.75),
            ('100','100'): -(1 - 1),
            ('25','50'): -(-(1 - 0.25) - (1 - 0.5)),
            ('25','75'): -(-(1 - 0.25) - (1 - 0.75)),
            ('25','100'): -(-(1 - 0.25) - -(1 - 1)),
            ('50','75'): -(-(1 - 0.5) - (1 - 0.75)),
            ('50','100'): -(-(1 - 0.5) - (1 - 1)),
            ('75','100'): -(-(1 - 0.75) - (1 - 1))}

        # Attention level dependent on distance
        Q_dist = {('25','25'): -d,
            ('50','50'): -0.5*d - 0.4,
            ('75','75'): 0.5*d - 0.9,
            ('100','100'): d - 1,
            ('25','50'): -(-d -0.5*d - 0.4),
            ('25','75'): -(-d +0.5*d - 0.9),
            ('25','100'): -(-d +d - 1),
            ('50','75'): -(-0.5*d - 0.4 +0.5*d - 0.9),
            ('50','100'): -(-0.5*d - 0.4 + d - 1),
            ('75','100'): -(0.5*d - 0.9 + d - 1)}

        # Combine both QUBO formulations (cost and distance)
        Q_complete = {}
        for key in list(Q_cost.keys()):
            Q_complete[key] = Q_cost[key] + Q_dist[key]

        return Q_complete
    
    # Allocates to a character given the distance to their target
    def alloc_attention(self, dist):
        # Get the QUBO formulation for the given distance
        Q = self.QUBO(dist)

        # Run sampler
        sampler = EmbeddingComposite(DWaveSampler())
        
        # Retrieve output
        sampler_output = sampler.sample_qubo(Q, num_reads = NUM_READS)

        # Get the attention
        attention = sampler_output.record.sample[0]
        if attention[0] == 1:
            attention = 100
        elif attention[1] == 1:
            attention = 25
        elif attention[2] == 1:
            attention = 50
        else:
            attention = 75

        return attention


def main():
    # Initialize characters
    agent = Character("agent")
    prey = Character("prey")
    predator = Character("predator")

    # Initialize the attention allocation model
    attention_model = AttentionModel()

    # Run model for n iterations
    for _ in range(ITERATIONS):

        # Get the attention levels for all three characters
        # Prey's attention level using its distance to the agent
        attention_prey = attention_model.alloc_attention(dist(prey.loc, agent.loc)) 
        # Agent's attention level using the average between its distance to the prey and its distance to the predator
        attention_agent = attention_model.alloc_attention((dist(agent.loc, prey.loc) + dist(agent.loc, predator.loc))/2)
        # Predator's attention level using its distance to the agent
        attention_predator = attention_model.alloc_attention(dist(predator.loc, agent.loc))

        # Normalize attention levels so that they don't exceed 100
        total_attention = attention_prey + attention_agent + attention_predator
        attention_prey = attention_prey/total_attention * 100
        attention_agent = attention_agent/total_attention * 100
        attention_predator = attention_predator/total_attention * 100

        # Keep track of attention levels
        prey.track_attention(attention_prey)
        agent.track_attention(attention_agent)
        predator.track_attention(attention_predator)
        
        # Update the location of the characters accordingly
        prey.avoid(agent.perceive(attention_prey)) # Prey avoids agent
        predator.pursue(agent.perceive(attention_predator)) # Predator pursues agent
        if agent.escaping:
            # Agent avoids predator
            agent.avoid(predator.perceive(attention_agent)) 
            agent.escaping = False
        else:
            # Agent pursues prey
            agent.pursue(prey.perceive(attention_agent)) 
            agent.escaping = True

        # Keep track of distances
        prey.track_dist(dist(prey.loc, agent.loc))
        agent.track_dist([dist(prey.loc, agent.loc), dist(predator.loc, agent.loc)])
        predator.track_dist(dist(predator.loc, agent.loc))
    
    print(agent)
    print(prey)
    print(predator)

    print(prey.trace)
    print(agent.trace)
    print(predator.trace)

    return

if __name__ == "__main__":
    main()
