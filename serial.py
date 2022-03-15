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
        agent.track_attention_prey(attention_prey)
        agent.track_attention_agent(attention_agent)
        agent.track_attention_predator(attention_predator)
        
        # Move Prey and Predator
        prey.avoid(agent.perceive(100)) # Prey avoids agent
        predator.pursue(agent.perceive(100)) # Predator pursues agent

        # Move Agent
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

    return

if __name__ == "__main__":
    main()




# The agent will move in a superposition of avoidance and pursuit (give a heavier weight to pursuit)
# Calculate the error based on the attention that would be allocated by a classical system vs that of dwave
# Calculate the error of where agent moves vs where it should be moving with full attention
# Calculate the error of where the agent moves with dwave vs with the classical version
# To get these errors, first do not have the prey and the predator move
