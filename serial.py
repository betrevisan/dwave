# This file implements the serial approach to the predator-prey model in quantum computing

from dwave.system import EmbeddingComposite, DWaveSampler
from random import randrange, randint

# Number of iterations of the model
ITERATIONS = 5
# Number of reads in the annealer
NUM_READS = 15
# Width and height of the coordinate plane
WIDTH = 500
HEIGHT = 500
# Maximum distance between two points in the plane
MAX_DIST = sqrt(WIDTH**2 + HEIGHT**2)
# For now, speed is always constant
SPEED = 10

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
        self.trace = [self.loc]
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
            self.loc[1] = slope * x + b
        
        # Update trace
        self.trace.append(self.loc)
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
            self.loc[1] = slope * x + b
        
        # Update trace
        self.trace.append(self.loc)
        return

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
        Q_dist = {('25','25'): d - 1,
            ('50','50'): 0.5*d - 0.9,
            ('75','75'): -0.5*d - 0.4,
            ('100','100'): -d,
            ('25','50'): -(d - 1 + 0.5*d - 0.9),
            ('25','75'): -(d - 1 - 0.5*d - 0.4),
            ('25','100'): -(d - 1 - d),
            ('50','75'): -(0.5*d - 0.9 - 0.5*d - 0.4),
            ('50','100'): -(0.5*d - 0.9 - d),
            ('75','100'): -(-0.5*d - 0.4 - d)}

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
        # TODO TODO TODO TODO STOPPED HERE

        
        # Update the location of the characters accordingly
        prey.avoid(agent.perceive(attention_prey)) # Prey avoids agent
        predator.pursue(agent.perceive(attention_predator)) # Predator pursues agent
        # IS THIS ALLOWED? OR SHOULD THE AGENT DECIDE ON ONLY ONE OF THESE TWO POSSIBLE MOVES?
        agent.avoid(predator.perceive(attention_agent)) # Agent avoids predator
        agent.pursue(prey.perceive(attention_agent)) # Agent pursues prey
    
    return


if __name__ == "__main__":
    main()
