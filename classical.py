# This file implements the classical version of the predator-prey model

from numpy import sqrt
import math
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod

# Number of iterations of the model
ITERATIONS = 20
# Width and height of the coordinate plane
WIDTH = 500
HEIGHT = 500
# Maximum distance between two points in the plane
MAX_DIST = sqrt(WIDTH**2 + HEIGHT**2)
# For now, speed is always constant
SPEED = 30
# Bias on pursuing over avoiding for the agent's movement
BIAS = 0.8

# Auxiliar function for calculating the distance between two points
def dist(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Class for the attention allocation model
class AttentionModelClassical:

    def __init__(self):
        self.name = "AttentionModelClassical"

    # Allocates to a character given the distance to their target
    def alloc_attention(self, dist):
        # Ratio between distance and maximum possible distance
        d = dist/MAX_DIST

        attention_levels = [25, 50, 75, 100]

        minimum = math.inf

        attention = 0

        # Iterate over attention levels, keeping track of the one with minimum cost
        for level in attention_levels:
            cost = -(1 - level/100)
            
            if level == 25:
                cost += -d
            elif level == 50:
                cost += -0.5*d - 0.4
            elif level == 75:
                cost += 0.5*d - 0.9
            elif level == 50:
                cost += d - 1

            if cost < minimum:
                minimum = cost
                attention = level
        
        return attention
        
def main():
    # Initialize characters
    agent = agent_mod.Agent(WIDTH, HEIGHT)
    prey = prey_mod.Prey(WIDTH, HEIGHT)
    predator = predator_mod.Predator(WIDTH, HEIGHT)

    # Initialize the attention allocation model
    attention_model = AttentionModelClassical()

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
        agent.track_attention([attention_agent, attention_prey, attention_predator])
        
        # Move Prey and Predator
        prey.avoid(agent.perceive(100), agent.loc, SPEED) # Prey avoids agent
        predator.pursue(agent.perceive(100), agent.loc, SPEED) # Predator pursues agent

        # Move Agent
        agent.move(agent.perceive(attention_agent),
                    prey.perceive(attention_prey),
                    predator.perceive(attention_predator),
                    prey.loc,
                    predator.loc,
                    SPEED,
                    BIAS)

        # Keep track of distances
        agent.track_dist([dist(prey.loc, agent.loc), dist(predator.loc, agent.loc)])
    
    print(agent)
    # print(prey)
    # print(predator)

    return agent.attention_trace

if __name__ == "__main__":
    main()
