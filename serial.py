# This file implements the serial approach to the predator-prey model in quantum computing

from numpy import sqrt
from models import attention as attention_mod
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod

# Number of iterations in the game
ITERATIONS = 20
# Width and height of the game's coordinate plane
WIDTH = 500
HEIGHT = 500
# For now, speed (how fast a character moves at each time step) is always constant
SPEED = 30
# Bias on pursuing over avoiding for the agent's movement
BIAS = 0.8

# Auxiliar function for calculating the distance between two points
def dist(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def main():
    # Initialize characters
    agent = agent_mod.Agent(WIDTH, HEIGHT)
    prey = prey_mod.Prey(WIDTH, HEIGHT)
    predator = predator_mod.Predator(WIDTH, HEIGHT)

    # Initialize the attention allocation model
    attention_model = attention_mod.AttentionModel()

    # Initialize the movement model
    # movement_model = MovementModel()

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

        # Use the quantum model for the agent's movement
        # call the movement model

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
    print(prey)
    print(predator)

    return agent.attention_trace

if __name__ == "__main__":
    main()
    