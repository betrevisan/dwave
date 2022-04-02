# This file implements the serial approach to the predator-prey model in quantum computing

from numpy import sqrt
from models import attention as attention_mod
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod

# Number of iterations in the game
ITERATIONS = 5
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
    attention_model = attention_mod.AttentionModel(WIDTH, HEIGHT, 5)

    # Initialize the movement model
    # movement_model = MovementModel()

    # Run model for n iterations
    for _ in range(ITERATIONS):

        attn_agent, attn_prey, attn_pred = attention_model.get_attention_levels(agent,
                                                                                prey,
                                                                                predator)
        
        # Move Prey and Predator
        prey.avoid(agent.perceive(100), agent.loc, SPEED) # Prey avoids agent
        predator.pursue(agent.perceive(100), agent.loc, SPEED) # Predator pursues agent

        # Use the quantum model for the agent's movement
        # call the movement model

        # Move Agent
        agent.move(agent.perceive(attn_agent),
                    prey.perceive(attn_prey),
                    predator.perceive(attn_pred),
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
    