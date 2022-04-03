"""Predator-Prey Task (Classical Approach)

This implements the Predator-Prey task within classical computing using the 
serial approach (i.e. at each time step, allocate attention, observe locations,
and decide on optimal movement direction).
"""

import math
import time
from models import attention_classical as attention_mod
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod

# Number of iterations in the game
ITERATIONS = 1
# Width and height of the game's coordinate plane
WIDTH = 500
HEIGHT = 500
# For now, speed is always constant
SPEED = 30
# Bias on pursuing over avoiding for the agent's movement
BIAS = 0.8
        
def main():
    # Compute time stats
    start_time = time.time()

    # Initialize characters
    agent = agent_mod.Agent(WIDTH, HEIGHT)
    prey = prey_mod.Prey(WIDTH, HEIGHT)
    predator = predator_mod.Predator(WIDTH, HEIGHT)

    # Initialize the attention allocation model
    attention_model = attention_mod.AttentionModelClassical(WIDTH, HEIGHT)

    # Run model for n iterations
    for _ in range(ITERATIONS):

        attn_agent, attn_prey, attn_predator = attention_model.get_attention_levels(agent,
                                                                                    prey,
                                                                                    predator)

        # Prey avoids agent
        prey.avoid(agent.loc, SPEED)
        # Predator pursues agent
        predator.pursue(agent.loc, SPEED)

        # Get the perceived locations
        agent_perceived = agent.perceive(agent, attn_agent)
        prey_perceived = agent.perceive(prey, attn_prey)
        predator_perceived = agent.perceive(predator, attn_predator)

        # Move Agent
        agent.move(agent_perceived, prey_perceived, predator_perceived, prey.loc, predator.loc, SPEED, BIAS)
    
    print(agent)
    # print(prey)
    # print(predator)

    # Print compute time (in microseconds)
    end_time = time.time()
    print("Compute time for the classical version: " +  str((end_time - start_time) * 1000000))

    return agent.attn_trace

if __name__ == "__main__":
    main()
