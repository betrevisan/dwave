"""Predator-Prey Task (Classical Approach)

This implements the Predator-Prey task within classical computing using the 
serial approach (i.e. at each time step, allocate attention, observe locations,
and decide on optimal movement direction).
"""

import math
import time
from metrics import metrics as metrics_mod
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
    # Initialize metrics instance
    metrics = metrics_mod.Metrics("Serial Classical Implementation")

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

        start_attn_time = time.time()
        attn_agent, attn_prey, attn_predator = attention_model.get_attention_levels(agent,
                                                                                    prey,
                                                                                    predator)
        metrics.attention_time += (time.time() - start_attn_time) * 1000000

        # Prey avoids agent
        prey.avoid(agent.loc, SPEED)
        # Predator pursues agent
        predator.pursue(agent.loc, SPEED)

        # Get the perceived locations
        agent_perceived = agent.perceive(agent, attn_agent)
        prey_perceived = agent.perceive(prey, attn_prey)
        predator_perceived = agent.perceive(predator, attn_predator)

        # Move Agent
        start_movement_time = time.time()
        agent.move(agent_perceived, prey_perceived, predator_perceived, prey.loc, predator.loc, SPEED, BIAS)
        metrics.movement_time += (time.time() - start_movement_time) * 1000000 

    # Add agent to metrics
    metrics.agent_alive = agent.alive
    metrics.agent_feasted = agent.feasted
    metrics.agent_loc_trace = agent.loc_trace
    metrics.dist_agent2prey_trace = [dist[0] for dist in agent.dist_trace]
    metrics.dist_agent2predator_trace = [dist[1] for dist in agent.dist_trace]

    # Add prey to metrics
    metrics.prey_alive = prey.alive
    metrics.prey_loc_trace = prey.loc_trace

    # Add predator to metrics
    metrics.predator_feasted = predator.feasted
    metrics.predator_loc_trace = predator.loc_trace

    # Add attention trace to metrics
    metrics.attention_trace = agent.attn_trace

    # Add total time to metrics
    metrics.total_time = (time.time() - start_time) * 1000000

    print(metrics)

    return

if __name__ == "__main__":
    main()
