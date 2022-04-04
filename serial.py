"""Predator-Prey Task (Serial Approach)

This implements the Predator-Prey task within quantum computing using the 
serial approach (i.e. at each time step, allocate attention, observe locations,
and decide on optimal movement direction).
"""

import math
from metrics import metrics as metrics_mod
from models import attention as attention_mod
from models import movement as movement_mod
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod

# Number of iterations in the game
ITERATIONS = 1
# Number of reads in the annealer
NUM_READS = 5
# Width and height of the game's coordinate plane
WIDTH = 500
HEIGHT = 500
# For now, speed (how fast a character moves at each time step) is always constant
SPEED = 30

def main():
    # Initialize metrics instance
    metrics = metrics_mod.Metrics("Serial Quantum Implementation")

    # Initialize characters
    agent = agent_mod.Agent(WIDTH, HEIGHT)
    prey = prey_mod.Prey(WIDTH, HEIGHT)
    predator = predator_mod.Predator(WIDTH, HEIGHT)

    # Initialize the attention allocation model
    attention_model = attention_mod.AttentionModel(WIDTH, HEIGHT, NUM_READS)

    # Initialize the movement model
    movement_model = movement_mod.MovementModel(WIDTH, HEIGHT, NUM_READS)

    # Run model for n iterations
    for _ in range(ITERATIONS):

        attn_agent, attn_prey, attn_predator = attention_model.get_attention_levels(agent,
                                                                                prey,
                                                                                predator)
        
        # Prey avoids agent
        prey.avoid(agent.loc, SPEED)
        # Predator pursues agent
        predator.pursue(agent.loc, SPEED)

        # Use the quantum model for the agent's movement
        # call the movement model

        # Get the perceived locations
        agent_perceived = agent.perceive(agent, attn_agent)
        prey_perceived = agent.perceive(prey, attn_prey)
        predator_perceived = agent.perceive(predator, attn_predator)

        movement_model.move(agent, agent_perceived, prey_perceived, predator_perceived, prey.loc, predator.loc, SPEED)

        # Move Agent
        # agent.move(agent_perceived, prey_perceived, predator_perceived, prey.loc, predator.loc, SPEED, BIAS)

    # Add general metrics
    metrics.w = WIDTH
    metrics.h = HEIGHT
    metrics.iterations = ITERATIONS
    metrics.num_reads = NUM_READS

    # Add agent to metrics
    metrics.agent_alive = agent.alive
    metrics.agent_feasted = agent.feasted
    metrics.agent_loc_trace = agent.loc_trace
    metrics.agent_perceived_loc_trace = agent.perceived_agent_trace
    metrics.prey_perceived_loc_trace = agent.perceived_prey_trace
    metrics.predator_perceived_loc_trace = agent.perceived_predator_trace
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

    # Add time to metrics
    metrics.attention_time = attention_model.total_time
    metrics.movement_time = movement_model.total_time
    metrics.total_time = attention_model.total_time + movement_model.total_time

    return metrics

if __name__ == "__main__":
    main()
    