# This file implements the serial approach to the predator-prey model in quantum computing

from dwave.system import EmbeddingComposite, DWaveSampler
from random import randrange

ITERATIONS = 5
NUM_READS = 15

def updateQUBO(distance=None, total_distance=1000):
    if distance is None:
        distance = randrange(total_distance)
    
    d = distance/total_distance

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

    Q_distance = {('25','25'): d - 1,
        ('50','50'): 0.5*d - 0.9,
        ('75','75'): -0.5*d - 0.4,
        ('100','100'): -d,
        ('25','50'): -(d - 1 + 0.5*d - 0.9),
        ('25','75'): -(d - 1 - 0.5*d - 0.4),
        ('25','100'): -(d - 1 - d),
        ('50','75'): -(0.5*d - 0.9 - 0.5*d - 0.4),
        ('50','100'): -(0.5*d - 0.9 - d),
        ('75','100'): -(-0.5*d - 0.4 - d)}

    Q_complete = {}

    for key in list(Q_cost.keys()):
        Q_complete[key] = Q_cost[key] + Q_distance[key]

    return Q_complete

# Location of the target
real_location = {"x": 50, "y": 20}
# Location of the agent
agent_location = {"x": 100, "y": 40}

distance = sqrt((real_location["x"] - agent_location["x"])**2 + (real_location["y"] - agent_location["y"])**2)
total_distance = 500

for i in range(ITERATIONS):
    # Pass current distance to function that updates the attention QUBO
    Q = updateQUBO(distance, total_distance)

    # Run sampler
    sampler = EmbeddingComposite(DWaveSampler())

    sampler_output = sampler.sample_qubo(Q,
                                        num_reads = NUM_READS)

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
    print(attention)

    # Observe the location
    perceived_location = blur(real_location, attention)

    # Move towards the location
    slope = (perceived_location["y"] - agent_location["y"]) / (perceived_location["x"] - agent_location["x"])
    b = agent_location["y"] - (slope * agent_location["x"])
    movex = -min([abs(perceived_location["x"] - agent_location["x"]), steps])
    if perceived_location["x"] - agent_location["x"] < 0:
        movex = -movex
    agent_location["x"] = agent_location["x"] + movex
    agent_location["y"] = slope * x + b

    # Update the distance
    distance = sqrt((real_location["x"] - agent_location["x"])**2 + (real_location["y"] - agent_location["y"])**2)

def blur(real_location, attention_level):
    blur = 100 - attention_level
    real_location["x"] = real_location["x"] + blur
    real_location["y"] = real_location["y"] + blur
    return real_location
