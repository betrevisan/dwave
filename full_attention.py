# This file implements the full attention allocatin formulation using distance
# to target and attention cost

from dwave.system import EmbeddingComposite, DWaveSampler

distance = 470
total_distance = 500
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

# Define sampler and run it
sampler = EmbeddingComposite(DWaveSampler())

sampler_output = sampler.sample_qubo(Q_complete,
                                    num_reads = 50)

print(sampler_output)
