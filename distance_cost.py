# This file implements the QUBO for attention allocation based on the
# most recent perceived distance to target.

from dwave.system import EmbeddingComposite, DWaveSampler

distance = 470
total_distance = 500
d = distance/total_distance
# Can be tuned to change how much influence distance has on attention allocation
constant = 80

# QUBO Matrix
# Q = {('50','50'): d - 1,
#     ('75','75'): -d,
#     ('50','75'): 1}
Q = {('25','25'): d - 1,
    ('50','50'): 0.5*d - 0.9,
    ('75','75'): -0.5*d - 0.4,
    ('100','100'): -d,
    ('25','50'): -(d - 1 + 0.5*d - 0.9),
    ('25','75'): -(d - 1 - 0.5*d - 0.4),
    ('25','100'): -(d - 1 - d),
    ('50','75'): -(0.5*d - 0.9 - 0.5*d - 0.4),
    ('50','100'): -(0.5*d - 0.9 - d),
    ('75','100'): -(-0.5*d - 0.4 - d)}
    
# Define sampler and run it
sampler = EmbeddingComposite(DWaveSampler())

sampler_output = sampler.sample_qubo(Q,
                                    num_reads = 50)

print(sampler_output)