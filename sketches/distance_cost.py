# This file implements the QUBO for attention allocation based on the
# most recent perceived distance to target.

from dwave.system import EmbeddingComposite, DWaveSampler

distance = 10
total_distance = 500
d = distance/total_distance

# QUBO Matrix
# Q = {('50','50'): d - 1,
#     ('75','75'): -d,
#     ('50','75'): 1}
Q = {('25','25'): -d,
    ('50','50'): -0.5*d - 0.4,
    ('75','75'): 0.5*d - 0.9,
    ('100','100'): d - 1,
    ('25','50'): -(-d -0.5*d - 0.4),
    ('25','75'): -(-d +0.5*d - 0.9),
    ('25','100'): -(-d +d - 1),
    ('50','75'): -(-0.5*d - 0.4 +0.5*d - 0.9),
    ('50','100'): -(-0.5*d - 0.4 + d - 1),
    ('75','100'): -(0.5*d - 0.9 + d - 1)}

# Define sampler and run it
sampler = EmbeddingComposite(DWaveSampler())

sampler_output = sampler.sample_qubo(Q,
                                    num_reads = 50)

print(sampler_output)