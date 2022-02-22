# This file implements the QUBO for attention allocation based on the
# most recent perceived distance to target.

from dwave.system import EmbeddingComposite, DWaveSampler

distance = 250
total_distance = 500
d = distance/total_distance
# Can be tuned to change how much influence distance has on attention allocation
constant = 80

# QUBO Matrix
Q = {('50','50'): d - 1,
    ('75','75'): -d,
    ('50','75'): 1}

# Define sampler and run it
sampler = EmbeddingComposite(DWaveSampler())

sampler_output = sampler.sample_qubo(Q,
                                    num_reads = 50)

print(sampler_output)