# Implements the simplest version of attention selection - selects between 50% and 75%
# with no distinction between the two.

from dwave.system import EmbeddingComposite, DWaveSampler

# QUBO Matrix
Q = {('50','50'): -1,
    ('75','75'): -1,
    ('50','75'): 2}

# Define sampler and run it
sampler = EmbeddingComposite(DWaveSampler())

sampler_output = sampler.sample_qubo(Q,
                                    num_reads = 50)

print(sampler_output)
