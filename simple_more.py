# Implements a simple version of attention selection, selecting between 25%, 50%, 75%,
# and 100% with no distinction between them.

from dwave.system import EmbeddingComposite, DWaveSampler

# QUBO Matrix
Q = {('25','25'): -1,
    ('50','50'): -1,
    ('75','75'): -1,
    ('100','100'): -1,
    ('25','50'): 2,
    ('25','75'): 2,
    ('25','100'): 2,
    ('50','75'): 2,
    ('50','100'): 2,
    ('75','100'): 2}

# Define sampler and run it
sampler = EmbeddingComposite(DWaveSampler())

sampler_output = sampler.sample_qubo(Q,
                                    num_reads = 40)

print(sampler_output)
