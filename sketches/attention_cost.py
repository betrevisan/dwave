# This file implements the QUBO for attention allocation based on the
# cost of each attention level.

from dwave.system import EmbeddingComposite, DWaveSampler

Q = {('25','25'): -(1 - 0.25),
    ('50','50'): -(1 - 0.5),
    ('75','75'): -(1 - 0.75),
    ('100','100'): -(1 - 1),
    ('25','50'): -(-(1 - 0.25) - (1 - 0.5)),
    ('25','75'): -(-(1 - 0.25) - (1 - 0.75)),
    ('25','100'): -(-(1 - 0.25) - -(1 - 1)),
    ('50','75'): -(-(1 - 0.5) - (1 - 0.75)),
    ('50','100'): -(-(1 - 0.5) - (1 - 1)),
    ('75','100'): -(-(1 - 0.75) - (1 - 1))}

# Define sampler and run it
sampler = EmbeddingComposite(DWaveSampler())

sampler_output = sampler.sample_qubo(Q,
                                    num_reads = 50)

print(sampler_output)