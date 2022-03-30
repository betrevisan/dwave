# Quantum implementation of the predator-prey model using a CQM (Constrained Quadratic Model)
# Hopefully this can make the implementation of the parallel model easier
# Joins the attention allocation for all three characters into one model

from dimod import ConstrainedQuadraticModel
import numpy as np
from dimod import Binary
from dwave.system import LeapHybridCQMSampler

# Define possible attention allocations

# Define CQM
cqm = ConstrainedQuadraticModel()
# Objective function: minimize cost of allocated attention
allocation_cost = # cost function (writing it in terms of reward - cost might make it easier/more intuitive)

cqm.set_objective(allocation_cost)

# Constrain #1: only one attention level per character
cqm.add_constraint()

# Constraint #2: total of attention levels cannot exceed 100
cqm.add_constraint()

# Sampler
sampler = LeapHybridCQMSampler()

sampleset = sampler.sample_cqm(cqm, time_limit=180, label="Predator-Prey (CQM)")
sampleset.first
