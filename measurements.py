"""Comparing Quantum and Classical Implementations

This runs both the classical and the quantum implementations to the Predator-Prey
task and compares their performances using varying metrics.
"""

import math
import classical as classical_mod
import serial as serial_mod

# Run classical version
metrics_classical = classical_mod.main()

# Run quantum version
metrics_quantum = serial_mod.main()

# Compare attention allocation
correct_attention = 0
for (c, q) in zip(metrics_classical.attention_trace, metrics_quantum.attention_trace):
    if c == q:
        correct_attention += 1

# Compare movement
total_distance = 0
for (c, q) in zip(metrics_classical.agent_loc_trace[1:], metrics_quantum.agent_loc_trace[1:]):
    total_distance += math.dist(c, q)
average_distance = total_distance/len(metrics_quantum.agent_loc_trace[1:])

# Print metrics
print(metrics_classical)
print(metrics_quantum)

# # Print comparison metrics
print('\n===============================')
print("C O M P A R I S O N\n")
print('Attention Allocation Comparison')
print("\tNumber of equal attention allocations:             " + str(correct_attention))
print("\tTotal number of attention allocations:             " + str(len(metrics_quantum.attention_trace)))
print("\tAttention allocation accuracy rate:                " + str(correct_attention/len(metrics_quantum.attention_trace) * 100) + "%")

print('\nMovement Comparison')
print("\tNumber of steps taken:                             " + str(len(metrics_quantum.agent_loc_trace[1:])))
print("\tAverage distance between step taken by each model: " + "{:.2f}".format(average_distance))
