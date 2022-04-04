"""Comparing Quantum and Classical Implementations

This runs both the classical and the quantum implementations to the Predator-Prey
task and compares their performances using varying metrics.
"""

import classical as classical_mod
import serial as serial_mod

# Run classical version
metrics_classical = classical_mod.main()

# # Run quantum version
# metrics_quantum = serial_mod.main()

# # Compare attention allocation
# correct_attention = 0
# for (c, q) in zip(metrics_classical.attention_trace, metrics_quantum.attention_trace):
#     if c == q:
#         correct_attention += 1

# Print metrics
print(metrics_classical)
# print(metrics_quantum)

# # Print comparison metrics

# print("Number of equal attention allocations: " + str(correct_attention))
# print("Accuracy rate of Quantum Model vs Classical Model (regarding attention allocation): " + str(correct_attention/len(metrics_quantum.attention_trace)))
