"""Comparing Quantum and Classical Implementations

This runs both the classical and the quantum implementations to the Predator-Prey
task and compares their performances using varying metrics.
"""

import classical as classical_mod
import serial as serial_mod

# Run classical version
classical_trace = classical_mod.main()

# Run quantum version
quantum_trace = serial_mod.main()

# Compare attention allocation
correct = 0
for (c, q) in zip(classical_trace, quantum_trace):
    if c == q:
        correct += 1

# Print comparison metrics
print("Number of iterations on each model: " + str(len(classical_trace)))
print("Reads in the Annealer (specific to the Quantum Model): 5")
print("Number of equal attention allocations: " + str(correct))
print("Accuracy rate of Quantum Model vs Classical Model (regarding attention allocation): " + str(correct/len(classical_trace)))
