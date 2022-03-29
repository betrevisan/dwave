import classical as classical_mod
import serial as serial_mod

classical_trace = classical_mod.main()
quantum_trace = serial_mod.main()

correct = 0
for (c, q) in zip(classical_trace, quantum_trace):
    if c == q:
        correct += 1

print("Number of iterations on each model: " + str(len(classical_trace)))
print("Reads in the Anneler (specific to the Quantum Model): 5")
print("Number of equal attention allocations: " + str(correct))
print("Accuracy rate of Quantum Model vs Classical Model (regarding attention allocation): " + str(correct/len(classical_trace)))
