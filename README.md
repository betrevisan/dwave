# Predator-Prey Model in Quantum Computing

This repository implements a version of the predator-prey model of cognitive science
in quantum computing. 

The serial.py file has the code to implement a serial approach to the predator-prey model.
Such an approach consists of the following steps:

1) Update the model's parameters from the previous iteration
2) Allocate attention based on a model (maximizing predicted reward)
3) Observe locations
4) Decide on the next optimal location