import math
import numpy as np
from dwave.system import EmbeddingComposite, DWaveSampler

class MovementModel:
    """
    The MovementModel class represents the quantum model for movement.

    ...

    Attributes
    ----------
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane
    max_dist : float
        Maximum possible distance in the coordinate plane
    num_reads : int
        Number of reads in the annealer
    name : str, optional
        The name of the model

    Methods
    -------
    qubo(dist2prey, dist2pred)
        Updates the QUBO formulation given distance to the prey and to the predator.
    decide_movement(agent, prey, predator)
        Decide on the direction of movement given the three characters.
    move(agent, prey, predator)
        Moves the agent into the direction decided by the quantum model.
    """

    def __init__(self, w, h, num_reads, name="MovementModel"):
        """
        Parameters
        ----------
        w : int
            Width of the coordinate plane
        h : int
            Height of the coordinate plane
        num_reads : int
            Number of reads in the annealer
        name : str, optional
            The name of the model (default is "MovementModel")
        """

        self.w = w
        self.h = h
        self.max_dist = np.sqrt(w**2 + h**2)
        self.num_reads = num_reads
        self.name = name

    def qubo(self, dist2prey, dist2predator):
        # Build the QUBO on the prey's perceived location
        Q_prey = {}
        max_dist_prey = max(dist2prey)
        for i in range(8):
            Q_prey[(str(i),str(i))] = -(1 - dist2prey[i]/max_dist_prey)
        for i in range(8):
            for j in range(i+1, 8):
                Q_prey[(str(i),str(j))] = -(Q_prey[(str(i),str(i))] + Q_prey[(str(j),str(j))])

        # Build the QUBO on the predator's perceived location
        Q_predator = {}
        max_dist_predator = max(dist2predator)
        for i in range(8):
            Q_predator[(str(i),str(i))] = -(1 - dist2predator[i]/max_dist_predator)
        for i in range(8):
            for j in range(i+1, 8):
                Q_predator[(str(i),str(j))] = -(Q_predator[(str(i),str(i))] + Q_predator[(str(j),str(j))])

        # Combine both QUBO formulations
        Q_complete = {}
        for key in list(Q_prey.keys()):
            Q_complete[key] = Q_prey[key] + Q_predator[key]

        return Q_complete

    def decide_movement(self, agent, agent_perceived, prey_perceived, predator_perceived, speed):
        # Calculate the possible directions of movement
        center = agent_perceived
        radius = speed
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        directions = []

        for angle in angles:
            x = radius * np.cos(np.radians(angle)) + center[0]
            y = radius * np.sin(np.radians(angle)) + center[1]
            directions.append([x, y])

        dist2prey = []
        for p in directions:
            dist2prey.append(math.dist(p, prey_perceived))
        
        dist2predator = []
        for p in directions:
            dist2predator.append(math.dist(p, predator_perceived))

        # Pass movement options and location of prey and location of predator to QUBO
        Q = self.qubo(dist2prey, dist2predator)

        # Define sampler
        sampler = EmbeddingComposite(DWaveSampler())
        
        # Run sampler
        sampler_output = sampler.sample_qubo(Q, num_reads = self.num_reads)

        # Get the movement direction
        move_dir_idx = sampler_output.record.sample[0]
        idx = 0
        for i in range(8):
            if move_dir_idx[i] == 1:
                idx = i
                break
        move_dir = directions[idx]
        return move_dir

    def move(self, agent, agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed):
        # Get the point to move to
        move_dir = self.decide_movement(agent, agent_perceived, prey_perceived, predator_perceived, speed)

        # Move the agent in the given direction
        agent.move_quantum(agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed, move_dir)

