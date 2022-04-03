from numpy import sqrt

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
        self.max_dist = sqrt(w**2 + h**2)
        self.num_reads = num_reads
        self.name = name

    def move(self, agent, prey, predator):
        # Get the point to move to

        # Update location

        # Update location trace