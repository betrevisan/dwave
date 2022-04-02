import math
import numpy as np
from random import randint, seed

class Agent:
    """
    The Agent class represents the agent in the Predator-Prey task.

    ...

    Attributes
    ----------
    loc : [float]
        Location of the agent [x, y]
    feasted : bool
        Says whether the agent has caught the prey or not
    alive : bool
        Says whether the agent is still alive (i.e. has not been caught)
    loc_trace : [[float]]
        Keeps track of all the locations the agent was in
    attn_trace : [[float]]
        Keeps track of all the attention allocations
    dist_trace : [[float]]
        Keeps track of all distances at each time step
    perceived_agent_trace : [[float]]
        Keeps track of the perceived agent locations
    perceived_prey_trace : [[float]]
        Keeps track of the perceived prey locations
    perceived_predator_trace : [[float]]
        Keeps track of the perceived predator locations
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane

    Methods
    -------
    perceive(target, attention)
        Get the perceived location of the target given an attention level.
    move(agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed, bias)
        Move the agent using the perceived locations, speed of movement, and pursuit bias.
    bounce_back()
        If the agent's location is outside the coordinate plane, bounce back into it.
    track_attn(attention)
        Add the given set of attention levels to the attention trace.
    track_dist(attention)
        Add the given set of attention levels to the attention trace.
    """

    def __init__(self, w, h):
        """
        Parameters
        ----------
        w : int
            Width of the coordinate plane
        h : int
            Height of the coordinate plane
        """
        seed(1)
        self.loc = [randint(int(w/3), int(2*w/3)), randint(0, h)]
        self.feasted = False
        self.alive = True
        self.loc_trace = [list(self.loc)]
        self.attn_trace = []
        self.dist_trace = []
        self.perceived_agent_trace = []
        self.perceived_prey_trace = []
        self.perceived_predator_trace = []
        self.w = w
        self.h = h
        return
    
    def perceive(self, target, attention):
        """Get the target's location given the attention level

        Parameters
        ----------
        target : Agent, Prey, or Predator
            The target of attention
        attention : float
            The attention level

        Returns
        -------
        [float]
            The perceived location

        Raises
        ------
        ValueError
            If given arguments are invalid.
        """
        if target is None or attention < 0:
            raise ValueError("invalid perceived arguments")

        blur = 100 - attention
        x = target.loc[0] + blur
        y = target.loc[1] + blur
        return [x, y]

    def move(self, agent_perceived, prey_perceived, predator_perceived,
                prey_real, predator_real, speed, bias):
        """Move the agent using the perceived locations, speed of movement, and pursuit bias

        Parameters
        ----------
        agent_perceived : [float]
            The agent's perceived location [x, y]
        prey_perceived : [float]
            The prey's perceived location [x, y]
        predator_perceived : [float]
            The predator's perceived location [x, y]
        prey_real : [float]
            The prey's real location [x, y]
        predator_real : [float]
            The predator's real location [x, y]
        speed : float
            The speed of movement
        bias : float
            The bias on pursuing over avoiding

        Returns
        -------
        void

        Raises
        ------
        ValueError
            If given arguments are invalid.
        """
        if agent_perceived is None or prey_perceived is None or predator_perceived is None or
            prey_real is None or predator_real is None:
            raise ValueError("locations must all be valid")

        if speed <= 0:
            raise ValueError("speed must be positive number")
        
        if bias < 0 or bias > 1:
            raise ValueError("bias must be a number between 0 and 1")

        # Track perceived locations
        self.perceived_agent_trace.append(list(agent_perceived))
        self.perceived_prey_trace.append(list(prey_perceived))
        self.perceived_predator_trace.append(list(predator_perceived))

        # If the distance between prey and predator is less than 10 it counts as a contact
        buffer = 10
        # Vector for the agent's real location
        agent_real_v = np.array(self.loc)
        # Vector for the agent's perceived location
        agent_perceived_v = np.array(agent_perceived)
        # Vector for the prey's real location
        prey_real_v = np.array(prey_real)
        # Vector for the prey's perceived location
        prey_perceived_v = np.array(prey_perceived)
        # Vector for the predator's real location
        pred_real_v = np.array(predator_real)
        # Vector for the predator's perceived location
        pred_perceived_v = np.array(predator_perceived)

        # Real distance between the agent and the predator
        real_dist2pred = np.linalg.norm(pred_real_v - agent_real_v)
        # If the agent has been caught, set alive to False
        if real_dist2pred < buffer:
            self.alive = False

        # Reflect the predator's location along the agent's coordinates
        new_pred_perceived_v = agent_perceived_v - (pred_perceived_v - agent_perceived_v)

        # Get point in between predator's and prey's (superposition of pursuit and avoidance)
        super_v = new_pred_perceived_v + bias * (prey_perceived_v - new_pred_perceived_v)
        
        # Vector for the direction of movement
        move_v =  super_v - agent_perceived_v

        # Move agent alongside the superposition vector at a given speed
        d = speed / np.linalg.norm(move_v)
        if d > 1:
            d = 1
        new_loc = np.floor((agent_real_v + d * move_v))

        # Update agent's location
        self.loc = new_loc

        # Real distance between the prey and the agent
        real_dist2prey = np.linalg.norm(prey_real_v - np.array(self.loc))
        # If the agent has reached its prey, set feasted to True
        if real_dist2prey < buffer:
            self.feasted = True
        
        # If the new location is out of range, bounce back
        self.bounce_back()

        # Update location trace
        self.loc_trace.append(list(self.loc))

        # Keep track of distances
        self.track_dist([math.dist(prey.loc, agent.loc), math.dist(predator.loc, agent.loc)])

    # If the location is out of range, bounces it back into range
    def bounce_back(self):
        # Fix x-coordinate, if needed
        if self.loc[0] < 0:
            self.loc[0] = 1
        elif self.loc[0] > self.w:
            self.loc[0] = self.w - 1
        
        # Fix y-coordinate, if needed
        if self.loc[1] < 0:
            self.loc[1] = 1
        elif self.loc[1] > self.h:
            self.loc[1] = self.h - 1
        
        return

    # Add attentions to the attention_trace
    def track_attn(self, attention):
        self.attention_trace.append(attention)
        return

    # Add distance to the dist_trace
    def track_dist(self, dist):
        self.dist_trace.append(dist)
        return

    # Displays information about the character
    def __repr__(self):
        display = ['\n======<AGENT>======']
        display.append('Is alive? ' + str(self.alive))
        display.append('Did it reach the target? ' + str(self.feasted))
        display.append('Number of steps taken: ' + str(len(self.trace)))
        display.append('Location trace:')
        trace_str = ""
        for loc in self.trace:
            trace_str += ", " + str(loc)
        display.append(trace_str)
        display.append('Agent perceived location trace:')
        agent_str = ""
        for loc in self.perceived_agent_trace:
            agent_str += ", " + str(loc)
        display.append(agent_str)
        display.append('Prey perceived location trace:')
        prey_str = ""
        for loc in self.perceived_prey_trace:
            prey_str += ", " + str(loc)
        display.append(prey_str)
        display.append('Predator perceived location trace:')
        predator_str = ""
        for loc in self.perceived_predator_trace:
            predator_str += ", " + str(loc)
        display.append(predator_str)
        display.append('Attention trace (agent, prey, predator):')
        for attn in self.attention_trace:
            display.append(str(attn))
        display.append('Distances trace (dist to prey, dist to predator):')
        for dist in self.dist_trace:
            display.append(str(dist))
        display.append('===============================\n')
        return "\n".join(display)
