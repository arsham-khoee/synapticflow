"""
Module for monitoring objects.
"""

from typing import Union, Iterable, Optional

import torch

from .neural_populations import NeuralPopulation
from .connections import AbstractConnection


class Monitor:
    """
    Record desired state variables.

    You can record variables of different SNN objects using an instance of this class. For\
    this purpose, you pass the object as `obj` and provide a list of string name of variables\
    you intend to record as `state_variables`. All the recordings will reside on cpu unless\
    you change the `device` value to `cuda`.

    To save the variables at each time step, you should call `record` method at the desired\
    step. By default, number of timesteps is set to 0 and hence the monitor object will\
    basically save the data of that single step you called the `record` for and so is\
    equivalent to the object's variable itself. To record the variables for a specified\
    duration of time, use `set_time_steps` to define the duration and time resolution. Use\
    `get` with the variable name you want to retrieve to obtain the recorded values.

    Also make sure to call `reset_state_variables` before starting any simulation to make the\
    allocations.

    Examples
    --------
    >>> from network.neural_populations import LIFPopulation
    >>> from network.monitors import Monitor
    >>> neuron = LIF(shape=(1,))
    Now, assume there are two variables `s` and `v` in LIFPopulation which indicate spikes
    and voltages respectively.
    >>> monitor = Monitor(neuron, state_variables=["s", "v"])
    >>> time = 10  # time of simulation
    >>> dt = 1.0  # time resolution
    >>> monitor.set_time_steps(time, dt)  # record the whole simulation
    >>> monitor.reset_state_variables()
    >>> for t in range(time):
    ...     # compute input spike trace and call `neuron.forward(input_trace)`
    ...     monitor.record()
    `monitor.record()` should be called within the simulation process. The state variables of
    the given object are so recorded in the simulation time step and is kept in the recording.
    >>> s = monitor.get("s")
    >>> v = monitor.get("v")
    `s` and `v` hold the tensor of spikes and voltages during the simulation. Their shape would
    be `(time, **neuron.shape)`.

    Arguments
    ---------
    obj : NeuralPopulation or AbstractConnection
        The object, states of which is desired to record.
    state_variables : Iterable of str
        Name of variables of interest.
    device : str, Optional
        The device to run the monitor. The default is "cpu".

    """

    def __init__(
        self,
        obj: Union[NeuralPopulation, AbstractConnection],
        state_variables: Iterable[str],
        device: Optional[str] = "cpu",
    ) -> None:
        self.obj = obj
        self.state_variables = state_variables
        self.time_steps = 0
        self.device = device

        self.recording = []

    def set_time_steps(self, time: int, dt: float):
        """
        Set number of time steps to record.

        Parameters
        ----------
        time : int
            The simulation time we intend to record. If 0, Only records one time step\
            at each point.
        dt : float
            Simulation time resolution.

        """
        self.time_steps = int(time / dt)

    def get(self, variable: str) -> torch.Tensor:
        """
        Return recording to user.

        Parameters
        ----------
        variable : str
            The requested variable.

        Returns
        -------
        logs : torch.Tensor
            The recording log of the requested variable.

        """
        logs = torch.cat(self.recording[variable], 0)
        if self.time_steps == 0:
            self.recording[variable] = []
        return logs

    def record(self) -> None:
        """
        Append the current value of the recorded state variables to the\
        recording.

        Returns
        -------
        None

        """
        for var in self.state_variables:
            data = getattr(self.obj, var).unsqueeze(0)
            self.recording[var].append(
                torch.empty_like(data, device=self.device).copy_(
                    data, non_blocking=True
                )
            )
            if self.time_steps > 0:
                self.recording[var].pop(0)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        if self.time_steps == 0:
            self.recording = {var: [] for var in self.state_variables}
        else:
            self.recording = {
                var: [[] for i in range(self.time_steps)] for var in self.state_variables
            }
