"""
Module for neuronal dynamics and populations.
"""

from functools import reduce
from abc import abstractmethod
from operator import mul
from typing import Union, Iterable, Optional

import torch


class NeuralPopulation(torch.nn.Module):
    """
    Base class for implementing neural populations.

    Make sure to implement the abstract methods in your child class. Note that this template\
    will give you homogeneous neural populations in terms of excitations and inhibitions. You\
    can modify this by removing `is_inhibitory` and adding another attribute which defines the\
    percentage of inhibitory/excitatory neurons or use a boolean tensor with the same shape as\
    the population, defining which neurons are inhibitory.

    The most important attribute of each neural population is its `shape` which indicates the\
    number and/or architecture of the neurons in it. When there are connected populations, each\
    pre-synaptic population will have an impact on the post-synaptic one in case of spike. This\
    spike might be persistent for some duration of time and with some decaying magnitude. To\
    handle this coincidence, four attributes are defined:
    - `spike_trace` is a boolean indicating whether to record the spike trace in each time step.
    - `additive_spike_trace` would indicate whether to save the accumulated traces up to the\
        current time step.
    - `tau_s` will show the duration by which the spike trace persists by a decaying manner.
    - `trace_scale` is responsible for the scale of each spike at the following time steps.\
        Its value is only considered if `additive_spike_trace` is set to `True`.

    Make sure to call `reset_state_variables` before starting the simulation to allocate\
    and/or reset the state variables such as `s` (spikes tensor) and `traces` (trace of spikes).\
    Also do not forget to set the time resolution (dt) for the simulation.

    Each simulation step is defined in `forward` method. You can use the utility methods (i.e.\
    `compute_potential`, `compute_spike`, `refractory_and_reset`, and `compute_decay`) to break\
    the differential equations into smaller code blocks and call them within `forward`. Make\
    sure to call methods `forward` and `compute_decay` of `NeuralPopulation` in child class\
    methods; As it provides the computation of spike traces (not necessary if you are not\
    considering the traces). The `forward` method can either work with current or spike trace.\
    You can easily work with any of them you wish. When there are connected populations, you\
    might need to consider how to convert the pre-synaptic spikes into current or how to\
    change the `forward` block to support spike traces as input.

    There are some more points to be considered further:
    - Note that parameters of the neuron are not specified in child classes. You have to\
        define them as attributes of the corresponding class (i.e. in __init__) with suitable\
        naming.
    - In case you want to make simulations on `cuda`, make sure to transfer the tensors\
        to the desired device by defining a `device` attribute or handling the issue from\
        upstream code.
    - Almost all variables, parameters, and arguments in this file are tensors with a\
        single value or tensors of the shape equal to population`s shape. No extra\
        dimension for time is needed. The time dimension should be handled in upstream\
        code and/or monitor objects.

    Arguments
    ---------
    n : int, Optional
        Number of neurons in the population
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    is_inhibitory : False, Optional
        Whether the neurons are inhibitory or excitatory. The default is False.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 15.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        sum_input: bool = False,
        is_inhibitory: bool = False,
        learning: bool = True,
        dt: Union[float, torch.Tensor] = 0.1,
        R: Union[float, torch.Tensor] = 20.0,
        **kwargs
    ) -> None:
        """
        Initializes the attributes of the class.
        
        Parameters:
        ----------
        n : int or None, optional
            Number of neurons in the layer (default value is None)
        shape : Iterable[int] or None, optional
            Shape of the layer (default value is None)
        spike_trace : bool, optional
            Whether to use spike trace or not (default value is True)
        additive_spike_trace : bool, optional
            Whether to add spike trace to existing trace or not (default value is True)
        tau_s : float or torch.Tensor, optional
            Decay time constant for spike trace (default value is 15.)
        trace_scale : float or torch.Tensor, optional
            Scaling factor for spike trace (default value is 1.)
        sum_input : bool, optional
            Whether to sum all given inputs or not (default value is False)
        is_inhibitory : bool, optional
            Whether the layer is inhibitory or not (default value is False)
        learning : bool, optional
            Whether the layer is capable of learning or not (default value is True)
        dt : float or torch.Tensor, optional
            Time step for simulation (default value is 0.1)
            
        Returns:
        -------
        None
        """
        super().__init__()

        assert (n is not None or shape is not None), "Must provide either number of neurons or shape of layer"
        
        if n is None:
            self.n = reduce(mul, shape) # Number of neurons product of shape
        else:
            self.n = n # Number of neurons
            
        if shape is None:
            self.shape = [self.n] # Shape is equal to the of the layer
        else:
            self.shape = shape # Shape is passed in as an argument
        
        assert self.n == reduce(mul, self.shape), "Number of neurons and shape do not match"
        
        self.spike_trace = spike_trace
        self.additive_spike_trace = additive_spike_trace
        self.sum_input = sum_input # Whether to sum all inputs

        if self.spike_trace:
            self.register_buffer("traces", torch.zeros(*self.shape))
            self.register_buffer("tau_s", torch.tensor(tau_s))

            if self.additive_spike_trace:
                self.register_buffer("trace_scale", torch.tensor(trace_scale))

            self.register_buffer("trace_decay", torch.empty_like(self.tau_s))

        self.is_inhibitory = is_inhibitory
        self.learning = learning

        self.register_buffer("s", torch.ByteTensor())
        self.s = torch.zeros(*self.shape, device=self.s.device, dtype=torch.bool)
        self.register_buffer("R", torch.tensor(R))
        
        if self.sum_input:
            self.register_buffer("summed", torch.FloatTensor()) # Inputs summation
        
        self.dt = dt

    @abstractmethod
    def forward(self, x: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        x : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        if self.spike_trace:
            self.traces *= self.trace_decay

            if self.additive_spike_trace:
                self.traces += self.trace_scale * self.s.float()
            else:
                self.traces.masked_fill_(self.s, 1)
        
        # Add current recent input tensor to previous ones
        if self.sum_input:
            self.summed += x.float()

    @abstractmethod
    def compute_potential(self) -> None:
        """
        Compute the potential of neurons in the population.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_spike(self) -> None:
        """
        Compute the spike tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Refractor and reset the neurons.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Set the decays.

        Returns
        -------
        None

        """
        self.dt = torch.tensor(self.dt)

        if self.spike_trace:
            self.trace_decay = torch.exp(-self.dt/self.tau_s)  # Spike trace decay (per timestep).

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        self.s.zero_()

        if self.spike_trace:
            self.traces.zero_()

        if self.sum_input:
            self.summed.zero_()  # Inputs summation.

    def train(self, mode: bool = True) -> "NeuralPopulation":
        """
        Set the population's training mode.

        Parameters
        ----------
        mode : bool, optional
            Mode of training. `True` turns on the training while `False` turns\
            it off. The default is True.

        Returns
        -------
        NeuralPopulation

        """
        self.learning = mode
        return super().train(mode)

class InputPopulation(NeuralPopulation):
    """
    Neural population for user-defined spike pattern.

    This class is implemented for future usage. Extend it if needed.

    Arguments
    ---------
    n : int, Optional
        Number of neurons in the population.
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            n = n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            sum_input=sum_input,
            trace_scale=trace_scale,
            learning=learning,
        )
        """
        Initializes the attributes of the class.
        
        Parameters:
        ----------
        n : int or None, optional
            Number of neurons in the layer (default value is None)
        shape : Iterable[int] or None, optional
            Shape of the layer (default value is None)
        spike_trace : bool, optional
            Whether to use spike trace or not (default value is True)
        additive_spike_trace : bool, optional
            Whether to add spike trace to existing trace or not (default value is True)
        tau_s : float or torch.Tensor, optional
            Decay time constant for spike trace (default value is 15.)
        trace_scale : float or torch.Tensor, optional
            Scaling factor for spike trace (default value is 1.)
        sum_input : bool, optional
            Whether to sum all given inputs or not (default value is False)
        is_inhibitory : bool, optional
            Whether the layer is inhibitory or not (default value is False)
        learning : bool, optional
            Whether the layer is capable of learning or not (default value is True)
        dt : float or torch.Tensor, optional
            Time step for simulation (default value is 0.1)
            
        Returns:
        -------
        None
        """

    def forward(self, x: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        x : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        self.s = x

        super().forward(x)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()

class IFPopulation(NeuralPopulation):
    """
    A Integrate and Fire population layer
    """
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        R: Union[float, torch.Tensor] = 20.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Initializes the attributes of the class.
        
        Parameters:
        ----------
        n : int or None, optional
            Number of neurons in the layer (default value is None)
        shape : Iterable[int] or None, optional
            Shape of the layer (default value is None)
        spike_trace : bool, optional
            Whether to use spike trace or not (default value is True)
        additive_spike_trace : bool, optional
            Whether to add spike trace to existing trace or not (default value is False)
        tau_s : float or torch.Tensor, optional
            Decay time constant for spike trace (default value is 10.)
        threshold : float or torch.Tensor, optional
            Threshold potential (default value is -52.)
        rest_pot : float or torch.Tensor, optional
            Resting potential (default value is -62.)
        refrac_length : float or torch.Tensor, optional
            Refractory period length (default value is 5)
        dt : float, optional
            Time step for simulation (default value is 0.1)
        lower_bound : float or None, optional
            Lower bound for the neuron's potential (default value is None)
        sum_input : bool, optional
            Whether to sum all given inputs or not (default value is False)
        trace_scale : float or torch.Tensor, optional
            Scaling factor for spike trace (default value is 1.)
        is_inhibitory : bool, optional
            Whether the layer is inhibitory or not (default value is False)
        R : Union[float, torch.Tensor], optional
            Resistance of neuron. (default: 20.0)
        learning : bool, optional
            Whether the layer is capable of learning or not (default value is True)
        **kwargs : Any
            Additional keyword arguments
            
        Returns:
        -------
        None
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            sum_input=sum_input,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float))
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.rest_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound
        

    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)
        

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.v += (self.refrac_count <= 0).float() * x * self.R

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.rest_pot)
        

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()


    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset
        

class LIFPopulation(NeuralPopulation):
    """
    Leaky Integrate and Fire Neural Population
    """
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        R: Union[float, torch.Tensor] = 20.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Initializes a spiking neuron with the given parameters.

        Parameters:
        ----------
        n : int, optional
            Number of neurons in this layer.
        shape : Iterable[int], optional
            Shape of the input tensor to the layer.
        spike_trace : bool, optional
            Indicates whether to use synaptic traces or not.
        additive_spike_trace : bool, optional
            If true, uses additive spike traces instead of multiplicative ones.
        tau_s : float or torch.Tensor, optional
            Decay time constant for spike trace (default value is 10.)
        threshold : float or torch.Tensor, optional
            The spike threshold of the neuron.
        rest_pot : float or torch.Tensor, optional
            The resting potential of the neuron.
        reset_pot : float or torch.Tensor, optional
            The reset potential of the neuron.
        refrac_length : float or torch.Tensor, optional
            The refractory period length of the neuron in timesteps.
        dt : float, optional
            The time step length.
        lower_bound : float, optional
            Minimum value for the membrane potential of the neuron.
        sum_input : bool, optional
            If true, sums input instead of averaging it.
        trace_scale : float, optional
            Scaling factor for the synaptic traces.
        is_inhibitory : bool, optional
            Indicates whether the neuron is inhibitory or not.
        R : Union[float, torch.Tensor], optional
            Resistance of neuron. (default: 20.0)
        learning : bool, optional
            Indicates whether the neuron should update its weights during training.

        Returns:
        -------
        None
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            sum_input=sum_input,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float))
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float))
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.rest_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float))  # Time constant of neuron voltage decay.
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> None:
        """
        Performs a forward pass through the spiking neuron.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)
        
    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the potential of the neuron.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
            
        self.v += (( - (self.v - self.rest_pot) + self.R * x) * self.dt / self.tau_s) * (self.refrac_count <= 0).float()
    
    def compute_spike(self) -> None:
        """
        Checks if the neuron is spiking.
        
        Returns:
        -------
        None
        """
        self.s = self.v >= self.pot_threshold

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)
        
    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the voltage decay of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()

    def reset_state_variables(self) -> None:
        """
        Resets the state variables of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

        

class BLIFPopulation(NeuralPopulation):
    """
    Boosted Leaky Integrate and Fire Neural Population
    """
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.0,
        threshold: Union[float, torch.Tensor] = 40.0,
        refrac_length: Union[float, torch.Tensor] = 5.0,
        dt: float = 0.1,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        R: Union[float, torch.Tensor] = 20.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Constructor method for spiking neuron.
        
        Parameters:
        ----------
        n : int or None, optional
            Number of neurons in the layer.
        shape : tuple of ints or None, optional
            Shape of the input tensor to the layer, excluding batch size.
        spike_trace : bool, default=True
            Whether to use spike traces or not. If True, enables exponential decay of spike traces.
        additive_spike_trace : bool, default=False
            Whether to use additive spike traces or not. If True, uses additive update rule for spike traces.
        tau_s : float or torch.Tensor, default=10.0
            Decay time constant for spike trace.
        threshold : float or torch.Tensor, default=40.0
            Threshold potential at which a neuron spikes.
        refrac_length : float or torch.Tensor, default=5.0
            Refractory period length after a neuron spikes in ms.
        dt : float, default=0.1
            Time step length for the neurons in ms.
        sum_input : bool, default=False
            Whether to sum incoming input instead of averaging.
        trace_scale : float or torch.Tensor, default=1.0
            Scaling factor for spike traces.
        is_inhibitory : bool, default=False
            Determines whether the neurons are inhibitory or excitatory.
        R : Union[float, torch.Tensor], optional
            Resistance of neuron. (default: 20.0)
        learning : bool, default=True
            Whether to enable learning on the layer.
        **kwargs
            Other arguments to pass to the parent constructor.
            
        Returns:
        -------
        None
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            sum_input=sum_input,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R = R
        )

        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = torch.zeros(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float))  # Tau_s
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()


    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        super().forward(x)
        

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """

        self.v += (( - (self.v) + self.R * x) * self.dt / self.tau_s) * (self.refrac_count <= 0).float()

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check for spiking neuron
        self.s = self.v >= self.pot_threshold


    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, 0)
        

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()


    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(0) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset


class ALIFPopulation(NeuralPopulation):
    """
    Layer of Adaptive Leaky Integrate and Fire neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        tau_w: Union[float, torch.Tensor] = 20.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5.,
        dt: float = 0.1,
        a0: Union[float, torch.Tensor] = 1.,
        b: Union[float, torch.Tensor] = 2.,
        R: Union[float, torch.Tensor] = 20.,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        tau_v: Union[float, torch.Tensor] = 4.0,
        **kwargs
    ) -> None:
        """
        Set class parameters
        
        Args:
            n (Optional[int]): Number of neurons. (default: None)
            shape (Optional[Iterable[int]]): Shape of the input tensor. (default: None)
            spike_trace (bool): Whether to use spike trace or not. (default: True)
            additive_spike_trace (bool): Whether to use additive spike trace or not. (default: False)
            tau_s (Union[float, torch.Tensor]): Synaptic time constant. (default: 10.0)
            tu_w (Union[float, torch.Tensor]): Adaptation time constant. (default: 20.0)
            threshold (Union[float, torch.Tensor]): Spiking threshold. (default: -52.0)
            rest_pot (Union[float, torch.Tensor]): Resting potential. (default: -62.0)
            reset_pot (Union[float, torch.Tensor]): Reset potential. (default: -62.0)
            refrac_length (Union[float, torch.Tensor]): Refractory length. (default: 5.0)
            dt (float): Time step size. (default: 0.1)
            a0 (Union[float, torch.Tensor]): Parameter used in calculating adaptation current. (default: 1.0)
            b (Union[float, torch.Tensor]): Parameter used in calculating adaptation current. (default: 2.0)
            R (Union[float, torch.Tensor]): Resistance of neuron. (default: 20.0)
            lower_bound (float): Lower bound on membrane potential. (default: None)
            sum_input (bool): Whether to sum input over last dimension. (default: False)
            trace_scale (Union[float, torch.Tensor]): Scaling factor for trace. (default: 1.0)
            is_inhibitory (bool): Whether neuron is inhibitory. (default: False)
            learning (bool): Whether to enable synaptic plasticity. (default: True)
            tau_v (Union[float, torch.Tensor]): Membrane potential time constant. (default: 4.0)
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            sum_input=sum_input,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float)) # Rest potential
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float)) # Tau_s
        self.register_buffer("tau_v", torch.tensor(tau_v, dtype=torch.float)) # Tau_v
        self.register_buffer("tau_w", torch.tensor(tau_w, dtype=torch.float)) # Tau_w
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float)) # Reset potential
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float)) # Spiking Threshold
        self.register_buffer("refrac_length", torch.tensor(refrac_length)) # Refractor length
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("w", torch.FloatTensor()) # Adaptation variable
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.rest_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.w = torch.zeros_like(self.v, device=self.w.device)
        self.register_buffer("a0", torch.tensor(a0, dtype=torch.float)) # a_0
        self.register_buffer("b", torch.tensor(b, dtype=torch.float)) # b
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound


    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)
        

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        # Compute new potential with decay voltages.
        self.v += (( - (self.v - self.rest_pot) - self.R * self.w + self.R * x) * self.dt / self.tau_v) * (self.refrac_count <= 0).float()
        self.w += ((self.a0 * (self.v - self.rest_pot) - self.w + self.b * self.tau_w * (self.s.float())) * self.dt / self.tau_w) * (self.refrac_count <= 0).float()
                

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check for spiking neuron
        self.s = self.v >= self.pot_threshold


    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to reset potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)
        

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()


    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset
        

class ELIFPopulation(NeuralPopulation):
    """
    Layer of Exponential Leaky Integrate and Fire neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        theta_rh: Union[float, torch.Tensor] = -60.,
        delta_T: Union[float, torch.Tensor] = 1.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        R: Union[float, torch.Tensor] = 20.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Constructor method for a exponential leaky neural network neuron model.

        Parameters:
        - n (Optional[int]): An optional integer for the number of neurons. Default is None.
        - shape (Optional[Iterable[int]]): An optional iterable representing the shape of the neuron model. Default is None.
        - spike_trace (bool): A flag indicating whether to use spike trace or not. Default is True.
        - additive_spike_trace (bool): A flag indicating whether to add spike trace or replace it. Default is False.
        - tau_s (Union[float, torch.Tensor]): A float or tensor representing tau_s value for neuron model. Default is 10.
        - threshold (Union[float, torch.Tensor]): A float or tensor representing spike threshold value for neuron model. Default is -52.
        - theta_rh (Union[float, torch.Tensor]): A float or tensor representing the resting potential value for neuron model. Default is -60.
        - delta_T (Union[float, torch.Tensor]): A float or tensor representing sharpness of the neuron's voltage threshold. Default is 1.
        - rest_pot (Union[float, torch.Tensor]): A float or tensor representing the resting potential value for neuron model. Default is -62.
        - reset_pot (Union[float, torch.Tensor]): A float or tensor representing the reset potential value for neuron model. Default is -62.
        - refrac_length (Union[float, torch.Tensor]): A float or tensor representing the refractory period length for neuron model. Default is 5.
        - dt (float): A float representing the time step for simulation. Default is 0.1.
        - lower_bound (float): A float representing the lower bound limit for the neuron potential. Default is None.
        - sum_input (bool): A flag indicating whether to sum the input or not. Default is False.
        - trace_scale (Union[float, torch.Tensor]): A float or tensor representing the scaling factor for spike trace. Default is 1.
        - is_inhibitory (bool): A flag indicating whether neuron is inhibitory. Default is False.
        - R (Union[float, torch.Tensor]): Resistance of neuron. (default: 20.0)
        - learning (bool): A flag indicating whether learning is enabled or disabled. Default is True.
        - **kwargs: Additional keyword arguments.

        Returns:
        - None.
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float)) # Rest potential
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float)) # Tau_s
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float)) # Reset potential
        self.register_buffer("theta_rh", torch.tensor(theta_rh, dtype=torch.float)) # Theta_rh potential
        self.register_buffer("delta_T", torch.tensor(delta_T, dtype=torch.float)) # Delta_T : sharpness
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float)) # Spiking Threshold
        self.register_buffer("refrac_length", torch.tensor(refrac_length)) # Refractor length
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.reset_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        # Compute new potential of neuron
        self.v += ((self.R * x + self.rest_pot - self.v + self.delta_T * torch.exp((self.v - self.theta_rh)/ self.delta_T)) / self.tau_s * self.dt) * (self.refrac_count <= 0).float()

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check if neuron is spiking or not
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()
        

    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.reset_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset
        
class QLIFPopulation(NeuralPopulation):
    """
    Layer of Quadratic Leaky Integrate and Fire neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 30.,
        threshold: Union[float, torch.Tensor] = -52.,
        a0: Union[float, torch.Tensor] = 1,
        critical_pot: Union[float, torch.Tensor] = -54.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        R: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Constructor method for a quadratic leaky integrate and fire neural network.

        Parameters:
        - n (Optional[int]): An optional integer for the number of neurons. Default is None.
        - shape (Optional[Iterable[int]]): An optional iterable representing the shape of the neuron model. Default is None.
        - spike_trace (bool): A flag indicating whether to use spike trace or not. Default is True.
        - additive_spike_trace (bool): A flag indicating whether to add spike trace or replace it. Default is False.
        - tau_s (Union[float, torch.Tensor]): A float or tensor representing tau_s value for neuron model. Default is 10.
        - threshold (Union[float, torch.Tensor]): A float or tensor representing spike threshold value for neuron model. Default is -52.
        - a0 (Union[float, torch.Tensor]): A float or tensor representing initial quadratic parameter value. Default is 1.
        - critical_pot (Union[float, torch.Tensor]): A float or tensor representing the critical potential value for quadratic parameter. Default is 0.8.
        - rest_pot (Union[float, torch.Tensor]): A float or tensor representing the resting potential value for neuron model. Default is -62.
        - reset_pot (Union[float, torch.Tensor]): A float or tensor representing the reset potential value for neuron model. Default is -62.
        - refrac_length (Union[float, torch.Tensor]): A float or tensor representing the refractory period length for neuron model. Default is 5.
        - dt (float): A float representing the time step for simulation. Default is 0.1.
        - lower_bound (float): A float representing the lower bound limit for the neuron potential. Default is None.
        - sum_input (bool): A flag indicating whether to sum the input or not. Default is False.
        - trace_scale (Union[float, torch.Tensor]): A float or tensor representing the scaling factor for spike trace. Default is 1.
        - is_inhibitory (bool): A flag indicating whether neuron is inhibitory. Default is False.
        - R (Union[float, torch.Tensor]): Resistance of neuron. (default: 20.0)
        - learning (bool): A flag indicating whether learning is enabled or disabled. Default is True.
        - **kwargs: Additional keyword arguments.
        
        Returns:
        - None.
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float)) # Rest potential
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float)) # Tau_s
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float)) # Reset potential
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float)) # Spiking Threshold
        self.register_buffer("a0", torch.tensor(a0, dtype=torch.float))
        self.register_buffer("critical_pot", torch.tensor(critical_pot, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length)) # Refractor length
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.reset_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        # Compute new potential of neuron
        self.v += (self.refrac_count <= 0).float() * (self.R * x + self.a0 * (self.v - self.rest_pot) * (self.v - self.critical_pot)) / self.tau_s

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check if neuron is spiking or not
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()
        

    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.reset_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

class AELIFPopulation(NeuralPopulation):
    """
    Layer of Adaptive Exponential Leaky Integrate and Fire neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        tau_w: Union[float, torch.Tensor] = 20.,
        threshold: Union[float, torch.Tensor] = -52.,
        theta_rh: Union[float, torch.Tensor] = -60.,
        delta_T: Union[float, torch.Tensor] = 1.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5.,
        dt: float = 0.1,
        a0: Union[float, torch.Tensor] = 1.,
        b: Union[float, torch.Tensor] = 2.,
        R: Union[float, torch.Tensor] = 4.0,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Constructor method for a adaptive exponential leaky integrate and fire neural population.

        Parameters:
        - n (Optional[int]): An optional integer for the number of neurons. Default is None.
        - shape (Optional[Iterable[int]]): An optional iterable representing the shape of the neuron model. Default is None.
        - spike_trace (bool): A flag indicating whether to use spike trace or not. Default is True.
        - additive_spike_trace (bool): A flag indicating whether to add spike trace or replace it. Default is False.
        - tau_s (Union[float, torch.Tensor]): A float or tensor representing tau_s value for neuron model. Default is 10.
        - tau_w (Union[float, torch.Tensor]): A float or tensor representing tau_w value for neuron model. Default is 4.
        - threshold (Union[float, torch.Tensor]): A float or tensor representing spike threshold value for neuron model. Default is -52.
        - theta_rh (Union[float, torch.Tensor]): A float or tensor representing the resting potential value for neuron model. Default is -60.
        - delta_T (Union[float, torch.Tensor]): A float or tensor representing sharpness of the neuron's voltage threshold. Default is 1.
        - rest_pot (Union[float, torch.Tensor]): A float or tensor representing the resting potential value for neuron model. Default is -62.
        - reset_pot (Union[float, torch.Tensor]): A float or tensor representing the reset potential value for neuron model. Default is -62.
        - refrac_length (Union[float, torch.Tensor]): A float or tensor representing the refractory period length for neuron model. Default is 5.
        - dt (float): A float representing the time step for simulation. Default is 0.1.
        - a0 (Union[float, torch.Tensor]): A float or tensor representing the target firing rate for the neuron. Default is 1.
        - b (Union[float, torch.Tensor]): A float or tensor representing the adaptation variable. Default is 2.
        - R (Union[float, torch.Tensor]): A float or tensor representing the resistance of the neuron. Default is 0.001.
        - lower_bound (float): A float representing the lower bound limit for the neuron potential. Default is None.
        - sum_input (bool): A flag indicating whether to sum the input or not. Default is False.
        - trace_scale (Union[float, torch.Tensor]): A float or tensor representing the scaling factor for spike trace. Default is 1.
        - is_inhibitory (bool): A flag indicating whether neuron is inhibitory. Default is False.
        - learning (bool): A flag indicating whether learning is enabled or disabled. Default is True.
        - **kwargs: Additional keyword arguments.

        Returns:
        - None.
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float)) # Rest potential
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float)) # Tau_s
        self.register_buffer("tau_w", torch.tensor(tau_w, dtype=torch.float)) # Tau_w
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float)) # Reset potential
        self.register_buffer("theta_rh", torch.tensor(theta_rh, dtype=torch.float)) # Theta_rh potential
        self.register_buffer("delta_T", torch.tensor(delta_T, dtype=torch.float)) # Delta_T : sharpness
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float)) # Spiking Threshold
        self.register_buffer("refrac_length", torch.tensor(refrac_length)) # Refractor length
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("w", torch.FloatTensor()) # Adaptation variable
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.reset_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.w = torch.zeros_like(self.v, device=self.w.device)
        self.register_buffer("a0", torch.tensor(a0, dtype=torch.float)) # a_0
        self.register_buffer("b", torch.tensor(b, dtype=torch.float)) # b
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        # Compute new potential of neuron
        self.v += ((x + self.rest_pot - self.v + self.delta_T * torch.exp((self.v - self.theta_rh)/ self.delta_T) - self.R * self.w) / self.tau_s * self.dt) * (self.refrac_count <= 0).float()
        self.w += ((self.a0 * (self.v - self.rest_pot) - self.w + self.b * self.tau_w * (self.s.float())) * self.dt /self.tau_w) * (self.refrac_count <= 0).float()

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check if neuron is spiking or not
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()

    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.reset_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

class AQLIFPopulation(NeuralPopulation):
    """
    Layer of Adaptive Quadratic Leaky Integrate and Fire neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 20.,
        tau_w: Union[float, torch.Tensor] = 4.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5.,
        dt: float = 0.1,
        a0: Union[float, torch.Tensor] = 1.,
        b: Union[float, torch.Tensor] = 2.,
        R: Union[float, torch.Tensor] = 1.,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        critical_pot: Union[float, torch.Tensor] = -54.,
        tau_v: Union[float, torch.Tensor] = 4.0,
        **kwargs
    ) -> None:
        """
        Constructor method for a Izhikevich neuron

        Parameters:
        - n (Optional[int]): An optional integer for the number of neurons. Default is None.
        - shape (Optional[Iterable[int]]): An optional iterable representing the shape of the neuron model. Default is None.
        - spike_trace (bool): A flag indicating whether to use spike trace or not. Default is True.
        - additive_spike_trace (bool): A flag indicating whether to add spike trace or replace it. Default is False.
        - tau_s (Union[float, torch.Tensor]): A float or tensor representing tau_s value for neuron model. Default is 10.
        - tau_w (Union[float, torch.Tensor]): A float or tensor representing tau_w value for neuron model. Default is 4.
        - threshold (Union[float, torch.Tensor]): A float or tensor representing spike threshold value for neuron model. Default is -52.
        - rest_pot (Union[float, torch.Tensor]): A float or tensor representing the resting potential value for neuron model. Default is -62.
        - reset_pot (Union[float, torch.Tensor]): A float or tensor representing the reset potential value for neuron model. Default is -62.
        - refrac_length (Union[float, torch.Tensor]): A float or tensor representing the refractory period length for neuron model. Default is 5.
        - dt (float): A float representing the time step for simulation. Default is 0.1.
        - a0 (Union[float, torch.Tensor]): A float or tensor representing the target firing rate for the neuron. Default is 1.
        - b (Union[float, torch.Tensor]): A float or tensor representing the adaptation variable. Default is 2.
        - R (Union[float, torch.Tensor]): A float or tensor representing the resistance of the neuron. Default is 0.001.
        - lower_bound (float): A float representing the lower bound limit for the neuron potential. Default is None.
        - sum_input (bool): A flag indicating whether to sum the input or not. Default is False.
        - trace_scale (Union[float, torch.Tensor]): A float or tensor representing the scaling factor for spike trace. Default is 1.
        - is_inhibitory (bool): A flag indicating whether neuron is inhibitory. Default is False.
        - learning (bool): A flag indicating whether learning is enabled or disabled. Default is True.
        - critical_pot (Union[float, torch.Tensor]): A float or tensor representing the critical potential value for neuron model. Default is 0.8.
        - tau_v (Union[float, torch.Tensor]): A float or tensor representing tau_v value for neuron model. Default is 4.0.
        - **kwargs: Additional keyword arguments.

        Returns:
        - None.
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R=R
        )

        self.register_buffer("critical_pot", torch.tensor(critical_pot, dtype=torch.float))
        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float)) # Rest potential
        self.register_buffer("tau_v", torch.tensor(tau_v, dtype=torch.float)) # Tau_v
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float)) # Tau_s
        self.register_buffer("tau_w", torch.tensor(tau_w, dtype=torch.float)) # Tau_w
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float)) # Reset potential
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float)) # Spiking Threshold
        self.register_buffer("refrac_length", torch.tensor(refrac_length)) # Refractor length
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("w", torch.FloatTensor()) # Adaptation variable
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.reset_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.w = torch.zeros_like(self.v, device=self.w.device)
        self.register_buffer("a0", torch.tensor(a0, dtype=torch.float)) # a_0
        self.register_buffer("b", torch.tensor(b, dtype=torch.float)) # b
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        # Compute new potential of neuron
        self.v += (((self.v - self.rest_pot) * (self.v  - self.critical_pot) - self.R * self.w + self.R * x) * self.dt / self.tau_v) * (self.refrac_count <= 0).float()
        self.w += ((self.a0 * (self.v - self.rest_pot) - self.w + self.b * self.tau_w * (self.s.float())) * self.dt / self.tau_w) * (self.refrac_count <= 0).float()

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check if neuron is spiking or not
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()
        

    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.reset_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

class CLIFPopulation(NeuralPopulation):
    """
    Layer of Cumulative Leaky Integrate and Fire neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -65.,
        reset_pot: Union[float, torch.Tensor] = -65.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        tc_decay: Union[float,torch.Tensor] = 100.0,
        tc_i_decay: Union[float,torch.Tensor] = 2.0,
        lower_bound: float = None,
        sum_input: bool = False,
        R: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Initializes the parameters of the cumulative leaky integrate and fire neuron model.

        Args:
            n (Optional[int]): An optional integer specifying the number of neurons to create.
            shape (Optional[Iterable[int]]): An optional iterable specifying the shape of the neuron layer. If this argument is passed, `n` is ignored.
            spike_trace (bool): A boolean indicating whether to use spike tracing.
            additive_spike_trace (bool): A boolean indicating whether to use additive spike tracing.
            tau_s (Union[float, torch.Tensor]): A float or tensor specifying the time constant for spike tracing.
            threshold (Union[float, torch.Tensor]): A float or tensor specifying the firing threshold of the neuron.
            rest_pot (Union[float, torch.Tensor]): A float or tensor specifying the resting potential of the neuron.
            reset_pot (Union[float, torch.Tensor]): A float or tensor specifying the reset potential of the neuron.
            refrac_length (Union[float, torch.Tensor]): A float or tensor specifying the refractory period length of the neuron.
            dt (float): A float specifying the time step of the simulation.
            tc_decay (Union[float,torch.Tensor]): A float or tensor specifying the voltage decay time constant of the neuron.
            tc_i_decay (Union[float,torch.Tensor]): A float or tensor specifying the current decay time constant of the neuron.
            lower_bound (float): A float specifying the lower bound for the neuron potential.
            sum_input (bool): A boolean indicating whether to sum all input instead of taking the maximum.
            trace_scale (Union[float, torch.Tensor]): A float or tensor specifying the scaling factor for the spike trace.
            R (Union[float, torch.Tensor]): A float or tensor representing the resistance of the neuron. Default is 20.0 .
            is_inhibitory (bool): A boolean indicating whether the neuron is inhibitory.
            learning (bool): A boolean indicating whether the neuron should learn.

        Returns:
            None
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            sum_input=sum_input,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float))
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float))
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("i", torch.FloatTensor()) # Neuron's current'
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.v = self.rest_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.i = torch.zeros_like(self.v, device=self.i.device)
        self.register_buffer("tc_decay", torch.tensor(tc_decay, dtype=torch.float)) # Time constant neuron voltage decay
        self.register_buffer("tc_i_decay", torch.tensor(tc_i_decay, dtype=torch.float)) # Time constant input current decay
        self.register_buffer("decay", torch.empty_like(self.tc_decay)) # Main decay which applies to neuron voltage
        self.register_buffer("i_decay", torch.empty_like(self.tc_i_decay)) # Main current decay which applies to input current
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound


    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)
        

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """

        self.i *= self.i_decay  # Decay current
        self.i += x  # Add new input current to recent ones
        self.v += (( - (self.v - self.rest_pot) + self.R * self.i) * self.dt / self.tau_s) * (self.refrac_count <= 0).float()

    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check for spiking neuron
        self.s = self.v >= self.pot_threshold


    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.rest_pot)
        

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()
        self.i_decay = torch.exp(-self.dt / self.tc_i_decay) # Neuron current decay
        

    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset


class SRM0Node(NeuralPopulation):
    """
    Layer of Simplified Spike Response Model.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        tau_decay: Union[float, torch.Tensor] = 10.0,
        eps_0: Union[float, torch.Tensor] = 1.0,
        rho_0: Union[float, torch.Tensor] = 1.0,
        d_thresh: Union[float, torch.Tensor] = 5.0,
        dt: float = 0.1,
        R: Union[float, torch.Tensor] = 20.0,
        lower_bound: float = None,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Initializes the parameters of the simplified spike response neuron model.

        Args:
            n (Optional[int]): An optional integer specifying the number of neurons to create.
            shape (Optional[Iterable[int]]): An optional iterable specifying the shape of the neuron layer. If this argument is passed, `n` is ignored.
            spike_trace (bool): A boolean indicating whether to use spike tracing.
            additive_spike_trace (bool): A boolean indicating whether to use additive spike tracing.
            tc_trace (Union[float, torch.Tensor]): A float or tensor specifying the time constant for trace decay.
            trace_scale (Union[float, torch.Tensor]): A float or tensor specifying the scaling factor for the spike trace.
            sum_input (bool): A boolean indicating whether to sum all input instead of taking the maximum.
            tau_s (Union[float, torch.Tensor]): A float or tensor specifying the time constant for spike tracing.
            threshold (Union[float, torch.Tensor]): A float or tensor specifying the firing threshold of the neuron.
            rest_pot (Union[float, torch.Tensor]): A float or tensor specifying the resting potential of the neuron.
            reset_pot (Union[float, torch.Tensor]): A float or tensor specifying the reset potential of the neuron.
            refrac_length (Union[float, torch.Tensor]): A float or tensor specifying the refractory period length of the neuron.
            tau_decay (Union[float, torch.Tensor]): A float or tensor specifying the threshold decay time constant of the neuron.
            eps_0 (Union[float, torch.Tensor]): A float or tensor specifying the initial value of epsilon.
            rho_0 (Union[float, torch.Tensor]): A float or tensor specifying the initial value of rho.
            d_thresh (Union[float, torch.Tensor]): A float or tensor specifying the distance threshold.
            dt (float): A float specifying the time step of the simulation.
            R (Union[float, torch.Tensor]) : A float or tensor indicating the resistance of neuron.
            lower_bound (float): A float specifying the lower bound for the neuron potential.
            is_inhibitory (bool): A boolean indicating whether the neuron is inhibitory.
            learning (bool): A boolean indicating whether the neuron should learn.

        Returns:
            None
        """
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            sum_input=sum_input,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            dt=dt,
            R=R
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float))
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float))
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("tau_decay", torch.tensor(tau_decay))
        self.register_buffer("decay", torch.tensor(tau_decay))
        self.register_buffer("eps_0", torch.tensor(eps_0))
        self.register_buffer("rho_0", torch.tensor(rho_0))
        self.register_buffer("d_thresh", torch.tensor(d_thresh))
        self.register_buffer("v", torch.FloatTensor())
        self.register_buffer("refrac_count", torch.FloatTensor())
        self.v = self.rest_pot * torch.ones(*self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.compute_decay() # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound


    def forward(self, x: torch.Tensor) -> None:
        """
        Computes the forward pass of the layer.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        # Check lower bound condition for neuron.
        if self.lower_bound is not None:
            self.v.masked_fill_(self.lower_bound > self.v, self.lower_bound)
            
        super().forward(x)
        

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Computes the new potential of the neuron based on the given input tensor.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to the layer
            
        Returns:
        -------
        None
        """
                
        self.v = self.decay * (self.v - self.rest_pot) + self.rest_pot
        self.v += (self.refrac_count <= 0).float() * self.eps_0 * x * self.R
        self.rho = self.rho_0 * torch.exp((self.v - self.pot_threshold) / self.pot_threshold)
        self.s_prob = 1.0 - torch.exp(-self.rho * self.dt)
        


    def compute_spike(self) -> None:
        """
        Computes if the neuron has spiked or not based on its potential and the threshold.
        
        Returns:
        -------
        None
        """
        # Check for spiking neuron
        self.s = torch.rand_like(self.s_prob) < self.s_prob


    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Applies refractory and reset conditions to the neuron.
        
        Returns:
        -------
        None
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)
        

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Computes the decay rate of the neuron.
        
        Returns:
        -------
        None
        """
        super().compute_decay()
        self.decay = torch.exp(-self.dt / self.tau_decay)  # Neuron voltage decay (per timestep).



    def reset_state_variables(self) -> None:
        """
        Resets the state of the neuron.
        
        Returns:
        -------
        None
        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset


