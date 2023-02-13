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
        **kwargs
    ) -> None:
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
            # You can use `torch.Tensor()` instead of `torch.zeros(*shape)` if `reset_state_variables`
            # is intended to be called before every simulation.
            self.register_buffer("traces", torch.zeros(*self.shape))
            self.register_buffer("tau_s", torch.tensor(tau_s))

            if self.additive_spike_trace:
                self.register_buffer("trace_scale", torch.tensor(trace_scale))

            self.register_buffer("trace_decay", torch.empty_like(self.tau_s))

        self.is_inhibitory = is_inhibitory
        self.learning = learning

        # You can use `torch.Tensor()` instead of `torch.zeros(*shape, dtype=torch.bool)` if \
        # `reset_state_variables` is intended to be called before every simulation.
        self.register_buffer("s", torch.ByteTensor())
        
        # Add summed property to sum all given inputs
        if self.sum_input:
            self.register_buffer("summed", torch.FloatTensor()) # Inputs summation
        
        self.dt = None

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

    def set_batch_size(self, batch_size) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        
        Parameters
        ----------
        batch_size : int,
            Mini-batch size
            
        Returns
        -------
        None
        """
        self.batch_size = batch_size
        self.s = torch.zeros(batch_size, *self.shape, device=self.s.device, dtype=torch.bool)

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
    Layer of Integrate and Fire neurons.

    Implement IF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = False,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Arguments
        ---------
        n : int, Optional
            Number of neurons in the population.
        shape : Iterable of int
            Define the topology of neurons in the population.
        spike_trace : bool, Optional
            Specify whether to record spike traces. The default is False.
        additive_spike_trace : bool, Optional
            Specify whether to record spike traces additively. The default is False.
        tau_s : float or torch.Tensor, Optional
            Time constant of spike trace decay. The default is 10.0.
        threshold : float or torch.Tensor, Optional
            Threshold potential to spike. The default is -52.0v.
        rest_pot : float or torch.Tensor, Optional
            Rest potential for spike. The default is -62.0v.
        refrac_length : float or torch.Tensor, Optional
            Neuron refractor interval length. The default is 5 time steps.
        dt : float, Optional
            Length of each time step.
        lower_bound : float, Optional
            Lower bound for neuron potential. The default is None.
        trace_scale : float or torch.Tensor, Optional
            The scaling factor of spike traces. The default is 1.0.
        is_inhibitory : bool, Optional
            Whether the neurons are inhibitory or excitatory. The default is False.
        learning : bool, Optional
            Define the training mode. The default is True.
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
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float))
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.compute_decay(dt) # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound
        

    def forward(self, x: torch.Tensor) -> None:
        """
        TODO.

        1. Make use of other methods to fill the body. This is the main method\
           responsible for one step of neuron simulation.
        2. You might need to call the method from parent class.
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
        Compute new potential of neuron by given input tensor x and refrac_count
        """
        # Compute new potential
        self.v += (self.refrac_count <= 0).float() * x

    def compute_spike(self) -> None:
        """
        Compute spike condition and make changes directly on spike tensor
        """
        # Check for spiking neuron
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        In this function, three things will be done:
            1 - decrease refrac_count by time step size
            2 - Set refrac_count to refrac_length if spiking is occurred
            3 - Set neuron potential to rest_pot if spiking is occurred
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.rest_pot)
        

    @abstractmethod
    def compute_decay(self, dt: float) -> None:
        """
        Set the decays.

        Parameters
        ----------
        dt : float,
            Length of time steps.

        Returns
        -------
        None

        """
        self.dt = dt
        super().compute_decay()


    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        
        Parameters
        ----------
        batch_size: int,
            Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest_pot * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)

class LIFPopulation(NeuralPopulation):
    """
    Layer of Leaky Integrate and Fire neurons.

    Implement LIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = False,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        rest_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        tau_decay: Union[float, torch.Tensor] = 100.0,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Arguments
        ---------
        n : int, Optional
            Number of neurons in the population.
        shape : Iterable of int
            Define the topology of neurons in the population.
        spike_trace : bool, Optional
            Specify whether to record spike traces. The default is False.
        additive_spike_trace : bool, Optional
            Specify whether to record spike traces additively. The default is False.
        tau_s : float or torch.Tensor, Optional
            Time constant of spike trace decay. The default is 10.0.
        threshold : float or torch.Tensor, Optional
            Threshold potential to spike. The default is -52.0v.
        rest_pot : float or torch.Tensor, Optional
            Rest potential for spike. The default is -62.0v.
        refrac_length : float or torch.Tensor, Optional
            Neuron refractor interval length. The default is 5 time steps.
        dt : float, Optional
            Length of each time step.
        lower_bound : float, Optional
            Lower bound for neuron potential. The default is None.
        trace_scale : float or torch.Tensor, Optional
            The scaling factor of spike traces. The default is 1.0.
        is_inhibitory : bool, Optional
            Whether the neurons are inhibitory or excitatory. The default is False.
        tau_decay: 
            Time constant of neuron voltage decay. The default is 100.0.
        learning : bool, Optional
            Define the training mode. The default is True.
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
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float))
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.register_buffer("tau_decay", torch.tensor(tau_decay, dtype=torch.float))  # Time constant of neuron voltage decay.
        self.register_buffer("decay", torch.zeros(*self.shape))  # Set in compute_decays.
        self.compute_decay(dt) # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound


    def forward(self, x: torch.Tensor) -> None:
        """
        TODO.

        1. Make use of other methods to fill the body. This is the main method\
           responsible for one step of neuron simulation.
        2. You might need to call the method from parent class.
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
        Compute new potential of neuron by given input tensor x and refrac_count
        """

        # Compute new potential with decay voltages.
        self.v = self.decay * (self.v - self.rest_pot) + self.rest_pot

        # Integrate inputs.
        x.masked_fill_(self.refrac_count > 0, 0.0)

        # interlaced
        self.v += x 


    def compute_spike(self) -> None:
        """
        Compute spike condition and make changes directly on spike tensor
        """
        # Check for spiking neuron
        self.s = self.v >= self.pot_threshold


    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        In this function, three things will be done:
            1 - decrease refrac_count by time step size
            2 - Set refrac_count to refrac_length if spiking is occurred
            3 - Set neuron potential to rest_pot if spiking is occurred
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.rest_pot)
        

    @abstractmethod
    def compute_decay(self, dt: float) -> None:
        """
        Set the decays.

        Parameters
        ----------
        dt : float,
            Length of time steps.

        Returns
        -------
        None

        """
        self.dt = dt
        super().compute_decay()
        self.decay = torch.exp(-self.dt / self.tau_decay)  # Neuron voltage decay (per timestep).



    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        
        Parameters
        ----------
        batch_size: int,
            Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest_pot * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class BoostedLIFPopulation(NeuralPopulation):
    # Same as LIFNodes, faster: no rest, no reset, no lbound
    """
    Layer of Boosted Leaky Integrate and Fire neurons.

    Implement LIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = False,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.0,
        threshold: Union[float, torch.Tensor] = 40.0,
        refrac_length: Union[float, torch.Tensor] = 5.0,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        is_inhibitory: bool = False,
        tau_decay: Union[float, torch.Tensor] = 1,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Arguments
        ---------
        n : int, Optional
            Number of neurons in the population.
        shape : Iterable of int
            Define the topology of neurons in the population.
        spike_trace : bool, Optional
            Specify whether to record spike traces. The default is False.
        additive_spike_trace : bool, Optional
            Specify whether to record spike traces additively. The default is False.
        tau_s : float or torch.Tensor, Optional
            Time constant of spike trace decay. The default is 10.0.
        threshold : float or torch.Tensor, Optional
            Threshold potential to spike. The default is 40.0v.
        refrac_length : float or torch.Tensor, Optional
            Neuron refractor interval length. The default is 5 time steps.
        dt : float, Optional
            Length of each time step.
        lower_bound : float, Optional
            Lower bound for neuron potential. The default is None.
        trace_scale : float or torch.Tensor, Optional
            The scaling factor of spike traces. The default is 1.0.
        is_inhibitory : bool, Optional
            Whether the neurons are inhibitory or excitatory. The default is False.
        tau_decay: 
            Time constant of neuron voltage decay. The default is 1.0.
        learning : bool, Optional
            Define the training mode. The default is True.
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
        )

        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.register_buffer("tau_decay", torch.tensor(tau_decay, dtype=torch.float))  # Time constant of neuron voltage decay.
        self.register_buffer("decay", torch.zeros(*self.shape))  # Set in compute_decays.
        self.compute_decay(dt) # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound


    def forward(self, x: torch.Tensor) -> None:
        """
        TODO.

        1. Make use of other methods to fill the body. This is the main method\
           responsible for one step of neuron simulation.
        2. You might need to call the method from parent class.
        """
        self.compute_potential(x) # Compute new potential
        
        self.compute_spike() # Check if neuron is spiking
        
        self.refractory_and_reset() # Applies refractory and reset conditions
        
        super().forward(x)
        

    def compute_potential(self, x: torch.Tensor) -> None:
        """
        Compute new potential of neuron by given input tensor x and refrac_count
        """
                
        # Compute new potential with decay voltages.
        self.v *= self.decay

        # Integrate inputs.
        if x is not None:
            x.masked_fill_(self.refrac_count > 0, 0.0)

        # interlaced
        if x is not None:
            self.v += x 


    def compute_spike(self) -> None:
        """
        Compute spike condition and make changes directly on spike tensor
        """
        # Check for spiking neuron
        self.s = self.v >= self.pot_threshold


    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        In this function, three things will be done:
            1 - decrease refrac_count by time step size
            2 - Set refrac_count to refrac_length if spiking is occurred
            3 - Set neuron potential to rest_pot if spiking is occurred
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, 0)
        

    @abstractmethod
    def compute_decay(self, dt: float) -> None:
        """
        Set the decays.

        Parameters
        ----------
        dt : float,
            Length of time steps.

        Returns
        -------
        None

        """
        self.dt = dt
        super().compute_decay()
        self.decay = torch.exp(-self.dt / self.tau_decay)  # Neuron voltage decay (per timestep).



    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()
        self.v.fill_(0) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        
        Parameters
        ----------
        batch_size: int,
            Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)



class ELIFPopulation(NeuralPopulation):
    """
    Layer of Exponential Leaky Integrate and Fire neurons.

    Implement ELIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.

    Note: You can use LIFPopulation as parent class as well.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = False,
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
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
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
        self.compute_decay(dt) # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> None:
        """
        Simulate one step of a neuron
        Parameters
        ----------
        x : Tensor,
            Input current.
            
        Returns
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
        Compute new potential of neuron by given input tensor x and refrac_count
        """
        # Compute new potential of neuron
        if (self.refrac_count <= 0):
            self.v = self.v + (x + self.rest_pot - self.v + self.delta_T * torch.exp((self.v - self.theta_rh)/ self.delta_T)) / self.tau_s * self.dt

    def compute_spike(self) -> None:
        """
        Compute spike condition and make changes directly on spike tensor
        """
        # Check if neuron is spiking or not
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        In this function, three things will be done:
            1 - decrease refrac_count by time step size
            2 - Set refrac_count to refrac_length if spiking is occurred
            3 - Set neuron potential to rest_pot if spiking is occurred
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)

    @abstractmethod
    def compute_decay(self, dt: float) -> None:
        """
        Set the decays.

        Parameters
        ----------
        dt : float,
            Length of time steps.

        Returns
        -------
        None

        """
        self.dt = dt
        super().compute_decay()
        
    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        
        Parameters
        ----------
        batch_size: int,
            Mini-batch size.
        
        Returns
        -------
        None
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.reset_pot * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()
        self.v.fill_(self.reset_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

class QLIFPopulation(NeuralPopulation):
    """
    Layer of Exponential Leaky Integrate and Fire neurons.

    Implement ELIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.

    Note: You can use LIFPopulation as parent class as well.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = False,
        additive_spike_trace: bool = False,
        tau_s: Union[float, torch.Tensor] = 10.,
        threshold: Union[float, torch.Tensor] = -52.,
        theta_rh: Union[float, torch.Tensor] = -60.,
        delta_T: Union[float, torch.Tensor] = 1.,
        a0: Union[float, torch.Tensor] = 1,
        critical_pot: Union[float, torch.Tensor] = 0.8,
        rest_pot: Union[float, torch.Tensor] = -62.,
        reset_pot: Union[float, torch.Tensor] = -62.,
        refrac_length: Union[float, torch.Tensor] = 5,
        dt: float = 0.1,
        lower_bound: float = None,
        sum_input: bool = False,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float)) # Rest potential
        self.register_buffer("tau_s", torch.tensor(tau_s, dtype=torch.float)) # Tau_s
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float)) # Reset potential
        self.register_buffer("theta_rh", torch.tensor(theta_rh, dtype=torch.float)) # Theta_rh potential
        self.register_buffer("delta_T", torch.tensor(delta_T, dtype=torch.float)) # Delta_T : sharpness
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float)) # Spiking Threshold
        self.register_buffer("a0", torch.tensor(a0, dtype=torch.float))
        self.register_buffer("critical_pot", torch.tensor(critical_pot, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length)) # Refractor length
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.compute_decay(dt) # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> None:
        """
        Simulate one step of a neuron
        Parameters
        ----------
        x : Tensor,
            Input current.
            
        Returns
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
        Compute new potential of neuron by given input tensor x and refrac_count
        """
        # Compute new potential of neuron
        self.v += (self.refrac_count <= 0).float() * (x + self.a0 * (self.v - self.rest_pot) * (self.v - self.critical_pot)) / self.tau_s

    def compute_spike(self) -> None:
        """
        Compute spike condition and make changes directly on spike tensor
        """
        # Check if neuron is spiking or not
        self.s = (self.v >= self.pot_threshold)

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        In this function, three things will be done:
            1 - decrease refrac_count by time step size
            2 - Set refrac_count to refrac_length if spiking is occurred
            3 - Set neuron potential to rest_pot if spiking is occurred
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.reset_pot)

    @abstractmethod
    def compute_decay(self, dt: float) -> None:
        """
        Set the decays.

        Parameters
        ----------
        dt : float,
            Length of time steps.

        Returns
        -------
        None

        """
        self.dt = dt
        super().compute_decay()
        
    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        
        Parameters
        ----------
        batch_size: int,
            Mini-batch size.
        
        Returns
        -------
        None
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.reset_pot * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()
        self.v.fill_(self.reset_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

class CLIFPopulation(NeuralPopulation):
    """
    Layer of Leaky Integrate and Fire neurons.

    Implement LIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = False,
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
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        """
        Arguments
        ---------
        n : int, Optional
            Number of neurons in the population.
        shape : Iterable of int
            Define the topology of neurons in the population.
        spike_trace : bool, Optional
            Specify whether to record spike traces. The default is False.
        additive_spike_trace : bool, Optional
            Specify whether to record spike traces additively. The default is False.
        tau_s : float or torch.Tensor, Optional
            Time constant of spike trace decay. The default is 10.0.
        threshold : float or torch.Tensor, Optional
            Threshold potential to spike. The default is -52.0v.
        rest_pot : float or torch.Tensor, Optional
            Rest potential for spike. The default is -65.0v.
        reset_pot: float or torch.Tensor, Optional
            Reset potential for spike. The default is -65.0v.
        refrac_length : float or torch.Tensor, Optional
            Neuron refractor interval length. The default is 5 time steps.
        dt : float, Optional
            Length of each time step.
        lower_bound : float, Optional
            Lower bound for neuron potential. The default is None.
        trace_scale : float or torch.Tensor, Optional
            The scaling factor of spike traces. The default is 1.0.
        is_inhibitory : bool, Optional
            Whether the neurons are inhibitory or excitatory. The default is False.
        tc_decay: 
            Time constant of neuron voltage decay. The default is 100.0.
        learning : bool, Optional
            Define the training mode. The default is True.
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
        )

        self.register_buffer("rest_pot", torch.tensor(rest_pot, dtype=torch.float))
        self.register_buffer("reset_pot", torch.tensor(reset_pot, dtype=torch.float))
        self.register_buffer("pot_threshold", torch.tensor(threshold, dtype=torch.float))
        self.register_buffer("refrac_length", torch.tensor(refrac_length))
        self.register_buffer("v", torch.FloatTensor()) # Neuron's potential
        self.register_buffer("i", torch.FloatTensor()) # Neuron's current'
        self.register_buffer("refrac_count", torch.FloatTensor()) # Refractor counter
        self.register_buffer("tc_decay", torch.tensor(tc_decay, dtype=torch.float)) # Time constant neuron voltage decay
        self.register_buffer("tc_i_decay", torch.tensor(tc_i_decay, dtype=torch.float)) # Time constant input current decay
        self.register_buffer("decay", torch.empty_like(self.tc_decay)) # Main decay which applies to neuron voltage
        self.register_buffer("i_decay", torch.empty_like(self.tc_i_decay)) # Main current decay which applies to input current
        self.compute_decay(dt) # Compute decays and set time steps
        self.reset_state_variables()
        self.lower_bound = lower_bound


    def forward(self, x: torch.Tensor) -> None:
        """
        Simulate one step of neuron 
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
        Compute new potential of neuron by given input tensor x , recent current i and refrac_count
        """
        self.v = self.decay * (self.v - self.rest_pot) + self.rest_pot # Compute voltage with respect to recent potential and rest potential
        self.i *= self.i_decay # Compute input current
        
        self.i += x # Add new input current to recent ones
        self.v += (self.refrac_count <= 0).float() * self.i # Compute voltage if the neuron was not in refractory state


    def compute_spike(self) -> None:
        """
        Compute spike condition and make changes directly on spike tensor
        """
        # Check for spiking neuron
        self.s = self.v >= self.pot_threshold


    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        In this function, three things will be done:
            1 - decrease refrac_count by time step size
            2 - Set refrac_count to refrac_length if spiking is occurred
            3 - Set neuron potential to rest_pot if spiking is occurred
        """
        super().refractory_and_reset()
        
        # Decrease refactor count by time step length
        self.refrac_count -= self.dt
        
        # Set refrac_count equal to refrac_length if spiking is occurred.
        self.refrac_count.masked_fill_(self.s, self.refrac_length)
        
        # Set potential of neuron to rest potential if spiking is occurred.
        self.v.masked_fill_(self.s, self.rest_pot)
        

    @abstractmethod
    def compute_decay(self, dt: float) -> None:
        """
        Set the decays.

        Parameters
        ----------
        dt : float,
            Length of time steps.

        Returns
        -------
        None

        """
        self.dt = dt
        super().compute_decay()
        self.decay = torch.exp(-self.dt / self.tc_decay)  # Neuron voltage decay (per timestep).
        self.i_decay = torch.exp(-self.dt / self.tc_i_decay) # Neuron current decay



    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()
        self.v.fill_(self.rest_pot) # Reset neuron voltages
        self.refrac_count.zero_() # Refractory period reset

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        
        Parameters
        ----------
        batch_size: int,
            Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest_pot * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        self.i = torch.zeros_like(self.v, device=self.i.device)


class AELIFPopulation(NeuralPopulation):
    """
    Layer of Adaptive Exponential Leaky Integrate and Fire neurons.

    Implement adaptive ELIF neural dynamics(Parameters of the model must be\
    modifiable). Follow the template structure of NeuralPopulation class for\
    consistency.

    Note: You can use ELIFPopulation as parent class as well.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        """
        TODO.

        1. Add the required parameters.
        2. Fill the body accordingly.
        """

    def forward(self, x: torch.Tensor) -> None:
        """
        TODO.

        1. Make use of other methods to fill the body. This is the main method\
           responsible for one step of neuron simulation.
        2. You might need to call the method from parent class.
        """
        pass

    def compute_potential(self) -> None:
        """
        TODO.

        Implement the neural dynamics for computing the potential of adaptive\
        ELIF neurons. The method can either make changes to attributes directly\
        or return the result for further use.
        """
        pass

    def compute_spike(self) -> None:
        """
        TODO.

        Implement the spike condition. The method can either make changes to\
        attributes directly or return the result for further use.
        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        TODO.

        Implement the refractory and reset conditions. The method can either\
        make changes to attributes directly or return the computed value for\
        further use.
        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        TODO.

        Implement the dynamics of decays. You might need to call the method from
        parent class.
        """
        pass
