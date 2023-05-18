"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    You will define the time duration into which you want to encode the data \
    as `time` and define the time resolution as `dt`. All computations will be \
    performed on the CPU by default. To handle computation on both GPU and CPU, \
    make sure to set the device as defined in `device` attribute to all your \
    tensors. You can add any other attributes to the child classes, if needed.

    The computation procedure should be implemented in the `__call__` method. \
    Data will be passed to this method as a tensor for further computations. You \
    might need to define more parameters for this method. The `__call__`  should return \
    the tensor of spikes with the shape (time_steps, \*population.shape).

    Arguments
    ---------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.time = time
        self.dt = dt
        self.device = device

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> None:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            It should return the encoded tensor.

        """
        pass


class NullEncoder(AbstractEncoder):
    
    """
    Pass through of the datum that was input.
    """

    def __init__(self):
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data
    

class SingleEncoder(AbstractEncoder):

    """
    Generates timing based single-spike encoding. Spike occurs earlier if the \
    intensity of the input feature is higher. Features whose value is lower than \
    the threshold remain silent.

    Parameters:
    ----------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".
    sparsity: float, Optional  
        Sparsity of the input representation. 0 for no spikes and 1 \
        for all spikes. The default is 0.5.
    
    Returns:
    -------    
    Tensor of shape ``[time, n_1, ..., n_k]``.

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        

    def __call__(self, data: torch.Tensor, sparsity = 0.5) -> torch.Tensor:
        time_step = int(self.time / self.dt)
        quantile = torch.quantile(data, 1 - sparsity)
        spikes = torch.zeros([time_step, *data.shape], device=self.device)
        spikes[0] = torch.where(data > quantile, torch.ones(data.shape), torch.zeros(data.shape))
        return torch.Tensor(spikes)


class RepeatEncoder(AbstractEncoder):
    
    """
    Repeats a tensor along a new dimension in the 0th position for \
    ``int(time / dt)`` timesteps.

    Parameters:
    ----------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".

    Returns:
    -------    
    Tensor of shape ``[time, n_1, ..., n_k]`` of repeated data along the 0-th dimension.

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        
        time = int(self.time / self.dt)
        return data.repeat([time, 1])

    
class BernoulliEncoder(AbstractEncoder):
    
    """
    Generates Bernoulli-distributed spike trains based on input intensity. Inputs must \
    be non-negative. Spikes correspond to successful Bernoulli trials, with success \
    probability equal to (normalized in [0, 1]) input value.

    Parameters:
    ----------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".
    max_prob : float, Optional
        Maximum probability of spike per Bernoulli trial. The default is 1.0. 

    Returns:
    -------
    Tensor of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.
    
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )


    def __call__(self, data: torch.Tensor, max_prob = 1.0) -> None:
        time = int(self.time / self.dt)
        data = data.flatten().to(self.device)
        data /= data.max()
        spikes = torch.bernoulli(max_prob * data.repeat([time, 1]))
        spikes = spikes.view(time, *data.shape)
        return spikes


class PoissonEncoder(AbstractEncoder):
    
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be \
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for \
    non-negative data incremented by one to avoid zero intervals while maintaining ISI \
    distributions.

    Parameters:
    ----------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".
    
    Returns:
    -------
    Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        shape, size = data.shape, data.numel()
        data = data.flatten().to(self.device)
        time = int(self.time / self.dt)

        rate = torch.zeros(size, device=self.device)
        rate[data != 0] = 1 / data[data != 0] * (1000 / self.dt)

        dist = torch.distributions.Poisson(rate=rate, validate_args=False)
        intervals = dist.sample(sample_shape=torch.Size([time + 1]))
        intervals[:, data != 0] += (intervals[:, data != 0] == 0).float()

        times = torch.cumsum(intervals, dim=0).long()
        times[times >= time + 1] = 0

        spikes = torch.zeros(time + 1, size, device=self.device).byte()
        spikes[times, torch.arange(size)] = 1
        spikes = spikes[1:]

        return spikes.view(time, *shape)
    

class RankOrderEncoder(AbstractEncoder):
     
    """
    Encodes data via a rank order coding-like representation. One spike per neuron,
    temporally ordered by decreasing intensity. Inputs must be non-negative.

    Parameters:
    ----------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".
    
    Returns:
    -------
    Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        
    def __call__(self, data: torch.Tensor) -> None:
        shape, size = data.shape, data.numel()
        data = data.flatten().to(self.device)
        time = int(self.time / self.dt)

        times = torch.zeros(size)
        times[data != 0] = 1 / data[data != 0]
        times *= time / times.max() 
        times = torch.ceil(times).long()

        spikes = torch.zeros(time, size, device=self.device).byte()
        for i in range(size):
            if 0 < times[i] <= time:
                spikes[times[i] - 1, i] = 1

        return spikes.reshape(time, *shape)


class Timetofirstspike(AbstractEncoder):
     
    """
    Parameters:
    ----------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".
    
    Returns:
    -------
    Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        
    def __call__(self, data: torch.Tensor) -> None:
        shape, size = data.shape, data.numel()
        data = data.flatten().to(self.device)
        time = int(self.time / self.dt)

        times = torch.zeros(size)
        times[data != 0] = data[data != 0] / data.max()
        times = torch.floor(times).long()

        spikes = torch.zeros(time, size, device=self.device).byte()
        for i in range(size):
            if 0 < times[i] <= time:
                spikes[times[i] - 1, i] = 1

        return spikes.reshape(time, *shape)
