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


class RepeatEncoder(AbstractEncoder):
    """
    Reapeat coding.

    Implement repeat coding.
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
        
        time = int(self.time / self.dt)
        return data.repeat([time, 1])


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.

    Implement Time-to-First-Spike coding.
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
        """
        TODO.

        Add other attributes if needed and fill the body accordingly.
        """

    def __call__(self, data: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation for coding the data. Return resulting tensor.
        """
        pass


class PositionEncoder(AbstractEncoder):
    """
    Position coding.

    Implement Position coding.
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
        """
        TODO.

        Add other attributes if needed and fill the body accordingly.
        """

    def __call__(self, data: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation for coding the data. Return resulting tensor.
        """
        pass


class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        """
        Constructor.

        :param time: Length of time for which to generate spike trains.
        :param dt: Time resolution of the spike trains.
        :param device: Device to use for computations.
        """
        super().__init__(
            time=time,
            dt=dt,
            device=device
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        
        """
        :param data: Tensor of shape ``[n_1, ..., n_k]``.
        :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
        """

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
