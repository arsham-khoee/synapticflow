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
        approx: bool = False,
        **kwargs
    ) -> None:
        """
        Constructor.

        :param time: Length of time for which to generate spike trains.
        :param dt: Time resolution of the spike trains.
        :param device: Device to use for computations.
        :param approx: Flag indicating whether to use an approximation method for generating Poisson spikes.
        """
        super().__init__(
            time=time,
            dt=dt,
            device=device
        )
        self.approx = approx

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        
        min_rate = 0.0
        max_rate = 80.0
        num_levels = 16
        level_size = 1 / num_levels
        rate_range = max_rate - min_rate
        rates = np.linspace(min_rate, max_rate, num_levels)

        levels = torch.floor(data / level_size).clamp(max=num_levels - 1)

        num_neurons = num_levels
        num_steps = int(self.time / self.dt)
        spikes = torch.zeros((num_neurons, num_steps), device=self.device)
        for i in range(num_neurons):
            rate = rates[i]
            for j in range(num_steps):
                if self.approx:
                    prob = rate * self.dt
                    spikes[i, j] = torch.bernoulli(prob.unsqueeze(0)).squeeze()
                else:
                    rate_tensor = torch.tensor(rate, device=self.device)
                    exp_val = torch.exp(-rate_tensor * self.dt)
                    spikes[i, j] = torch.bernoulli(1 - exp_val.unsqueeze(0)).squeeze()

        indices = levels.long()
        spikes = spikes[indices, :]

        return spikes