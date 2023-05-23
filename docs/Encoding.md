# Encoders

## Introduction
Encoding input data into spike trains is one of the most important tasks in building Spiking Neural Networks (SNNs). A good encoding scheme can greatly improve the accuracy and efficiency of an SNN, while a poor one can introduce noise and reduce performance. There are several approaches to encoding, including rate coding, temporal coding, and population coding, each with its own advantages and disadvantages. Rate coding involves using the firing rate of neurons to represent the magnitude of the input, while temporal coding represents information through the precise timing of individual spikes. Population coding combines both rate and temporal coding across multiple neurons to represent complex inputs. In this SNN package, we provide various encoding techniques and tools that enable users to efficiently encode their input data into spiking events and achieve optimal performance in their neural networks.

## Table of Contents
- AbstractEncoder
- NullEncoder
- SingleEncoder
- RepeatEncoder
- BernoulliEncoder
- PoissonEncoder
- RankOrderEncoder
- Timetofirstspike

## AbstractEncoder
```python
AbstractEncoder(ABC)
```
Abstract class to define encoding mechanism.


### Functions
```python
__init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None
```
Initialize abstract encoder.

```python
__call__(self, data: torch.Tensor) -> None
```
Compute the encoded tensor of the given data.

### Parameters
- `time` (int): Length of encoded tensor.
- `dt` (float = 1.0): Simulation time step. The default is 1.0.
- `device` (str = "cpu"): The device to do the computations. The default is "cpu".

## NullEncoder
```python
NullEncoder(AbstractEncoder)
```
A null encoder which pass through of the datum that was input.

### Functions
```python
__init__(self) -> None
```
Initialize null encoder.

```python
__call__(self, data: torch.Tensor) -> torch.Tensor
```
Pass through of the datum that was input without any modification.

### Parameters

__Note:__ This class doesn't have any parameters
### Example

```python
import torch
import synapticflow as sf

datum = torch.empty(10).uniform_(20, 100)
null_encoder = sf.NullEncoder()
encoded = null_encoder(datum)
print(encoded)
```
## SingleEncoder
```python
SingleEncoder(AbstractEncoder)
```
Generates timing based single-spike encoding. Spike occurs earlier if the intensity of the input feature is higher. Features whose value is lower than the threshold remain silent.

### Functions
```python
__init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None
```
Initialize single encoder with given parameters.

```python
__call__(self, data: torch.Tensor, sparsity = 0.5) -> torch.Tensor
```
Generates encoded spikes with given data and sparsity.

### Parameters
- `time` (int): Length of encoded tensor.
- `dt` (float = 1.0): Simulation time step. The default is 1.0.
- `device` (str = "cpu"): The device to do the computations. The default is "cpu".
- `sparsity` (float = 0.5): Indicates sparsity of encoded trace.

### Example
```python
import torch
import synapticflow as sf

datum = torch.empty(10).uniform_(20, 100) 
encoder = sf.SingleEncoder(time = 10)
spikes = encoder(data= datum)
print(spikes)
sf.raster_plot(spikes, dt = 0.1)
```
## RepeatEncoder
```python
RepeatEncoder(AbstractEncoder)
```
Repeats a tensor along a new dimension in the 0th position for `int(time / dt)` time steps.

### Functions
```python
__init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None
```
Initialize repeat encoder with given parameters.

```python
__call__(self, data: torch.Tensor) -> torch.Tensor
```
Generates encoded spikes with given data.

### Parameters
- `time` (int): Length of encoded tensor.
- `dt` (float = 1.0): Simulation time step. The default is 1.0.
- `device` (str = "cpu"): The device to do the computations. The default is "cpu".

### Example
```python
import torch
import synapticflow as sf

datum = torch.empty(10).uniform_(20, 100)
encoder = sf.RepeatEncoder(time = 10)
spikes = encoder(data = datum)

print(spikes)
sf.raster_plot(spikes, dt = 0.1)
```

## BernoulliEncoder
```python
BernoulliEncoder(AbstractEncoder)
```
Generates Bernoulli-distributed spike trains based on input intensity. Inputs must be non-negative. Spikes correspond to successful Bernoulli trials, with success probability equal to (normalized in [0, 1]) input value.

### Functions
```python
__init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None
```
Initialize bernoulli encoder with given parameters.

```python
__call__(self, data: torch.Tensor, max_prob = 1.0) -> None
```
Generates encoded spikes with given data and max probability.

### Parameters
- `time` (int): Length of encoded tensor.
- `dt` (float = 1.0): Simulation time step. The default is 1.0.
- `device` (str = "cpu"): The device to do the computations. The default is "cpu".
- `max_prob` (float = 1.0): The maximum probability in Bernoulli distribution.

### Example
```python
import torch
import synapticflow as sf

datum = torch.empty(10).uniform_(20, 100) 
encoder = sf.BernoulliEncoder(time = 10, max_prob = 0.6)
spikes = encoder(datum)

print(spikes)
sf.raster_plot(spikes, dt = 1)
```

## PoissonEncoder
```python
PoissonEncoder(AbstractEncoder)
```
Generates Poisson-distributed spike trains based on input intensity. Inputs must be non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for non-negative data incremented by one to avoid zero intervals while maintaining ISI distributions.

### Functions
```python
__init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None
```
Initialize poisson encoder with given parameters.

```python
__call__(self, data: torch.Tensor) -> torch.Tensor
```
Generates encoded spikes with given data.

### Parameters
- `time` (int): Length of encoded tensor.
- `dt` (float = 1.0): Simulation time step. The default is 1.0.
- `device` (str = "cpu"): The device to do the computations. The default is "cpu".

### Example
```python
import torch
import synapticflow as sf

datum = torch.empty(10).uniform_(60, 100)
encoder = sf.PoissonEncoder(time = 10, dt = 1)
spikes = encoder(data = datum)

print(spikes)
sf.raster_plot(spikes, dt = 1)
```

## RankOrderEncoder
```python
RankOrderEncoder(AbstractEncoder)
```
Encodes data via a rank order coding-like representation. One spike per neuron, temporally ordered by decreasing intensity. Inputs must be non-negative.

### Functions
```python
__init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None
```
Initialize rank order encoder with given parameters.

```python
__call__(self, data: torch.Tensor) -> None
```
Encodes data via rank order coding with given data.

### Parameters
- `time` (int): Length of encoded tensor.
- `dt` (float = 1.0): Simulation time step. The default is 1.0.
- `device` (str = "cpu"): The device to do the computations. The default is "cpu".

### Example
```python
import torch
import synapticflow as sf

datum = torch.empty(10).uniform_(20, 100) 
encoder = sf.RankOrderEncoder(time = 10)
spikes = encoder(data = datum)

print(spikes)
sf.raster_plot(spikes, dt = 1)
```

## Timeoffirstspike
```python
Timeoffirstspike(AbstractEncoder)
```
Time to first spike encoding is a neural coding strategy where the information is encoded based on the timing of the first action potential fired by a neuron. This approach is effective for representing sensory stimuli that have precise temporal features.

### Functions
```python
__init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None
```
Initialize time of first spike encoder with given parameters.

```python
__call__(self, data: torch.Tensor) -> None
```
Encodes data via time of first spike coding with given data.

### Parameters
- `time` (int): Length of encoded tensor.
- `dt` (float = 1.0): Simulation time step. The default is 1.0.
- `device` (str = "cpu"): The device to do the computations. The default is "cpu".

### Example
```python
import torch
import synapticflow as sf

encoder = sf.Timetofirstspike(time=10)
spikes = encoder(datum)

print(spikes)
sf.raster_plot(spikes, dt=0.1)
```
