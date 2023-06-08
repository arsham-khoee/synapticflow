# Connection

## Introduction
Neural population connection, also known as synaptic connectivity, is a crucial aspect of building biologically-plausible spiking neural networks (SNNs). In the brain, neurons communicate with one another through the release of chemical neurotransmitters across synapses. These synapses can be excitatory, causing the receiving neuron to increase its firing rate, or inhibitory, causing the receiving neuron to decrease its firing rate. The precise pattern of connectivity between neurons within a neural population plays a significant role in shaping the network's emergent behavior and ultimately determines its computational capabilities. Therefore, understanding and implementing various types of neural population connections is essential for accurately modeling the brain's functionality and developing sophisticated SNN algorithms for a wide range of applications.

## Table of Contents
- AbstractConnection
- Connection
- SparseConnection
- RandomConnection

## AbstractConnection
```python
AbstractConnection(ABC, torch.nn.Module)
```

AbstractConnection is the base class for all connections. It is responsible for managing the weight and bias tensors.

### Functions
```python
    __init__(self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        w: torch.Tensor = None,
        d: torch.Tensor = None,
        d_min: float = 0.0,
        d_max: float = 100.0,
        mask: torch.ByteTensor = True,
        **kwargs) -> None:
```
    Initializes the AbstractConnection.

```python
    @abstractmethod
    update(self, **kwargs) -> None:
```
  Updates the weight and bias tensors.

```python
    @abstractmethod
    reset_state_variables(self) -> None:
```
  Resets the state variables of the connection.

```python
    @abstractmethod
    compute(self, s: torch.Tensor) -> None:
```
  Computes the post synaptic input according to weight and bias tensors.

### Parameters
- `pre` (NeuralPopulation): The neural population that the connection is receiving spikes from.
- `post` (NeuralPopulation): The neural population that the connection is sending spikes to.
- `w` (torch.Tensor, optional): The weight tensor.
- `d` (torch.Tensor, optional): The delay tensor.
- `d_min` (float, optional): The minimum delay.
- `d_max` (float, optional): The maximum delay.
- `w_min` (float, optional): The minimum weight.
- `w_max` (float, optional): The maximum weight.

__Note:__ This class is not instantiable due to abstract methods. You can implement your own connection via extending this class.

## Connection
```python
Connection(AbstractConnection)
```
The `Connection` class is normal fully connected connection between two neural populations.

### Functions
```python
__init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        w: torch.Tensor = None,
        d: torch.Tensor = None,
        d_min: float = 0.0,
        d_max: float = 100.0,
        mask: torch.ByteTensor = True,
        **kwargs
    ) -> None:
```
Initializes the connection.

```python
    reset_state_variables(self) -> None:
```
Resets the state variables of the connection.

```python
    normalize(self) -> None:
```
Normalizes the weight tensor.

```python
    update(self, **kwargs) -> None:
```
Updates the weight tensor.

```python
    compute(self, s: torch.Tensor) -> None:
```
Computes the post synaptic input according to weight and bias tensors.

### Parameters
- `pre` (NeuralPopulation): The neural population that the connection is receiving spikes from.
- `post` (NeuralPopulation): The neural population that the connection is sending spikes to.
- `w` (torch.Tensor, optional): The weight tensor.
- `d` (torch.Tensor, optional): The delay tensor.
- `d_min` (float, optional): The minimum delay.
- `d_max` (float, optional): The maximum delay.
- `w_min` (float, optional): The minimum weight.
- `w_max` (float, optional): The maximum weight.

### Example
Code:
```python
# Import required libraries
import torch
import synapticflow as sf

# Declare two neural populations
n1 = sf.LIFPopulation(n = 5)
n2 = sf.LIFPopulation(n = 10)

# Establish connection between two neural populations
connection = sf.Connection(pre=n1, post=n2)

# Print random uniform weights
print(connection.w)
```

Output:
```
Parameter containing:
tensor([[0.4637, 0.3979, 0.9405, 0.3354, 0.7101, 0.0211, 0.5543, 0.9122, 0.9170,
         0.3592],
        [0.3630, 0.6550, 0.3645, 0.6296, 0.0214, 0.7687, 0.3418, 0.8894, 0.5924,
         0.3200],
        [0.0604, 0.7112, 0.7600, 0.8927, 0.3791, 0.6776, 0.2899, 0.0476, 0.5319,
         0.9097],
        [0.4704, 0.9776, 0.5323, 0.4599, 0.2025, 0.1068, 0.4187, 0.7231, 0.4796,
         0.4809],
        [0.7463, 0.2404, 0.0485, 0.6853, 0.1235, 0.6587, 0.1126, 0.3066, 0.5925,
         0.1395]])
```


## SparseConnection
```python
SparseConnection(AbstractConnection)
```
The `SparseConnection` class is providing a sparse connection between two neural populations. It gets a sparsity constant between 0 and 1. This constant indicates portion of weights which should be zero.

### Functions
```python
__init__(self, 
        pre: NeuralPopulation, 
        post: NeuralPopulation,
        w: torch.Tensor = None,
        d: torch.Tensor = None,
        d_min: float = 0.0,
        d_max: float = 100.0,
        mask: torch.ByteTensor = True,
        sparsity: float = 0.7,
        random_seed: int = 1234,
        **kwargs) -> None:
```
Initializes the `SparseConnection`.

```python
reset_state_variables(self) -> None:
```
Resets the state variables.

```python
normalize(self) -> None:
```
Normalizes the weight tensor.

```python
update(self, **kwargs) -> None:
```
Updates the weight tensor.

```python
compute(self, s: torch.Tensor) -> None:
```
Computes the post synaptic input according to weight and bias tensors.

### Parameters
- `pre` (NeuralPopulation): The neural population that the connection is receiving spikes from.
- `post` (NeuralPopulation): The neural population that the connection is sending spikes to.
- `w` (torch.Tensor, optional): The weight tensor.
- `d` (torch.Tensor, optional): The delay tensor.
- `d_min` (float, optional): The minimum delay.
- `d_max` (float, optional): The maximum delay.
- `w_min` (float, optional): The minimum weight.
- `w_max` (float, optional): The maximum weight.
- `mask` (torch.ByteTensor, optional): The mask tensor.
- `sparsity` (float, optional): The sparsity constant.
- `random_seed` (int, optional): The random seed.

### Example
Code:
```python
# Import required libraries
import torch
import synapticflow as sf

# Declare two neural populations
n1 = sf.LIFPopulation(n = 5)
n2 = sf.LIFPopulation(n = 10)

# Establish sparse connection between two neural populations
connection = sf.SparseConnection(pre=n1, post=n2, sparsity=0.7, random_seed=1234)

# Print random uniform weights
print(connection.w)
```
Output
```
Parameter containing:
tensor([[0.0000, 0.0000, 0.2598, 0.0000, 0.0000, 0.7006, 0.0518, 0.4681, 0.6738,
         0.0000],
        [0.0000, 0.0000, 0.0000, 0.8208, 0.0000, 0.0000, 0.2837, 0.6567, 0.0000,
         0.0000],
        [0.0000, 0.3043, 0.0000, 0.6294, 0.0000, 0.0000, 0.0000, 0.0000, 0.7842,
         0.0000],
        [0.0000, 0.0000, 0.0000, 0.3216, 0.0000, 0.0000, 0.8436, 0.0000, 0.0000,
         0.0000],
        [0.4108, 0.0000, 0.0000, 0.6419, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000]])
```


## RandomConnection
```python
RandomConnection(AbstractConnection)
```
The `RandomConnection` class is a subclass of the `AbstractConnection` class. It is used to connect neurons in a neural population and __weights won't change__.

### Functions
```python
__init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        w: torch.Tensor = None,
        d: torch.Tensor = None,
        d_min: float = 0.0,
        d_max: float = 100.0,
        mask: torch.ByteTensor = True,
        **kwargs
    ) -> None:
```
Initializes the `RandomConnection` class.

```python
reset_state_variables(self) -> None:
```
Resets the state variables.

```python
normalize(self) -> None:
```
Normalizes the weight tensor.

```python
update(self, **kwargs) -> None:
```
Updates the weight tensor.

```python
compute(self, s: torch.Tensor) -> None:
```
Computes the post synaptic input according to weight and bias tensors.

### Parameters
- `pre` (NeuralPopulation): The neural population that the connection is receiving spikes from.
- `post` (NeuralPopulation): The neural population that the connection is sending spikes to.
- `w` (torch.Tensor, optional): The weight tensor.
- `d` (torch.Tensor, optional): The delay tensor.
- `d_min` (float, optional): The minimum delay.
- `d_max` (float, optional): The maximum delay.
- `w_min` (float, optional): The minimum weight.
- `w_max` (float, optional): The maximum weight.
- `mask` (torch.ByteTensor, optional): The mask tensor.

### Example
```python
# Import required libraries
import torch
import synapticflow as sf

# Declare two neural populations
n1 = sf.LIFPopulation(n = 5)
n2 = sf.LIFPopulation(n = 10)

# Establish random connection between two neural populations
connection = sf.RandomConnection(pre=n1, post=n2)

# Print random uniform weights
print(connection.w)
```
