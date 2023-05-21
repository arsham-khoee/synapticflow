# Connection:

## Introduction:
Neural population connection, also known as synaptic connectivity, is a crucial aspect of building biologically-plausible spiking neural networks (SNNs). In the brain, neurons communicate with one another through the release of chemical neurotransmitters across synapses. These synapses can be excitatory, causing the receiving neuron to increase its firing rate, or inhibitory, causing the receiving neuron to decrease its firing rate. The precise pattern of connectivity between neurons within a neural population plays a significant role in shaping the network's emergent behavior and ultimately determines its computational capabilities. Therefore, understanding and implementing various types of neural population connections is essential for accurately modeling the brain's functionality and developing sophisticated SNN algorithms for a wide range of applications.

## Table of Contents:
- AbstractConnection
- Connection
- SparseConnection
- RandomConnection

## AbstractConnection:
```python
AbstractConnection(ABC, torch.nn.Module)
```

AbstractConnection is the base class for all connections. It is responsible for managing the weight and bias tensors.

### Functions:
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

### Parameters:
- `pre` (NeuralPopulation): The neural population that the connection is receiving spikes from.
- `post` (NeuralPopulation): The neural population that the connection is sending spikes to.
- `w` (torch.Tensor, optional): The weight tensor.
- `d` (torch.Tensor, optional): The delay tensor.
- `d_min` (float, optional): The minimum delay.
- `d_max` (float, optional): The maximum delay.
- `w_min` (float, optional): The minimum weight.
- `w_max` (float, optional): The maximum weight.

__Note:__ This class is not instantiable due to abstract methods. You can implement your own connection via extending this class.

## Connection:
```python
Connection(AbstractConnection)
```
The `Connection` class is normal fully connected connection between two neural populations.

### Functions:
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

### Parameters:
- `pre` (NeuralPopulation): The neural population that the connection is receiving spikes from.
- `post` (NeuralPopulation): The neural population that the connection is sending spikes to.
- `w` (torch.Tensor, optional): The weight tensor.
- `d` (torch.Tensor, optional): The delay tensor.
- `d_min` (float, optional): The minimum delay.
- `d_max` (float, optional): The maximum delay.
- `w_min` (float, optional): The minimum weight.
- `w_max` (float, optional): The maximum weight.

## SparseConnection:
```python
SparseConnection(AbstractConnection)
```
The `SparseConnection` class is providing a sparse connection between two neural populations. It gets a sparsity constant between 0 and 1. This constant indicates portion of weights which should be zero.

### Functions:
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

### Parameters:
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

## RandomConnection:
```python
RandomConnection(AbstractConnection)
```
The `RandomConnection` class is a subclass of the `AbstractConnection` class. It is used to connect neurons in a neural population and weights won't change.

### Functions:
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

### Parameters:
- `pre` (NeuralPopulation): The neural population that the connection is receiving spikes from.
- `post` (NeuralPopulation): The neural population that the connection is sending spikes to.
- `w` (torch.Tensor, optional): The weight tensor.
- `d` (torch.Tensor, optional): The delay tensor.
- `d_min` (float, optional): The minimum delay.
- `d_max` (float, optional): The maximum delay.
- `w_min` (float, optional): The minimum weight.
- `w_max` (float, optional): The maximum weight.
- `mask` (torch.ByteTensor, optional): The mask tensor.