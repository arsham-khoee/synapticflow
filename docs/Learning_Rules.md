# Learning Rules

## Introduction
Spiking neural networks (SNNs) are a type of artificial neural network that model the behavior of biological neurons. One of the key features of SNNs is their ability to communicate information through discrete electrical impulses or "spikes," which closely resemble the behavior of real neurons in the brain.

Learning in SNNs involves adjusting the strengths of connections between neurons (known as synapses) based on patterns of spike activity. This process is crucial for enabling SNNs to perform complex tasks such as pattern recognition, decision-making, and control.

In this context, learning rules refer to the mathematical algorithms used to update the synaptic weights in an SNN during training. There are many different types of learning rules, each with its own strengths and weaknesses depending on the problem domain and target application.

## Table of Contents
- LearningRule
- NoOp
- STDP
- MSTDP
<!-- - FlatSTDP -->
<!-- - RSTDP -->
<!-- - DSTDP -->
<!-- - MNSTDP -->
<!-- - WeightDependent -->


## LearningRule
```python
LearningRule(ABC)
```

Abstract class for defining learning rules.

### Functions
```python
    __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        reduction: Optional[callable] = None,
        boundary: str = 'hard',
        **kwargs
    ) -> None:
```
    Initializes the LearningRule.

```python
    update(self) -> None:
```
  Updates the weights and biases tensors and apply weight decay and boundaries if its necessary.

### Parameters
- `connection` (AbstractConnection): The connection between two neural populations.
- `lr` (Union[float, Sequence[float]], optional): The pre-synaptic and post-synaptic learning rates.
- `weight_decay` (float, optional): Weight decay coefficient.
- `reduction` (callable, optional): Function to reduce the weight difference.
- `boundary` (str, optional): Indicates hard or soft boundary on weights. The default is `hard`.



## NoOp
```python
NoOp(LearningRule)
```

Learning rule with no effect.

### Functions
```python
__init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
```
    Initializes given parameters in super class `LearningRule`

```python
update(self, **kwargs) -> None:
```
    Only take care about synaptic decay and possible range of synaptic weights.

### Parameters
- `connection` (AbstractConnection): The connection between two neural populations.
- `lr` (Union[float, Sequence[float]], optional): The pre-synaptic and post-synaptic learning rates.
- `weight_decay` (float, optional): Weight decay coefficient.
- `reduction` (callable, optional): Function to reduce the weight difference.
- `boundary` (str, optional): Indicates hard or soft boundary on weights. The default is `hard`.

### Example
```python
# Import required libraries
import synapticflow as sf
import torch

# Create neural populations
n1 = sf.LIFPopulation(n = 3, refrac_length=0)
n2 = sf.LIFPopulation(n = 1, refrac_length=0)

# Create connection
connection = sf.Connection(pre = n1, post = n2)
# Weights before learning rule
print(connection.w)

# Define learning rule
learning_rule = sf.NoOp(connection, lr=[0.0002, 0.0003])

# Stimulate neural inputs and dynamics for 1000 time step
for i in range(1000):
    n1.forward(torch.tensor([16,0,1]))
    n2.forward(torch.tensor([4]))
    learning_rule.update()
    print('-' * 10 , i, '-' * 10)
    print('weights')
    print(connection.w)

# Weights after update
print(connection.w)
```

## STDP
```python
STDP(LearningRule):
```
Spike-Time Dependent Plasticity learning rule.

### Functions

```python
__init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        boundary: str = 'hard',
        **kwargs
    ) -> None:
```
    Initialize STDP learning rule.

```python
update(self, **kwargs) -> None:
```

    Update weights according to pre and post-synaptic spikes.

### Parameters

- `connection` (AbstractConnection): The connection between two neural populations.
- `lr` (Union[float, Sequence[float]], optional): The pre-synaptic and post-synaptic learning rates.
- `weight_decay` (float, optional): Weight decay coefficient.
- `reduction` (callable, optional): Function to reduce the weight difference.
- `boundary` (str, optional): Indicates hard or soft boundary on weights. The default is `hard`.

### Example
Code:
```python
import synapticflow as sf
import torch

n1 = sf.LIFPopulation(n=2, refrac_length=0)
n2 = sf.LIFPopulation(n=1, refrac_length=0)

connection = sf.Connection(pre=n1, post=n2)
stdp = sf.STDP(connection=connection, lr=[0.0003, 0.0002], weight_decay=0.0001)

print("Pre-trained weights: ", connection.w)

for i in range(1000):
    n1.forward(torch.tensor([0, 50]))
    n2.forward(torch.tensor([50]))
    stdp.update()

print("Post-trained weights: ", connection.w)
```

Output:
```
Pre-trained weights:  
    Parameter containing:
    tensor([[0.4434],
            [0.8602]])
Post-trained weights:  
    Parameter containing:
    tensor([[0.4012],
            [0.9686]])
```

As you can see, neurons which spike fire together are wired together and their weights increased by iteration.

## MSTDP

```python
    MSTDP(LearningRule)
```

Modulated Spike-Time Dependent Plasticity

### Functions
```python
    __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            reduction: Optional[callable] = None,
            weight_decay: float = 0.,
            boundary: str = 'hard',
            **kwargs
        ) -> None:
```
    Initializing Modulate STDP

```python
    update(self, **kwargs) -> None:
```
    Weights update with respect to pre/post-synaptic spikes. Note in `kwargs`, reward should be included according to model's performance

### Parameters

- `connection` (AbstractConnection): The connection between two neural populations.

- `lr` (Union[float, Sequence[float]], optional): The pre-synaptic and post-synaptic learning rates.

- `weight_decay` (float, optional): Weight decay coefficient.

- `reduction` (callable, optional): Function to reduce the weight difference.

- `boundary` (str, optional): Indicates hard or soft boundary on weights. The default is `hard`.

- `kwargs`: There are several keyword arguments which may be passed to this learning rule :
    - `tc_plus` (float, optional): Positive time constant to update weights
    - `tc_minus` (float, optional): Negative time constant to update weights
    - `reward` (float, mandatory): This attribute should be set in each call of `update` function to indicates positive or negative feedback for updating weights
    - `a_plus` (float, optional): This attribute indicate positive a coefficient in RSTDP.
    - `a_minus` (float, optional): This attribute indicate negative a coefficient in RSTDP.

### Example

```python
import synapticflow as sf
import torch
import random

n1 = sf.LIFPopulation(n=2, refrac_length=0)
n2 = sf.LIFPopulation(n=1, refrac_length=0)

connection = sf.Connection(pre=n1, post=n2, w_min=0, w_max=10)

mstdp = sf.MSTDP(connection=connection, lr=[0.0003, 0.0002])


nums = [0, 1]
for i in range(1000):
    print(f"Epoch: {i}")
    for j in range(1000):
        n = random.sample(nums,1)
        if n == [0]:
            n1.forward(torch.tensor([0, 100]))
            n2.forward(connection.compute(n1.s))
            reward = 0
            if n2.s.any():
                reward = -100
            else:
                reward = 100
        
        else:
            n1.forward(torch.tensor([100, 0]))
            n2.forward(connection.compute(n1.s))
            reward = 0
            if n2.s.any():
                reward = 100
            else:
                reward = -100
        
        mstdp.update(reward=reward)
    

print("Post-trained weights", connection.w)
```