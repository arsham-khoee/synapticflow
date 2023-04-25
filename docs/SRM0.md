<script type="text/javascript" src="https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js"></script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>


# SRM0

## Introduction
The spike response model 0 (SRM0) is a simple yet powerful neuron model often used in computational neuroscience. It is based on the idea that neurons communicate through brief electrical spikes or action potentials, and captures the essential dynamics of spike generation in a mathematically tractable way. Unlike more complex models that require detailed knowledge of the underlying biophysical mechanisms, SRM0 only requires a few parameters to describe the basic properties of a neuron's firing behavior. This makes it an ideal tool for studying large-scale neural systems and their emergent properties. In this model, the neuron integrates incoming synaptic currents until its membrane potential reaches a threshold, at which point it generates a spike and resets to a baseline level. The timing and frequency of spikes are determined by the strength and timing of the incoming inputs, making SRM0 a powerful framework for exploring how network connectivity and input patterns shape neural activity.

## How does it work
The spike response model 0 (SRM0) works by modeling the essential dynamics of spike generation in a neuron using a simple mathematical framework. Specifically, it describes how the neuron's membrane potential changes as a function of incoming synaptic currents, which are modeled as time-varying input currents.

In SRM0, the neuron's membrane potential is represented by a scalar variable $V(t)$, which evolves over time according to the following equation:

$$C \frac{dV}{dt} = -V(t) + I_{\text{syn}}(t)$$

Here, $C$ is the capacitance of the neuron's membrane, and $I_{\text{syn}}(t)$ represents the total synaptic input current at time $t$. The term $-V(t)$ represents leakage, or the tendency for the membrane potential to return to a resting state over time.

When the membrane potential reaches a certain threshold value, denoted by $\theta$, the neuron generates a spike or action potential. This is modeled as a brief, all-or-nothing event that rapidly depolarizes the membrane potential before returning it to a baseline level, as follows:

$$V(t) \implies V_{\text{reset}}$$
$$\text{Spikes} = \{t: V(t) \ge \theta \}$$
$$V(t+dt) = V_{\text{reset}}$$

Here, $V_{\text{reset}}$ represents the reset potential of the neuron, which is the value to which the membrane potential returns after a spike. The set Spikes denotes the times at which spikes occur, and is updated each time the membrane potential crosses the threshold $\theta$.

The key parameters of `SRM0` are the membrane capacitance $C$, threshold voltage $\theta$, and reset voltage $V_{\text{reset}}$. By adjusting these parameters, researchers can explore how different input patterns and network topologies affect the firing behavior of the neuron. Additionally, the model can be extended to include more complex features such as refractoriness, synaptic plasticity, and noise, making it a versatile tool for studying neural systems across scales.

## Usage
SRM0 model can be used by following code:

```python
from synapticflow.network import neural_populations
model = neural_populations.SRM0Node(n=10) # Creates a neural population with 10 neurons
```

Then you can stimulate each time step by calling the forward function:

```python
# Stimulate one time step with 4A input current
model.forward(torch.tensor([4 for _ in range(10)]))
```

All available attributes like spike trace and membrane potential are available by model instance:

```python
print(model.s) # Spike trace
print(model.v) # Membrane potential
```

## Reference

<li> Gerstner, Wulfram, et al. Neuronal dynamics: From single neurons to networks and models of cognition. Cambridge University Press, 2014.
