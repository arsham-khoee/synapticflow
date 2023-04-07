<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# AELIF

## Introduction
Adaptive Exponential Leaky-Integrate-and-Fire (AELIF) is an extension of the classical LIF neuron model that incorporates an adaptive exponential function to capture the sub-threshold dynamics of the neuron membrane potential. The AELIF model provides a more accurate representation of the biological behavior of neurons by accounting for the non-linear relationship between the input current and the membrane potential. The adaptive component allows the model to adjust the firing threshold in response to changes in input statistics, making it more biologically plausible and suitable for modeling neural systems that exhibit adaptive behavior.

<br>

## How does it work?
The ALIF neuron is an extension of the LIF neuron that takes into account the adaptation of the neuron's firing threshold. The membrane equation for the ALIF neuron is given by:

$$
\begin{align*}
\\
\tau_m\frac{du}{dt}\ =  -[u(t) - u_{rest}] - R\sum_{k} w_k + RI(t) \\
\end{align*}
$$

$$
\begin{align*}
\tau_k\frac{dw_k}{dt}\ = a_k (u - u_{rest}) - w_k + b_k\tau_k \sum_{t {(f)}} \delta (t - t^{(f)}) \\
\\
\end{align*}
$$

where $u(t)$ is the membrane potential, $\tau_m$ is the membrane time constant, $R$ is the membrane resistance, $C$ is the membrane capacitance, $I(t)$ is the synaptic input current, $u_{rest}$ is the resting potential, $w(t)$ is the adaptation variable, $z(t)$ is the input from the adaptation current, $\Delta_{th}$ is the adaptation strength for the firing threshold, $\Delta_{w}$ is the adaptation strength for the adaptation variable, and $\tau_w$ is the time constant for the adaptation variable.

The ALIF neuron has two dynamics: 
- The membrane potential $u(t)$ and the adaptation variable $w(t)$.
- The adaptation variable w(t) is increased after each spike and decays back to zero with a time constant $\tau_w$. 
This increase in w(t) causes the firing threshold to increase over time, resulting in a slower firing rate.

The ALIF neuron can be simulated using the forward Euler method as in the LIF neuron. However, since there are now two dynamics, we must solve two ODEs in each time step. This can be achieved by first updating the adaptation variable w(t) and then updating the membrane potential u(t).

<br>

## Strengths:
<li>The adaptive threshold in ALIF model provides a more realistic representation of neuronal behavior compared to traditional LIF models, as it accounts for the varying response of neurons to input stimuli.

<li>The ALIF model is capable of producing more precise spike timing compared to the LIF model, which can be useful in modeling and understanding various neural processes.

<li>The ALIF model is robust to changes in input statistics, making it more versatile and useful for a wider range of applications.
## Weaknesses:

<br>

## Weaknesses:
<li>The adaptive threshold in the ALIF model increases computational complexity compared to the LIF model, which may limit its use in certain applications.

<li>The increased complexity of the ALIF model may make it harder to interpret and understand compared to simpler models like the LIF model.

<li>The performance of the ALIF model depends on the specific parameter values chosen, which may require careful tuning for optimal results.

<br>

## Usage
To use a ALIF neuron, you need to create an object from the LIFPopulation class, which can be done using the following example code:
```python
neuron = ALIFPopulation(n=1)
```
When creating the object, you must specify the number of neurons (n) in that particular population of neurons.

After creating the object, the forward method can be used to activate the neuron for a one-time step with an input x that represents the amount of input current in that time step:

```python
neuron.forward(4)
```

<br>

## Reference
<li> Wikipedia
<li> Scholarpedia
