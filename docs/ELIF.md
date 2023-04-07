<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# ELIF

## Introduction
The ELIF neuron model is an extension of the LIF (leaky integrate-and-fire) neuron model, which allows for more complex dynamics and a wider range of behaviors. ELIF stands for "exponential integrate-and-fire," which refers to the addition of an exponential function to the basic LIF model. The ELIF neuron model is used to study the behavior of neurons in the brain and has been shown to accurately capture many aspects of real neuron behavior.

<br>

## How does it work?
In the exponential integrate-and-fire model, the differential equation for the *membrane potential* and a *reset condition* is given by:


$$
\begin{align*}
\\
&\tau_m\frac{du}{dt}\ = -[u(t) - u_{rest}] + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) + RI(t) &\text{if }\quad u(t) \leq u_{th}\\
\end{align*}
$$

$$
\begin{align*}
&u(t) = u_{rest} &\text{otherwise}\\
\\
\end{align*}
$$


The first term on the right-hand of this equation describes the leak of a passive membrane. The second term is an exponential nonlinearity with *sharpness* parameter $\Delta_T$ and *threshold* $\theta_{rh}$.

The membrane equation is an *ordinary differential equation (ODE)* that illustrates the time evolution of membrane potential $u(t)$ in response to synaptic input and leaking of change across the cell membrane.

The exponential integrate-and-fire model is a special case of the general nonlinear model where:

<br>

$$f(u) = -[u(t) - u_{rest}] + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) $$

<br>

To solve this particular ODE, we can apply the forward Euler method in order to solve the ODE with a given initial value. We simulate the evolution of the membrane equation in discrete time steps, with a sufficiently small $\Delta t$. We start by writing the time derivative $\frac{du}{dt}$ in the membrane equation without taking the limit $\Delta t \to 0$:

$$
\begin{align*}
\\
\tau_m\frac{ u(t+\Delta t)-u(t)}{\Delta t}\ = -[u(t) - u_{rest}] + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) + RI(t)
\\
\\
\end{align*}
$$

The equation can be transformed to the following well-formed equation:

$$
\begin{align*}
\\
u(t+\Delta t) = u(t)-\frac{\Delta t}{\tau_m} \left( [u(t) - u_{rest}]  - \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) - RI(t) \right)
\\
\\
\end{align*}
$$

The value of membrane potential $u(t+\Delta t)$ can be expressed in terms of its previous value $u(t)$ by simple algebraic manipulation. For *small enough* values of $\Delta t$, this provides a good approximation of the continuous-time integration.

<br>

## Strengths:
<li>The ELIF model is an extension of the LIF model, which adds an exponential term to capture subthreshold dynamics. This allows for more accurate modeling of the behavior of real neurons, which may exhibit complex subthreshold dynamics that cannot be captured by the LIF model.
<li>The ELIF model still maintains many of the computational advantages of the LIF model, such as computational efficiency and ease of implementation.
<li>The ELIF model has been shown to accurately capture many important aspects of real neural behavior, such as spike frequency adaptation and resonance.

## Weaknesses:
<li>The ELIF model is more complex than the LIF model, which may make it more difficult to understand and implement, particularly for non-experts in the field of computational neuroscience.
<li>The ELIF model may require more computational resources than the LIF model, particularly when simulating large-scale networks.
<li>The ELIF model still has some limitations, such as the assumption of instantaneous spikes and the lack of consideration of dendritic processing or synaptic plasticity, which may limit its usefulness in certain contexts.

<br>

## Usage
To use a ELIF neuron, you need to create an object from the LIFPopulation class, which can be done using the following example code:
```python
neuron = ELIFPopulation(n=1)
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
