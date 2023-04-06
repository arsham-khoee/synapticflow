<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# LIF

The Leaky-Integrate-and-Fire (LIF) neuron is a widely used model in computational neuroscience that simulates the behavior of a single neuron. It is a simplified model that captures essential features of neurons, including synaptic input integration and action potential generation. The model relies on the concept of membrane potential, which is the electrical potential difference across a neuron's cell membrane. Once the membrane potential reaches a specific threshold, an action potential is triggered, allowing the neuron to communicate with other neurons. 


A *membrane equation* and a *reset condition* define our *leaky-integrate-and-fire (LIF)* neuron:
<br>


$$
\begin{align*}
\\
&\tau_m\frac{du}{dt}\ = -[u(t) - u_{rest}] + RI(t) &\text{if }\quad u(t) \leq u_{th}\\
\end{align*}
$$

$$
\begin{align*}
&u(t) = u_{rest} &\text{otherwise}\\
\\
\end{align*}
$$

where $u(t)$ is the membrane potential, $\tau_m$ is the membrane time constant which is equal to $RC$, it is the characteristic time of the decay, $R$ is the membrane resistance, $C$ is the capacity of the capacitor, $I(t)$ is the synaptic input current, $u_{th}$ is the spiking threshold, and $u_{rest}$ is the resting voltage.

The membrane equation is an *ordinary differential equation (ODE)* that illustrates the time evolution of membrane potential $u(t)$ in response to synaptic input and leaking of change across the cell membrane.

To solve this particular ODE, we can apply the forward Euler method in order to solve the ODE with a given initial value. We simulate the evolution of the membrane equation in discrete time steps with a sufficiently small $\Delta t$. We start by writing the time derivative $\frac{du}{dt}$ in the membrane equation without taking the limit $\Delta t \to 0$:
<br>
$$
\begin{align}
\\
\tau_m\frac{ u(t+\Delta t)-u(t)}{\Delta t}\ = -[u(t) - u_{rest}] + RI(t) ,
\\
\end{align}
$$
<br>
The equation can be transformed into the following well-formed equation:
<br>
$$
\begin{align}
\\
u(t+\Delta t) = u(t)-\frac{\Delta t}{\tau_m} \left( [u(t) - u_{rest}] - RI(t) \right) .
\\
\end{align}
$$
<br>
The value of membrane potential $u(t+\Delta t)$ can be expressed in terms of its previous value $u(t)$ by simple algebraic manipulation. For *small enough* values of $\Delta t$, this provides a good approximation of the continuous-time integration.

Another concept to be considered is the refractory period. After the action potential occurs, however, there is a short period of refractoriness, which affects neuron firing. During the first part of the refractory period (the absolute refractory period), the neuron will not fire again, no matter how great the stimulation. 

<br>

<div class="sidebar-logo-container">
  <p align="center">
    <img class="sidebar-logo only-light" src="_static/membrane.jpeg" alt="Light Membrane" style="width: 600px; padding: 25px;"/>
    <img class="sidebar-logo only-dark" src="_static/dark-membrane.jpeg" alt="Dark Membrane" style="width: 600px; padding: 25px;"/>
  </p>
</div>

<br>

## How to simulate a LIF neuron

To simulate a LIF neuron, you need to create an object from the LIFPopulation class, which can be done using the following example code:

```python
neuron = LIFPopulation(n=1)
```
When creating the object, you must specify the number of neurons (n) in that particular population of neurons.

After creating the object, the forward method can be used to activate the neuron for a one-time step with an input x that represents the amount of input current in that time step:

```python
neuron.forward(4)
```


