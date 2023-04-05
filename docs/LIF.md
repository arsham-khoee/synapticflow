<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# LIF

The LIF (Leaky Integrate and Fire) neuron is a widely used model in computational neuroscience for simulating the behavior of a single neuron. It is a simplified model that captures the essential features of a neuron, including the integration of synaptic inputs and the generation of action potentials. The LIF neuron is based on the concept of the membrane potential, which is the electrical potential difference across the neuron's cell membrane. When the membrane potential reaches a certain threshold, an action potential is triggered, which allows the neuron to communicate with other neurons. The LIF neuron has been used extensively in the study of neural networks and information processing in the brain, and it continues to be an important tool in neuroscience research.


A *membrane equation* and a *reset condition* define our *leaky-integrate-and-fire (LIF)* neuron:
<br>
$$\tau_m \frac{du}{dt} = - [u(t) - u_{rest}] + RI(t) \ \ \ if \ \ \ u(t) \leq u_{th}$$
$$u(t) = u_{rest} \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad otherwise$$

\begin{align*}
\\
&\tau_m\frac{du}{dt}\ = -[u(t) - u_{rest}] + RI(t) &\text{if }\quad u(t) \leq u_{th}\\
<br>
<br>
&u(t) = u_{rest} &\text{otherwise}\\
\\
\end{align*}
where $u(t)$ is the membrane potential, $\tau_m$ is the membrane time constant which is equal to $RC$, it is the characteristic time of the decay, $R$ is the membrane resistance, $C$ is the capacity of the capacitor, $I(t)$ is the synaptic input current, $u_{th}$ is the spiking threshold, and $u_{rest}$ is the resting voltage.

The membrane equation is an *ordinary differential equation (ODE)* that illustrates the time evolution of membrane potential $u(t)$ in response to synaptic input and leaking of change across the cell membrane.

To solve this particular ODE, we can apply the forward Euler method in order to solve the ODE with a given initial value. We simulate the evolution of the membrane equation in discrete time steps, with a sufficiently small $\Delta t$. We start by writing the time derivative $\frac{du}{dt}$ in the membrane equation without taking the limit $\Delta t \to 0$:

$$\tau_m\,\frac{ u(t+\Delta t)-u(t)}{\Delta t}\ = -[u(t) - u_{rest}] + R\,I(t)$$

The equation can be transformed to the following well-formed equation:

$$u(t+\Delta t) = u(t)-\frac{\Delta t}{\tau_m} \left( [u(t) - u_{rest}] - R\,I(t) \right)$$

The value of membrane potential $u(t+\Delta t)$ can be expressed in terms of its previous value $u(t)$ by simple algebraic manipulation. For *small enough* values of $\Delta t$, this provides a good approximation of the continuous-time integration.

Another concept to be considered is the refractory period. After the action potential occurs, however, there is a short period of refractoriness, which affects neuron firing. During the first part of the refractory period (the absolute refractory period), the neuron will not fire again no matter how great the stimulation. The absolute refractory period lasts for only a short time. It is followed by the relative refractory period, during which a stronger than usual stimulus is required to trigger the action potential before the neuron returns to its resting state. After the refractory period, the neuron will fire when the neural threshold is reached.

<img src="_static/membrane.jpeg" alt="Membrane Potential" style="width: 450px; padding: 25px; margin: auto;"/>

## How to simulate LIF meuron

To simulate a LIF neuron using the synaptic flow package, you must first create an object from the LIFPopulation class as in the example below:

```python
neuron = LIFPopulation(n=1)
```
In the construction of the object, we must specify the number of neurons (n) in that population of neurons.

With the function method, the neuron can be activated for a one-time step with an input x, which is the same amount of input current in that time step:

```python
neuron.forward(4)
```


