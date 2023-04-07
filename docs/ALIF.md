<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# ALIF

## Introduction
The Adaptive Exponential Leaky Integrate and Fire (ALIF) neuron model is an extension of the Exponential Leaky Integrate and Fire (ELIF) neuron model, which incorporates an adaptive mechanism to capture the time-varying properties of the neuron's membrane potential. The ALIF model allows for more accurate modeling of real neurons, which may exhibit complex temporal dynamics that cannot be captured by the simpler LIF or ELIF models. The adaptive mechanism of the ALIF model enables it to capture important features of neural behavior such as spike-frequency adaptation and subthreshold oscillations, making it a useful tool for studying the dynamics of individual neurons and neural networks. The ALIF model has been used successfully in a variety of theoretical and experimental neuroscience studies, and its flexibility and accuracy make it a valuable tool for investigating the behavior of neurons and neural networks in the brain.

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

Same as other integrate-and-fire models, the voltage variable $u$ is set to $u_{rest}$ if the membrane potential reaches the threshold. 
The $\delta - function$ in the $w_k$ equations indicates that the adaptation currents $w_k$ are increased by an amount $b_k$. For example, a value $b_k = 10 pA$ means that the adaptation current $w_k$is a $10pA$ stronger after a spike that it was just before the spike. The parameters $b_k$ are the *jump* of the spike-triggered adaptation.

The ALIF neuron has two dynamics: 
- The membrane potential $u(t)$ and the adaptation variable $w(t)$.
- The adaptation variable w(t) is increased after each spike and decays back to zero with a time constant $\tau_w$. 
This increase in w(t) causes the firing threshold to increase over time, resulting in a slower firing rate.

The ALIF neuron can be simulated using the forward Euler method as in the LIF neuron. However, since there are now two dynamics, we must solve two ODEs in each time step. This can be achieved by first updating the adaptation variable w(t) and then updating the membrane potential u(t).

$$
\begin{align*}
\\
\tau_m\frac{ u(t+\Delta t)-u(t)}{\Delta t}\ =  -[u(t) - u_{rest}] - R\sum_{k} w_k + RI(t) \\
\end{align*}
$$

$$
\begin{align*}
\tau_k\frac{ w(t+\Delta t)-w(t)}{\Delta t}\ = a_k (u - u_{rest}) - w_k + b_k\tau_k \sum_{t {(f)}} \delta (t - t^{(f)}) \\
\\
\end{align*}
$$

The equation can be transformed into the following well-formed equation:

$$
\begin{align*}
\\
u(t+\Delta t)\ = \ u(t)-\frac{\Delta t}{\tau_m} \left( [u(t) - u_{rest}] + R\sum_{k} w_k - RI(t) \right)
\\
\end{align*}
$$

$$
\begin{align*}
w(t+\Delta t) = w(t)-\frac{\Delta t}{\tau_k} \left( a_k (u - u_{rest}) - w_k + b_k\tau_k \sum_{t {(f)}} \delta (t - t^{(f)}) \right)
\\
\end{align*}
$$

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
