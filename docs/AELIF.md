<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# AELIF

## Introduction
Adaptive Exponential Leaky-Integrate-and-Fire (AELIF) is an extension of the classical LIF neuron model that incorporates an adaptive exponential function to capture the sub-threshold dynamics of the neuron membrane potential. The AELIF model provides a more accurate representation of the biological behavior of neurons by accounting for the non-linear relationship between the input current and the membrane potential. The adaptive component allows the model to adjust the firing threshold in response to changes in input statistics, making it more biologically plausible and suitable for modeling neural systems that exhibit adaptive behavior.

<br>

## How does it work?
The adaptive exponential integrate-and-fire (AELIF) neuron is an extension of the LIF and ALIF models, which includes an exponential term that controls the neuron's adaptation. The membrane equation for AELIF is given by:

$$
\begin{align*}
\\
\tau_m\frac{du}{dt}\ =  -[u(t) - u_{rest}] + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) - R\sum_{k} w_k + RI(t) \\
\end{align*}
$$

where the exponential term with time constant $\Delta_T$ is responsible for the adaptation of the neuron, $\theta_{rh}$ is the rheobase threshold, and $\Delta_T$ is the slope factor. The AELIF model also includes a set of adaptation currents $w_k$ which are controlled by the second equation:

$$
\begin{align*}
\\
\tau_k\frac{dw_k}{dt}\ = a_k (u - u_{rest}) - w_k + b_k\tau_k \sum_{t {(f)}} \delta (t - t^{(f)})
\\
\end{align*}
$$

where $\tau_k$ is the time constant of the $k^{th}$ adaptation current, $a_k$ and $b_k$ control the amplitude and decay rate of the adaptation current, respectively. The adaptation current is incremented by $b_k$ every time an action potential is fired at time $t^{(f)}$.

To solve this set of differential equations, we can use the forward Euler method or other numerical techniques. The AELIF model provides a more realistic representation of neuron behavior, including adaptation effects that are present in real neurons.

<br>

## Strengths:
<li>AELIF model combines the advantages of adaptive and non-adaptive spiking neural models, allowing it to produce complex spiking behaviors while still being computationally efficient.

<li>AELIF model provides a good approximation of biological neurons that can adapt their firing rates in response to changing input patterns, making it suitable for modeling neural plasticity and learning.

<li>AELIF model has a low computational cost and can simulate large-scale neural networks efficiently.

<br>

## Weaknesses:
<li>AELIF model is a simplification of biological neurons and therefore may not capture all the complexities of neural dynamics.

<li>AELIF model requires the tuning of several parameters to match experimental data, which can be time-consuming and difficult.

<li>AELIF model may not accurately model some forms of neural plasticity or learning that involve more complex mechanisms.

<br>

## Usage

 AELIF Population model can be used by given code:
 ```python
 from synapticflow.network import neural_population
 model = AELIFPopulation(n=10)
 model.set_batch_size(10)
 ```

 Then you can stimulate each time step by calling `forward` function:
 ```python
 model.forward(torch.tensor([10 for _ in range(model.n)]))
 ```

 All available attributes like spike trace and membrane potential is available by `model` instance:
 ```python
 print(model.s) # Model spike trace
 print(model.v) # Model membrane potential
 ```

<br>

## Reference
<li> Wikipedia
<li> Scholarpedia
