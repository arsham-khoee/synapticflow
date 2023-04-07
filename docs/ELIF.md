<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# ELIF

## Introduction
The ELIF neuron model is an extension of the LIF (leaky integrate-and-fire) neuron model, which allows for more complex dynamics and a wider range of behaviors. ELIF stands for "exponential integrate-and-fire," which refers to the addition of an exponential function to the basic LIF model. The ELIF neuron model is used to study the behavior of neurons in the brain and has been shown to accurately capture many aspects of real neuron behavior.

<br>

## How does it work?
The Exponential Leaky Integrate-and-Fire (ELIF) neuron model is a modified version of the LIF model that includes a subthreshold depolarizing current. ELIF neurons have a resting potential and a threshold potential like LIF neurons, but the membrane potential dynamics are slightly different. The membrane potential of an ELIF neuron is given by:

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

If the membrane potential $u$ exceeds the threshold potential $u_{th}$, then the neuron fires a spike and the membrane potential is reset to the resting potential $u_{rest}$. If $u(t) \leq u_{th}$, the neuron remains in a subthreshold regime and the dynamics are governed by a subthreshold depolarizing current term $\Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T})$.

The ELIF model adds a degree of flexibility to the LIF model by allowing for the generation of subthreshold depolarizations in response to input. The model is also computationally efficient and relatively simple to implement. However, like the LIF model, the ELIF model is limited in its ability to accurately capture the complex dynamics of biological neurons. In particular, the model does not capture the effects of active membrane properties such as ion channels, and the subthreshold dynamics are based on an exponential function rather than the more physiologically realistic Hodgkin-Huxley formalism.

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

 ELIF Population model can be used by given code:
 ```python
 from synapticflow.network import neural_population
 model = ELIFPopulation(n=10)
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
