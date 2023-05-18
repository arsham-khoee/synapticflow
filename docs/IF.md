<script type="text/javascript" src="https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js"></script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>


# IF

## Introduction
The Integrate-and-Fire (IF) neuron model is a simple but effective mathematical model used to simulate the behavior of individual neurons in the brain. IF model assumes that a neuron integrates incoming electrical signals from other neurons and fires a spike of electrical activity when its membrane potential reaches a certain threshold value. The IF model has been widely used in computational neuroscience to understand the dynamics of single neurons and to build larger-scale neural networks.

<br>

## How does it work?
The Integrate-and-Fire (IF) neuron model is a simplified version of the LIF model that only considers the neuron's membrane potential crossing a fixed threshold. In this model, the membrane potential is assumed to increase linearly with the input current, and once it reaches the threshold value, the neuron fires a spike and the membrane potential is reset to the resting potential. The dynamics of the IF model can be described by the following equations:

$$
\begin{align*}
\\
&\tau_m\frac{du}{dt}\ = RI(t) &\text{if }\quad u(t) \leq u_{th}\\
\end{align*}
$$

$$
\begin{align*}
&u(t) = u_{rest} &\text{otherwise}\\
\\
\end{align*}
$$

where $u(t)$ is the membrane potential at time $t$, $u_{rest}$ is the resting potential, $u_{th}$ is the threshold potential, $R$ is the membrane resistance, $I(t)$ is the input current, and $\tau_m$ is the membrane time constant. If the membrane potential reaches the threshold value, a spike is emitted and the membrane potential is reset to the resting potential. The IF model is computationally efficient and can be used for large-scale simulations, but it lacks the ability to accurately model the neuron's subthreshold behavior and the effect of synaptic inputs on the membrane potential.

<br>

## Strengths:
<li>The IF neuron model is simple and computationally efficient, making it suitable for large-scale simulations.

<li>The model is capable of generating spike trains that exhibit certain statistical properties observed in real neurons, such as Poisson-like or regular firing.

<li>The IF neuron is often used in theoretical neuroscience to model the behavior of populations of neurons.

<br>

## Weaknesses:
<li>The IF model ignores many of the complex dynamics that are present in real neurons, such as spike-frequency adaptation, sub-threshold oscillations, and other nonlinear effects.

<li>The model assumes that the membrane potential of the neuron can be accurately described by a single scalar value, which is not always the case.

<li>The IF neuron model is unable to capture the dynamics of some important phenomena in neuroscience, such as the generation of action potentials, synaptic plasticity, and dendritic processing.

<br>

## Usage

 IF Population model can be used by given code:
 ```python
 from synapticflow.network import neural_populations
 model = neural_populations.IFPopulation(n=10)
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

 And in the same way, you can use the visualization file to draw plots of the obtained answer:

<p align="center">
  <img src="_static/IF-v.svg" alt="Voltage Plot" style="width: 600px; padding: 25px;"/>
  <img src="_static/IF-s.svg" alt="Raster Plot" style="width: 600px; padding: 25px;"/>
</p>

<br>
 
 ## Parameters:

  <b>n</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">int, optional</span>) - <span style="font-weight: 400;">Number of neurons in this layer.</span>

  <b>shape</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">Iterable[int], optional</span>) - <span style="font-weight: 400;">Shape of the input tensor to the layer.</span>

  <b>spike_trace</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">bool, optional</span>) - <span style="font-weight: 400;">Indicates whether to use synaptic traces or not.</span>

  <b>additive_spike_trace</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">bool, optional</span>) - <span style="font-weight: 400;">If true, uses additive spike traces instead of multiplicative ones.</span>

  <b>tau_s</b> (<span style="color:#2b8cee; font-style: italic; font-weight: 400;">float or torch.Tensor, optional</span>) - <span style="font-weight: 400;">Decay time constant for spike trace. Default : `10`</span>

  <b>threshold</b> (<span style="color:#2b8cee; font-style: italic; font-weight: 400;">float or torch.Tensor, optional</span>) - <span style="font-weight: 400;">The spike threshold of the neuron.</span>

  <b>rest_pot</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">float or torch.Tensor, optional</span>) - <span style="font-weight: 400;">The resting potential of the neuron.</span>

  <b>refrac_length</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">float or torch.Tensor, optional</span>) - <span style="font-weight: 400;">The refractory period length of the neuron in timesteps.</span>

  <b>dt</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">float, optional</span>) - <span style="font-weight: 400;">The time step length.</span>

  <b>lower_bound</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">float, optional</span>) - <span style="font-weight: 400;">Minimum value for the membrane potential of the neuron.</span>

  <b>sum_input</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">bool, optional</span>) - <span style="font-weight: 400;">If true, sums input instead of averaging it.</span>

  <b>trace_scale</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">float, optional</span>) - <span style="font-weight: 400;">Scaling factor for the synaptic traces.</span>

  <b>is_inhibitory</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">bool, optional</span>) - <span style="font-weight: 400;">Indicates whether the neuron is inhibitory or not.</span>

  <b>R</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">Union[float, torch.Tensor], optional</span>) - <span style="font-weight: 400;">The time constant of the neuron voltage decay.</span>

  <b>learning</b> (<span style="color: #2b8cee; font-style: italic; font-weight: 400;">bool, optional</span>) - <span style="font-weight: 400;">Indicates whether the neuron should update its weights during training.</span>


## Reference
<li> Wikipedia
<li> Scholarpedia
