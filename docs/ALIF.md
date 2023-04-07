<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# ALIF

## Introduction
The Adaptive Exponential Leaky Integrate and Fire (ALIF) neuron model is an extension of the Exponential Leaky Integrate and Fire (ELIF) neuron model, which incorporates an adaptive mechanism to capture the time-varying properties of the neuron's membrane potential. The ALIF model allows for more accurate modeling of real neurons, which may exhibit complex temporal dynamics that cannot be captured by the simpler LIF or ELIF models. The adaptive mechanism of the ALIF model enables it to capture important features of neural behavior such as spike-frequency adaptation and subthreshold oscillations, making it a useful tool for studying the dynamics of individual neurons and neural networks. The ALIF model has been used successfully in a variety of theoretical and experimental neuroscience studies, and its flexibility and accuracy make it a valuable tool for investigating the behavior of neurons and neural networks in the brain.

<br>

## How does it work?
The Adaptive Exponential Integrate-and-Fire (AEIF) neuron model is a biologically realistic spiking neuron model that extends the popular Leaky Integrate-and-Fire (LIF) model by adding an exponential term to the subthreshold dynamics of the membrane potential. In contrast, the Adaptive Leaky Integrate-and-Fire (ALIF) model simplifies the AEIF model by removing the exponential term from the subthreshold dynamics of the membrane potential. The ALIF model is described by the following equations:

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

where $u$ is the membrane potential, $u_{rest}$ is the resting potential, $\tau_m$ is the membrane time constant, $R$ is the membrane resistance, $w_k$ are the synaptic conductances, $a_k$ and $b_k$ are parameters that control the adaptation current, and $I(t)$ is the input current. As in the AEIF model, the adaptation current $I_{adapt}(t)$ is given by a spike-rate-dependent term that reflects the history of spiking activity. However, in the ALIF model, this adaptation current is linearly related to the membrane potential, and there is no explicit exponential term. When the membrane potential reaches a threshold value, the neuron emits a spike and the membrane potential is reset to the resting potential. The ALIF model is computationally efficient and has been used in large-scale simulations of spiking neural networks.

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

 ALIF Population model can be used by given code:
 ```python
 from synapticflow.network import neural_population
 model = ALIFPopulation(n=10)
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
