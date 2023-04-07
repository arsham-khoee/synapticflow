<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# QLIF

The quadratic LIF (QLIF) neuron model is a nonlinear version of the LIF neuron model. In the QLIF model, the membrane potential is presented by a quadratic function of the input signals. The membrane potential and a reset condition are given by:
<br>
$$
\begin{align*}
\\
\tau_m\frac{ u(t+\Delta t)-u(t)}{\Delta t}\ = a_0(u(t) - u_{rest}) (u(t) - u_{critical}) + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) + RI(t) ,
\\
\end{align*}
$$
<brwhere C is the membrane capacitance, g is the conductance, Vr is the resting potential, Vt is the firing threshold, and I is the input current. The conductance g and the firing threshold Vt are parameters of the model.

The quadratic term in this equation means that the response of the neuron to input signals is nonlinear. This can lead to behaviors such as subthreshold oscillations, which are not present in the linear LIF model.
When the membrane potential reaches the firing threshold, the neuron produces a spike, and the membrane potential is reset to a lower value. 

## How to simulate a QLIF neuron

To simulate a ELIF neuron, you need to create an object from the LIFPopulation class, which can be done using the following example code:

```python
neuron = QLIFPopulation(n=1)
```
When creating the object, you must specify the number of neurons (n) in that particular population of neurons.

After creating the object, the forward method can be used to activate the neuron for a one-time step with an input x that represents the amount of input current in that time step:

```python
neuron.forward(4)
```
