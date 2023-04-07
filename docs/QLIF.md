# QLIF

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

# LIF

The quadratic LIF (QLIF) neuron model is a nonlinear version of the LIF neuron model. In the QLIF model, the membrane potential is presented by a quadratic function of the input signals. The membrane potential and a reset condition are given by:
C dV/dt = -g(V-Vr)(V-Vt) + I
where C is the membrane capacitance, g is the conductance, Vr is the resting potential, Vt is the firing threshold, and I is the input current. The conductance g and the firing threshold Vt are parameters of the model.

The quadratic term in this equation means that the response of the neuron to input signals is nonlinear. This can lead to behaviors such as subthreshold oscillations, which are not present in the linear LIF model.
When the membrane potential reaches the firing threshold, the neuron produces a spike, and the membrane potential is reset to a lower value. 

