# ELIF

The ELIF neuron model is an extension of the LIF (leaky integrate-and-fire) neuron model, which allows for more complex dynamics and a wider range of behaviors. ELIF stands for "exponential integrate-and-fire," which refers to the addition of an exponential function to the basic LIF model. The ELIF neuron model is used to study the behavior of neurons in the brain and has been shown to accurately capture many aspects of real neuron behavior.

The ELIF neuron model includes an additional parameter called the adaptation current, which represents the slow, cumulative effects of synaptic input on the neuron's membrane potential. The adaptation current is proportional to the difference between the membrane potential and a threshold value, and is multiplied by a time constant parameter. This means that the adaptation current builds up over time in response to sustained synaptic input, and causes the neuron to become less excitable.

In the exponential integrate-and-fire model, the differential equation for the *membrane potential* and a *reset condition* is given by:
<br>
$$
\begin{align*}
\\
&\tau_m\,\frac{du}{dt}\ = -[u(t) - u_{rest}] + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) + R\,I(t) &\text{if }\quad u(t) \leq u_{th}\\
\\
&u(t) = u_{rest} &\text{otherwise}\\
\\
\end{align*}
$$
<br>
The first term on the right-hand of this equation describes the leak of a passive membrane. The second term is an exponential nonlinearity with *sharpness* parameter $\Delta_T$ and *threshold* $\theta_{rh}$.

The membrane equation is an *ordinary differential equation (ODE)* that illustrates the time evolution of membrane potential $u(t)$ in response to synaptic input and leaking of change across the cell membrane.

The exponential integrate-and-fire model is a special case of the general nonlinear model where $f(u) = -[u(t) - u_{rest}] + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) $.







In the above figure the function $f(u)$ is plotted for different choices of the *sharpness* of the threshold ($\Delta_T = 1, 0.5, 0.25,  mV$). If the limit $\Delta_T \rightarrow 0$ the exponential integrate-and-fire model converts to a LIF model (dashed line). The figure displays a zoom onto the threshold region (dotted box).

To solve this particular ODE, we can apply the forward Euler method in order to solve the ODE with a given initial value. We simulate the evolution of the membrane equation in discrete time steps, with a sufficiently small $\Delta t$. We start by writing the time derivative $\frac{du}{dt}$ in the membrane equation without taking the limit $\Delta t \to 0$:
<br>
$$
\begin{align*}
\\
\tau_m\,\frac{ u(t+\Delta t)-u(t)}{\Delta t}\ = -[u(t) - u_{rest}] + \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) + R\,I(t) ,
\\
\end{align*}
$$
<br>
The equation can be transformed to the following well-formed equation:
<br>
$$
\begin{align*}
\\
u(t+\Delta t) = u(t)-\frac{\Delta t}{\tau_m} \left( [u(t) - u_{rest}]  - \Delta_T exp(\frac{u(t) - \theta_{rh}}{\Delta_T}) - R\,I(t) \right) .
\\
\end{align*}
$$
<br>
The value of membrane potential $u(t+\Delta t)$ can be expressed in terms of its previous value $u(t)$ by simple algebraic manipulation. For *small enough* values of $\Delta t$, this provides a good approximation of the continuous-time integration.

