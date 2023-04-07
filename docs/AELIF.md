


we explore the adaptive exponential integrate-and-fire model, we observerd that the dynamics of the memrane voltage in the nonlinear integrate-and-fire neuron is characterized by a function $f(u)$ where in the ELIF model $f(u) =  -[u(t) - u_{rest}] + \Delta_T exp(\frac{u - \theta_{rh}}{\Delta_T})$. AELIF model is a two-dimensional spiking neuron model where we couple the voltage equation to abstract current variables $w_k$, each described by a linear differential equation as below:
$$
\begin{align*}
\\
&\tau_m\,\frac{du}{dt}\ = f(u) - R\sum_{k} w_k + R\,I(t) \\
\\
&\tau_k\,\frac{dw_k}{dt}\ = a_k (u - u_{rest}) - w_k + b_k\tau_k \sum_{t^{(f)}} \delta (t - t^{(f)}) \\
\\
\end{align*}
$$
Same as other integrate-and-fire models, the voltage variable $u$ is set to $u_{rest}$ if the membrane potential reaches the threshold. 
The $\delta - function$ in the $w_k$ equations indicates that the adaptation currents $w_k$ are increased by an amount $b_k$. For example, a value $b_k = 10 pA$ means that the adaptation current $w_k$is a $10pA$ stronger after a spike that it was just before the spike. The parameters $b_k$ are the *jump* of the spike-triggered adaptation.

For simplicity, the voltage equation of the exponential LIF model could be coupled to a single variable $w$:
$$
\begin{align*}
\\
&\tau_m\,\frac{du}{dt}\ = -[u(t) - u_{rest}] + \Delta_T exp(\frac{u - \theta_{rh}}{\Delta_T}) - Rw + R\,I(t) \\
\\
&\tau_w\,\frac{dw_k}{dt}\ = a_k (u - u_{rest}) - w + b\tau_k \sum_{t^{(f)}} \delta (t - t^{(f)}) \\
\\
\end{align*}
$$
If the membrane potential reaches the threshold, the voltage variable $u$ is set to $u_{rest}$, and the adaptation variable $w$ is increased by an amount $b$. Two parameters characterize adaptation, the parameter $a$ is the source of the subthreshold adaptation since it couples adaptation to the voltage, and the spike-triggered adaptation is controlled by a combination of $a$ and $b$. The choice of $a$ and mainly $b$ determines the firing patterns of the neuron.
