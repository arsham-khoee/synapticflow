# SynapticFlow

<img align='center' src="https://raw.githubusercontent.com/arsham-khoee/synapticflow/Document/docs/_static/logo-light-mode.png" alt="Light Logo" style="width: 600px; padding: 25px;"/>

Spiking Neural Networks (SNNs) are a type of artificial neural network that attempts to mimic the behavior of neurons in the brain. Unlike traditional neural networks that use continuous-valued signals, SNNs operate using discrete spikes of activity that are similar to the action potentials in biological neurons. SynapticFlow is a powerful Python package for prototyping and simulating SNNs. It is based on PyTorch and supports both CPU and GPU computation. SynapticFlow extends the capabilities of PyTorch and enables us to take advantage of using spiking neurons. Additionally, it offers different variations of synaptic plasticity as well as delay learning for SNNs.

Please consider supporting the SynapticFlow project by giving it a star ⭐️ on <a href="https://github.com/arsham-khoee/synapticflow">Github</a>, as it is a simple and effective way to show your appreciation and help the project gain more visibility.

If you encounter any problems, want to share your thoughts or have any questions related to training spiking neural networks, we welcome you to open an issue, start a discussion, or join our <a href="https://discord.gg/dhQyAMxM">Discord</a> channel where we can chat and offer advice.


## Installation

To install synapticflow, run the following command in your terminal:

```python
$ pip install synapticflow
```

We recommend using this method to install synapticflow since it will ensure that you have the latest stable version installed.

If you prefer to install synapticflow from source instead, follow these instructions:

```python
$ git clone https://github.com/arsham-khoee/synapticflow
$ cd synapticflow
$ python setup.py install
```


<h3> Requirements </h3>
The requirements for SynapticFlow are as follows: 

<ul>
  <li>torch</li>
  <li>matplotlib</li>
</ul>

## Usage
After package installation has been finished, you can use it by following command:

```python
import synapticflow as sf
```

In following code, a simple LIF neuron has been instantiated:

```python
model = sf.LIFPopulation(n=1)
print(model.v) # Membrane Potential
print(model.s) # Spike Trace
```

## SynapticFlow Structure
The following are the components included in SynapticFlow:
<br>

<div align="center">

|        Component        |                        Description                        |
|:-----------------------:|:---------------------------------------------------------:|
|   synapticflow.network  | A spiking network components like neurons and connections |
|  synapticflow.encoding  |              Several encoders implementation              |
|  synapticflow.learning  |           Learning rules and surrogate gradients          |
| synapticflow.evaluation |         Several evaluation functions for networks         |
|  synapticflow.datasets  | Include MNIST, Fashion-MNIST, CIFAR-10 benchmark datasets |
|   synapticflow.vision   |         Include vision components for neuroscience         |
|    synapticflow.plot    |        Plot tools for neural networks visualization       |

</div>