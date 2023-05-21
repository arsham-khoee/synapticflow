---
hide-toc: true
---

# SynapticFlow

## Introduction 

<a class="reference external image-reference" href="https://snntorch.readthedocs.io/en/latest/?badge=latest"><img alt="https://github.com/jeshraghian/snntorch/actions/workflows/build.yml/badge.svg" src="https://github.com/jeshraghian/snntorch/actions/workflows/build.yml/badge.svg" /></a>
<a class="reference external image-reference" href="https://snntorch.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/snntorch/badge/?version=latest" /></a>
<a class="reference external image-reference" href="https://discord.gg/cdZb5brajb"><img alt="Discord" src="https://img.shields.io/discord/906036932725841941" /></a>
<a class="reference external image-reference" href="https://pypi.python.org/pypi/snntorch"><img alt="https://img.shields.io/pypi/v/snntorch.svg" src="https://img.shields.io/pypi/v/snntorch.svg" /></a>
<a class="reference external image-reference" href="https://anaconda.org/conda-forge/snntorch"><img alt="https://img.shields.io/conda/vn/conda-forge/snntorch.svg" src="https://img.shields.io/conda/vn/conda-forge/snntorch.svg" /></a>
<a class="reference external image-reference" href="https://pepy.tech/project/snntorch"><img alt="https://static.pepy.tech/personalized-badge/snntorch?period=total&amp;units=international_system&amp;left_color=grey&amp;right_color=orange&amp;left_text=Downloads" src="https://static.pepy.tech/personalized-badge/snntorch?period=total&amp;units=international_system&amp;left_color=grey&amp;right_color=orange&amp;left_text=Downloads" /></a>


<div class="sidebar-logo-container">
  <img class="sidebar-logo only-light" src="_static/logo-light-mode.png" alt="Light Logo" style="width: 600px; padding: 25px;"/>
  <img class="sidebar-logo only-dark" src="_static/logo-dark-mode.png" alt="Dark Logo" style="width: 600px; padding: 25px;"/>
</div>
 

Spiking Neural Networks (SNNs) are a type of artificial neural network that attempts to mimic the behavior of neurons in the brain. Unlike traditional neural networks that use continuous-valued signals, SNNs operate using discrete spikes of activity that are similar to the action potentials in biological neurons. SynapticFlow is a powerful Python package for prototyping and simulating SNNs. It is based on PyTorch and supports both CPU and GPU computation. SynapticFlow extends the capabilities of PyTorch and enables us to take advantage of using spiking neurons. Additionally, it offers different variations of synaptic plasticity as well as delay learning for SNNs.

Please consider supporting the SynapticFlow project by giving it a star ⭐️ on <a href="https://github.com/arsham-khoee/synapticflow">Github</a>, as it is a simple and effective way to show your appreciation and help the project gain more visibility.

If you encounter any problems, want to share your thoughts or have any questions related to training spiking neural networks, we welcome you to open an issue, start a discussion, or join our <a href="https://discord.gg/dhQyAMxM">Discord</a> channel where we can chat and offer advice.


<h2> SynapticFlow Structure </h2>
The following are the components included in SynapticFlow:
<br>

|        Component        |                        Description                        |
|:-----------------------:|:---------------------------------------------------------:|
|   synapticflow.network  | A spiking network components like neurons and connections |
|  synapticflow.encoding  |              Several encoders implementation              |
|  synapticflow.learning  |           Learning rules and surrogate gradients          |
| synapticflow.evaluation |         Several evaluation functions for networks         |
|  synapticflow.datasets  | Include MNIST, Fashion-MNIST, CIFAR-10 benchmark datasets |
|   synapticflow.vision   |         Include vision components for neuroscience         |
|    synapticflow.plot    |        Plot tools for neural networks visualization       |




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
  <li>seaborn</li>
  <li>math</li>
</ul>



```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

```{toctree}
:caption: SynapticFlow
:hidden: 
Introduction
Installation
quickstart
```

```{toctree}
:caption: Plotting
:hidden: 
Plot
```

```{toctree}
:caption: Network
:hidden: 
IF
LIF
ALIF
ELIF
AELIF
QLIF
AQLIF
SRM0
```

```{toctree}
:caption: Development
:hidden:
Contributing
license
```
