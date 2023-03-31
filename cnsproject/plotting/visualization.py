import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import List, Callable
from functools import reduce
from operator import add, mul

from ..network.neural_populations import NeuralPopulations

def get_random_rgb() -> np.ndarray:
    r = random.random()
    g = random.random()
    b = random.random()
    color = (r, g, b)
    return np.array(color).reshape(1, -1)

def plot_current(currents: torch.Tensor, steps: int, dt: float) -> None:
    times = [dt * i for i in range(steps)]
    plt.plot(times, currents)
    plt.xlabel("Time")
    plt.ylabel("Current")
    plt.show()
    
def plot_adaptation(adaptations: List[float], dt: float) -> None:
    times = [dt * i for i in range(len(adaptations))]
    plt.plot(times, adaptations, c='r')
    plt.xlabel("Time")
    plt.ylabel("Adaptation Value")
    plt.show()
    
def plot_dopamin(dopamins: List[float], dt: float) -> None:
    plt.plot([dt * i for i in range(len(dopamins))], dopamins)
    plt.xlabel("Time")
    plt.ylabel("Dopamin")
    plt.show()
    
def get_spiked_neurons(spikes: torch.Tensor) -> torch.Tensor:
    spiked_neurons = list(map(lambda x: x[0], filter( lambda x: x[1] != 0, enumerate(spikes))))
    return torch.tensor(spiked_neurons)

def plot_activity(population_spikes: List[torch.Tensor], dt: float) -> None:
    steps = len(population_spikes[0])
    population_size = reduce(lambda acc, pop: acc + len(pop[0]), population_spikes, 0)
    activities = []
    for step in range(steps):
        active_neurons = reduce(add, map(lambda spikes: len(get_spiked_neurons(torch.flatten(spikes[step])), population_spikes)))
        activities.append(active_neurons / population_size)
    
    plt.plot([dt * i for i in range(steps)], activities)
    plt.xlabel("Time")
    plt.ylabel("Activities")
    
def raster_plot(populations_spikes: List[torch.Tensor], dt: float) -> None:
    acc = 0
    for spikes_per_step in populations_spikes:
        color = get_random_rgb()
        for step, spikes in enumerate(spikes_per_step):
            spikes_flatten = torch.flatten(spikes)
            spiked_neurons = get_spiked_neurons(spikes_flatten)
            plot_neuron_index = list(map(lambda x: x + acc, spiked_neurons))
            plt.scatter(
                [dt * step] * len(spiked_neurons), 
                plot_neuron_index,
                c=color, s=[1] * len(spiked_neurons)
            )
        
        acc = acc + len(spikes_flatten)
        
    plt.xlabel("Time")
    plt.ylabel("Raster Activity")
    plt.show()
    
def plot_weights(weights: List[torch.Tensor], dt: float):
    weights_in_time = torch.tensor(
        list(map(
            lambda w: list(w.flatten()), weights
        ))
    ).transpose(0, 1)
    
    number_of_post_synaptic_neurons = len(weights[0])
    number_of_weights = len(weights[0][0])
    
    fig, axs = plt.subplots(number_of_post_synaptic_neurons)
    fig.tight_layout(pad=4.0)
    fig.set_size_inches(10, 10)
    colors = [get_random_rgb() for _ in range(number_of_weights)]
    for i in range(number_of_post_synaptic_neurons):
        for j in range(i * number_of_weights, (i+1) * number_of_weights):
            w = weights_in_time[j]
            steps = len(w)
            weight_number = j - i * number_of_weights
            axs[i].plot([dt * k for k in range(steps)], w, color=colors[weight_number])
            
        axs[i].set_title(f'Weights changes for post synaptic neuron {i + 1}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Weight Value')
        
    plt.show()
    