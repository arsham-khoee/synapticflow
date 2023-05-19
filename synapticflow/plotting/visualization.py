import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import List, Callable
from functools import reduce
from operator import add, mul
import copy

from ..network.neural_populations import NeuralPopulation

def get_random_rgb() -> np.ndarray:
    """
    Generate a random RGB color and return it as an ndarray.

    Returns:
        np.ndarray: A 1D array of shape (3,) representing the RGB color.
    """
    r = random.random()
    g = random.random()
    b = random.random()
    color = (r, g, b)
    return np.array(color).reshape(1, -1)

def plot_current(currents: List[torch.Tensor], dt: float, save_path: str = None, legend: bool = False, default_colors: List = None) -> None:
    """
    Plot the input current of neurons.

    Args:
        currents (List[torch.Tensor]): A list of Tensors representing the input currents of each neuron at each time step.
        dt (float): The time step used in the simulation.
        save_path (str, optional): The file path to save the plot. Defaults to None.
        legend (bool, optional): Whether to include a legend in the plot. Defaults to False.
        default_colors (List, optional): A list of colors to use for each neuron. Defaults to None.

    Returns:
        None
    """
    current_size = len(currents[0])
    steps = len(currents)
    data = {}
    for i in range(current_size):
        data[i] = []
    
    for s in range(steps):
        for i in range(current_size):
            data[i].append(currents[s][i].item())
    
    time = [dt * i for i in range(steps)]
    colors = []
    if default_colors:
        colors = default_colors
    else:
        colors = [get_random_rgb() for _ in range(current_size)]
    
    for i in range(current_size):
        plt.plot(time, data[i], color = colors[i])
    
    plt.xlabel("Time")
    plt.ylabel("Input Current")
    if legend:
        plt.legend([f'Neuron {i}' for i in range(current_size)])
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_adaptation(adaptations: List[float], dt: float, save_path: str = None) -> None:
    """
    Plot the adaptation value of neurons.

    Args:
        adaptations (List[float]): A list of adaptation values of neurons at each time step.
        dt (float): The time step used in the simulation.
        save_path (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        None
    """
    times = [dt * i for i in range(len(adaptations))]
    plt.plot(times, adaptations, c='r')
    plt.xlabel("Time")
    plt.ylabel("Adaptation Value")
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_dopamin(dopamins: List[float], dt: float, save_path: str = None) -> None:
    """
    Plot the dopamine level.

    Args:
        dopamins (List[float]): A list of dopamine levels at each time step.
        dt (float): The time step used in the simulation.
        save_path (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        None
    """
    plt.plot([dt * i for i in range(len(dopamins))], dopamins)
    plt.xlabel("Time")
    plt.ylabel("Dopamin")
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def get_spiked_neurons(spikes: torch.Tensor) -> torch.Tensor:
    """
    Get the indices of neurons that spiked.

    Args:
        spikes (torch.Tensor): A 1D Tensor representing the spike train of neurons.

    Returns:
        torch.Tensor: A 1D Tensor containing the indices of neurons that spiked.
    """
    spiked_neurons = list(map(lambda x: x[0], filter( lambda x: x[1] != 0, enumerate(spikes))))
    return torch.tensor(spiked_neurons)

def plot_activity(population_spikes: List[torch.Tensor], dt: float, save_path: str = None, legend: bool = False, default_colors: List = None) -> None:
    """
    Plot the activity of neurons as a function of time.

    Args:
        population_spikes (List[torch.Tensor]): A list of Tensors representing the spike trains of neurons in a population at each time step.
        dt (float): The time step used in the simulation.
        save_path (str, optional): The file path to save the plot. Defaults to None.
        legend (bool, optional): Whether to include a legend in the plot. Defaults to False.
        default_colors (List, optional): A list of colors to use for each neuron. Defaults to None.

    Returns:
        None
    """
    steps = len(population_spikes)
    population_size = len(population_spikes[0])
    data = {}
    for i in range(population_size):
        data[i] = []
    for s in range(steps):
        for i in range(population_size):
            if population_spikes[s][i] == True:
                data[i].append(1)
            else:
                data[i].append(0)
    colors = []
    if default_colors:
        colors = default_colors
    else:
        colors = [get_random_rgb() for _ in range(population_size)]
    time = [dt * i for i in range(steps)]
    for i in range(population_size):
        plt.plot(time, data[i], color=colors[i])
    
    plt.xlabel("Time")
    plt.ylabel("Activities")
    if legend:
        plt.legend([f'Neuron {i}' for i in range(population_size)])
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def raster_plot(population_spikes: List[torch.Tensor], dt: float, save_path: str = None, legend: bool = False) -> None:
    """
    Generates a raster plot of population spikes.

    Args:
        population_spikes (List[torch.Tensor]): A list of tensors, where each tensor represents the spike activity of neurons in a population at different time steps.
        dt (float): The time step size used to simulate the neural activity. 
        save_path (str, optional): Path to save the generated plot. Defaults to None.
        legend (bool, optional): Whether to show the legend indicating the neuron index. Defaults to False.

    Returns:
        None
    """
    population_size = len(population_spikes[0])
    steps = len(population_spikes)
    
    data = {}
    for i in range(population_size):
        data[i] = []
    
    for s in range(steps):
        for i in range(population_size):
            if population_spikes[s][i] == True:
                data[i].append(i)
            else:
                data[i].append(None)

    time = [dt * i for i in range(steps)]
    for i in range(population_size):
        plt.scatter(time, data[i], color='royalblue', s=2)
    plt.xlabel("Time")
    plt.ylabel("Neuron Index")
    if legend:
        plt.legend([f'Neuron {i}' for i in range(population_size)])
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_weights(weights: List[torch.Tensor], dt: float, save_path: str = None):
    """
    Plot the weights between neurons as a function of time.

    Args:
        weights (List[torch.Tensor]): A list of Tensors representing the weights between neurons at each time step.
        dt (float): The time step used in the simulation.
        save_path (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        None
    """
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
    
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
def plot_potential(population_potentials: List[torch.tensor], dt: float, threshold: float = None, save_path: str = None, legend: bool = False, default_colors: List = None) -> None:
    """
    Plot the membrane potential of neurons as a function of time.

    Args:
        population_potentials (List[torch.tensor]): A list of Tensors representing the membrane potential of neurons in a population at each time step.
        dt (float): The time step used in the simulation.
        save_path (str, optional): The file path to save the plot. Defaults to None.
        legend (bool, optional): Whether to include a legend in the plot. Defaults to False.
        default_colors (List, optional): A list of colors to use for each neuron. Defaults to None.

    Returns:
        None
    """
    steps = len(population_potentials)
    population_size = len(population_potentials[0])
    
    data = {}
    for i in range(population_size):
        data[i] = []
    
    for s in range(steps):
        for i in range(population_size):
            data[i].append(population_potentials[s][i].item())
    
    colors = []
    if default_colors:
        colors = default_colors
    else:
        colors = [get_random_rgb() for _ in range(population_size)]
    
    time = [dt * i for i in range(steps)]
    for i in range(population_size):
        plt.plot(time, data[i], color=colors[i])
    
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    if threshold:
        plt.plot(time, [threshold for _ in range(steps)], linestyle="dotted")
    if legend:
        plt.legend([f'Neuron {i}' for i in range(population_size)])
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_refractory_count(population_refracts: List[torch.tensor], dt: float, save_path: str = None, legend: bool = False, default_colors: List = None) -> None:
    """
    Plot the refractory length of the population as a function of time.

    Args:
        population_refracts (List[torch.tensor]): A list of refractory counts of a population at each time step.
        dt (float): The time step used in the simulation.
        save_path (str, optional): The file path to save the plot. Defaults to None.
        legend (bool, optional): Whether to include a legend in the plot. Defaults to False.
        default_colors (List, optional): A list of colors to use for each neuron. Defaults to None.
    
    Returns:
        None
    """
    steps = len(population_refracts)
    population_size = len(population_refracts[0])
    
    data = {}
    for i in range(population_size):
        data[i] = []
    
    for s in range(steps):
        for i in range(population_size):
            data[i].append(population_refracts[s][i].item())
    
    colors = []
    if default_colors:
        colors = default_colors
    else:
        colors = [get_random_rgb() for _ in range(population_size)]
    
    time = [dt * i for i in range(steps)]
    for i in range(population_size):
        plt.plot(time, data[i], color=colors[i])
    
    plt.xlabel("Time")
    plt.ylabel("Refractory Count")
    if legend:
        plt.legend([f'Neuron {i}' for i in range(population_size)])
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_neuron(neural_population: NeuralPopulation, input_current: List[torch.tensor], dt: float, threshold: float = None, does_plot_current: bool = True, does_plot_potential: bool = True, does_plot_refractory: bool = True, does_plot_activity: bool = True, save_path: str = None, legend: bool = False, colors: List = None) -> None:
    """
    Plot all attributes of the population as a function of time.

    Args:
        neural_population (NeuralPopulation): A neural population.
        input_current (List[torch.tensor]): The list of input current over time.
        dt (float): The time step used in the simulation.
        does_plot_current (bool, optional): Whether to plot input current figure. Defaults to True.
        does_plot_potential (bool, optional): Whether to plot membrane potential of population figure. Defaults to True.
        does_plot_refractory (bool, optional): Whether to plot refractory count of population figure. Defaults to True.
        does_plot_activity (bool, optional): Whether to plot activity of population figure. Defaults to True.
        save_path (str, optional): The file path to save the plot. Defaults to None.
        legend (bool, optional): Whether to include a legend in the plot. Defaults to False.
        default_colors (List, optional): A list of colors to use for each neuron. Defaults to None.
    
    Returns:
        None
    """
    population_size = neural_population.n
    steps = len(input_current)
    if colors:
        colors = colors
    else:
        colors = [get_random_rgb() for _ in range(population_size)]
    
    if does_plot_current:
        if save_path:
            plot_current(input_current, dt, legend=legend, save_path=save_path + '/current.png', default_colors=colors)
        else:
            plot_current(input_current, dt, legend=legend, default_colors=colors)
    
    voltage_data = [neural_population.v]
    spike_data = [neural_population.s]
    refractory_data = [neural_population.refrac_count]
    
    for current in input_current:
        neural_population.forward(current)
        spike_data.append(copy.deepcopy(neural_population.s))
        voltage_data.append(copy.deepcopy(neural_population.v))
        refractory_data.append(copy.deepcopy(neural_population.refrac_count))
    
    if does_plot_activity:
        if save_path:
            raster_plot(spike_data, dt=dt, save_path=save_path + '/activity.png', legend=legend)
        else:
            raster_plot(spike_data, dt=dt, legend=legend)
        
    if does_plot_potential:
        if save_path:
            plot_potential(voltage_data, threshold=threshold, dt=dt, save_path=save_path + '/potential.png', legend=legend, default_colors=colors)
        else:
            plot_potential(voltage_data, dt=dt, threshold=threshold, legend=legend, default_colors=colors)
        
    if does_plot_refractory:
        if save_path:
            plot_refractory_count(refractory_data, dt=dt, save_path=save_path + '/refractory.png', legend=legend, default_colors=colors)
        else:
            plot_refractory_count(refractory_data, dt=dt, legend=legend, default_colors=colors)

def plot_periodic(data: dict, dt: float, threshold: float = None, save_path: str = None, legend: bool = False) -> None:
    """
    Plots periodic data for neurons.

    Args:
    - data (dict): Dictionary containing data for each timestep
    - dt (float): The time step size
    - save_path (str, optional): The file path to save the plot. Default is None.
    - legend (bool, optional): Whether to show the legend. Default is False.

    Returns:
    - None
    """
    steps = len(data.keys())
    population_size = len(data[0]['spike'])
    spikes = {}
    voltages = {}
    refracts = {}
    currents = {}
    threshold = threshold
    for i in range(population_size):
        spikes[i] = []
        voltages[i] = []
        refracts[i] = []
        currents[i] = []
    
    time = [dt * i for i in range(steps)]
    
    fig, axs = plt.subplots(4, sharex= True, figsize=(10, 8))
    fig.suptitle('Neuron Periodic Plots')
    
    for k in data.keys():
        spike = data[k]['spike']
        voltage = data[k]['voltage']
        refract = data[k]['refract']
        current = data[k]['current']
        for i in range(population_size):
            if spike[i] == True:
                spikes[i].append(i)
            else:
                spikes[i].append(None)
            voltages[i].append(voltage[i].item())
            refracts[i].append(refract[i].item())
            currents[i].append(current[i].item())
    colors = [get_random_rgb() for _ in range(population_size)]
    
    for i in range(population_size):
        axs[0].plot(time, currents[i], color=colors[i])
        axs[1].plot(time, voltages[i], color=colors[i])
        if threshold:
            axs[1].plot(time, [threshold for _ in range(steps)], linestyle='--')
        axs[2].scatter(time, spikes[i], color='royalblue', s=2)
        axs[3].plot(time, refracts[i], color=colors[i])
    axs[0].set(ylabel='Input Current')
    axs[1].set(ylabel='Membrane Potential')
    axs[2].set(ylabel='Neuron Index')
    axs[3].set(ylabel='Refractory Count', xlabel='Time')
    for i in range(4):
        axs[i].grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    