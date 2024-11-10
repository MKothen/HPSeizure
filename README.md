# HPSeizure

## Overview
**The simulation models a network with:**

- **Excitatory Neurons:** Regular spiking neurons modeled with Izhikevich equations.
- **Inhibitory Neurons:** Fast spiking neurons, also using Izhikevich models.
- **Synapses:** Conductance based connections between neurons with AMPA, NMDA (excitatory), and GABA (inhibitory) receptors.
- **Plasticity:** Homeostatic synaptic scaling to maintain stable activity.
- **External Input:** Background Poisson input and optional external stimulation to a subset of neurons.

### Customization
You can adjust the simulation by modifying the CONFIG dictionary at the beginning of the script. Parameters you might want to change include:

- **Neuron Counts:** Number of excitatory (N_exc) and inhibitory (N_inh) neurons.
- **Simulation Time:** Total duration of the simulation (simulation_time).
- **Synaptic Weights:** Initial synaptic weights and their bounds.
- **Plasticity Parameters:** Toggle plasticity mechanisms or adjust their parameters.
- **External Inputs:** Define external currents or stimuli applied to the neurons.

### Results
After running the simulation, the script will display several plots:

- **Excitatory Neurons Raster Plot:** Shows spike times of excitatory neurons.
- **Inhibitory Neurons Raster Plot:** Displays spike times of inhibitory neurons.
- **Excitatory Neuron Membrane Potentials:** Voltage traces of sampled excitatory neurons.
- **Population Firing Rates:** Average firing rates of excitatory and inhibitory populations over time.
- **Average Synaptic Weights:** Changes in synaptic weights for different connection types.
- **Spike Traces:** Average spike traces representing the activity levels.
- **Inhibitory Neuron Membrane Potentials:** Voltage traces of sampled inhibitory neurons.

Notes
The model uses simplified neuron dynamics and synaptic mechanisms to focus on key aspects of epilepsy-related activity.
Homeostatic plasticity is enabled by default but can be turned off in the configuration.
The script is organized for clarity, making it easier to understand and modify different components.
Feel free to explore and modify the code to suit your interests or to extend the simulation with additional features.

### Requirements: 
Brian2, NumPy, Matplotlib





