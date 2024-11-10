import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# ============================
# Parameters to play around with
# ============================
CONFIG = {
    'simulation': {
        'N_exc': 4000,                   #number of excitatory neurons in the network(4000)
        'N_inh': 1000,                   #number of inhibitory neurons in the network(1000)
        'simulation_time': 10000*ms,      #total duration of the simulation(6000*ms)
        'dt': 0.1*ms                     #simulation time step (0.1*ms)
    },
    'neuron_params_exc': {                 #parameters for excitatory neurons (Regular Spiking)
        'C': 100*pF,                     #membrane capacitance - determines the neuron's time scale (100*pF)
        'kappa': 0.7*nS/mV,             #slope factor of the I-V curve - affects spike threshold dynamics (0.7*nS/mV)
        'vr': -60*mV,                    #resting membrane potential (-60*mV)
        'vt': -40*mV,                    #threshold potential for spike initiation(-40*mV)
        'a': 0.1/ms,                    #recovery variable time constant - controls spike frequency adaptation (0.1/ms)
        'b': -2*nS,                      #coupling parameter between v and u - determines subthreshold behavior (-2*nS)
        'v_reset': -50*mV,               #post-spike reset value for membrane potential (-50*mV)
        'd': 100*pA,                     #post-spike increment of recovery variable - affects burst behavior (100*pA)
        'v_peak': 35*mV,                 #peak voltage for spike detection (35*mV)
    },
    'neuron_params_inh': {                 #parameters for inhibitory neurons (Fast Spiking)
        'C': 50*pF,                      #smaller capacitance for faster response (50*pF)
        'kappa': 1.0*nS/mV,             #steeper I-V curve for rapid firing (1.0*nS/mV)
        'vr': -60*mV,                    #slightly higher resting potential than excitatory neurons (-55*mV)
        'vt': -40*mV,                    #threshold potential (-40*mV)
        'a': 1.0/ms,                    #slower adaptation for sustained rapid firing (0.02/ms)
        'b': -0.05*nS,                   #weak v-u coupling for minimal adaptation(-0.05*nS)
        'v_reset': -50*mV,               #higher reset for faster recovery (-45*mV)
        'd': 20*pA,                       #small spike-triggered adaptation (20*pA)
        'v_peak': 25*mV,                 #lower spike peak typical of fast-spiking cells (25*mV)
    },
    'syn_params': {                        #synaptic transmission parameters
        'tau_ampa': 5*ms,                #decay time constant for AMPA receptors (fast excitation) (5*ms)
        'tau_nmda': 100*ms,              #decay time constant for NMDA receptors (slow excitation) (100*ms)
        'tau_gaba': 10*ms,               #decay time constant for GABA receptors (inhibition) (10*ms)
        'E_exc': 0*mV,                   #reversal potential for excitatory synapses (0*mV)
        'E_inh': -80*mV,                 #reversal potential for inhibitory synapses (-80*mV)
        'w_min_exc':0.5,               #minimum weight for excitatory synapses (0.5)
        'w_max_exc': 2.0,                #maximum weight for excitatory synapses (1.5)
        'w_min_inh': 0.5,               #minimum weight for inhibitory synapses (0.5)
        'w_max_inh': 2.0                 #maximum weight for inhibitory synapses (1.5)
    },
    'homeo_params': {                      #Homeostatic plasticity parameters
        'eta_exc': 1,                  #scaling of excitatory synapse homeostasis (1)
        'eta_inh': 1                   #scaling of inhibitory synapse homeostasis (1)
    },
    'plasticity_params': {                 #STDP parameters ("healthy"/"unhealthy"), must remember to remove "#" further down the code before self.stdp_setup 
        'stdp_A_plus': 0.015,             #STDP potentiation amplitude for pre-before-post spikes (0.01/0.015)
        'stdp_A_minus': 0.01,           #STDP depression amplitude for post-before-pre spikes (0.01)
        'stdp_tau_plus': 20*ms,          #time constant for STDP potentiation window (20*ms)
        'stdp_tau_minus': 20*ms          #time constant for STDP depression window (20*ms )
    },
    'connection_probs': {                  #network connection probabilities (healthy/perhaps pathological)
        'S_ee': 0.01,                    #excitatory to excitatory connection probability (0.01/0.03)
        'S_ei': 0.02,                     #excitatory to inhibitory (0.02/0.01)
        'S_ie': 0.08,                    #inhibitory to excitatory (0.08)
        'S_ii': 0.1                     #inhibitory to inhibitory (0.1)
    },
    'synaptic_weights_init': {             #initial synaptic weight parameters
        'S_ee': {'mean': 1.0,            #mean initial weight for E->E connections (1.0)
                 'std': 0.3},            #standard deviation for E->E weights (0.3)
        'S_ei': {'mean': 1.0,            #mean initial weight for E->I connections  (1.0)
                 'std': 0.3},            #standard deviation for E->I weights (0.3)
        'S_ie': {'mean':1.0,            #mean initial weight for I->E connections   (1.0)
                 'std': 0.3},           #standard deviation for I->E weights (0.3)
        'S_ii': {'mean': 1.0,            #mean initial weight for I->I connections (1.0)
                 'std': 0.3}            #standard deviation for I->I weights (0.3)
    },
    'poisson_input': {                     #background input parameters
        'rate': 90*Hz,                   #rate of random background spikes (90*Hz)
        'N_poisson': 5000,               #number of independent Poisson input sources (5000)
        'S_pe_p': 0.015,                 #connection probability to excitatory neurons (0.015)
        'S_pi_p': 0.015                 #connection probability to inhibitory neurons (0.015)
    },
    'external_input': {                    #external stimulation parameters
        'indices': np.arange(600),       #neurons receiving external input (500)
        'I_input': 100*pA               #amplitude of external current injection (60*pA)
    },
    'scaling_params': {                    #homeostatic scaling parameters
        'scale_dt': 1 *ms,                #time step for synaptic scaling updates (1*ms)
        'decay_dt': 1*ms,                #time step for spike trace decay (1*ms)
        'scale_factor_min': 0.99,         #minimum allowed scaling factor (0.99)
        'scale_factor_max': 1.01          #maximum allowed scaling factor (1.01)
    },
    'homeostatic_plasticity': {            #global control of homeostatic mechanisms
        'enabled': True                #master switch for synaptic scaling
    }
}
#
# =================================
# EpilepsyModel Class with Refactoring
# =================================

class EpilepsyModel:
    def __init__(self, config): #loading the config
        self.config = config
        b2.defaultclock.dt = self.config['simulation']['dt']
        self.N_exc = self.config['simulation']['N_exc']
        self.N_inh = self.config['simulation']['N_inh']
        self.N_total = self.N_exc + self.N_inh
        self.simulation_time = self.config['simulation']['simulation_time']
        
        #homeostatic plasticity Flag
        self.homeo_enabled = self.config['homeostatic_plasticity'].get('enabled', True)

        #neuron parameters
        self.neuron_params_exc = self.config['neuron_params_exc']
        self.neuron_params_inh = self.config['neuron_params_inh']

        #synapse parameters
        self.syn_params = self.config['syn_params']

        #homeostatic parameters
        self.homeo_params = self.config['homeo_params']

        #plasticity parameters
        self.plasticity_params = self.config['plasticity_params']

        #connection probabilities
        self.connection_probs = self.config['connection_probs']

        #synaptic weights initialization
        self.syn_weights_init = self.config['synaptic_weights_init']

        #poisson input parameters
        self.poisson_input_params = self.config['poisson_input']

        #external input parameters
        self.external_input_params = self.config['external_input']

        #scaling parameters
        self.scaling_params = self.config['scaling_params']

        #additional parameters
        self.setup_equations()

        #initialize Network
        self.net = Network()
        self.create_network()

    def setup_equations(self):
        #izhikevich neuron equations
        self.neuron_eqs = '''
        dv/dt = (kappa*(v - vr)*(v - vt) - u + I_syn + I_input)/C : volt
        du/dt = a*(b*(v - vr) - u) : amp
        I_syn = I_ampa + I_nmda + I_gaba : amp
        I_ampa = g_ampa*(E_exc - v) : amp
        I_nmda = g_nmda*(E_exc - v)/(1 + exp(-(v + 60*mV)/(5*mV))) : amp
        I_gaba = g_gaba*(E_inh - v) : amp
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_nmda/dt = -g_nmda/tau_nmda : siemens
        dg_gaba/dt = -g_gaba/tau_gaba : siemens
        I_input : amp
        spike_trace : Hz
        '''

        #rate decay and spike trace increase
        self.tau_rate = 100*ms
        self.delta_rate = 1*Hz

    def create_network(self):
        #create excitatory neuron group
        self.neurons_exc = NeuronGroup(
            self.N_exc,
            self.neuron_eqs,
            threshold='v >= v_peak',
            reset='''
            v = v_reset
            u += d
            spike_trace += delta_rate 
            ''',
            method='euler',
            namespace={**self.neuron_params_exc, **self.syn_params,
                       **self.plasticity_params, 'delta_rate': self.delta_rate}
        )

        #create inhibitory neuron group
        self.neurons_inh = NeuronGroup(
            self.N_inh,
            self.neuron_eqs,
            threshold='v >= v_peak',
            reset='''
            v = v_reset
            u += d
            spike_trace += delta_rate
            ''',
            method='euler',
            namespace={**self.neuron_params_inh, **self.syn_params,
                       **self.plasticity_params, 
                       'delta_rate': self.delta_rate}
        )
        def initialize_membrane_potentials(N_neurons, v_rest, v_thresh, noise_std=15.0):
             #initialize to slightly different subthreshold values
                v_init = v_rest + noise_std * np.random.randn(N_neurons)
                v_init = np.clip(v_init, v_rest, v_thresh-1)
                return v_init*mV
        #initialize membrane potentials and recovery variables
        self.neurons_exc.v = initialize_membrane_potentials(4000,-70,-40)
        self.neurons_exc.u = 'b*(v - vr)'
        self.neurons_exc.spike_trace = 0*Hz
        
        self.neurons_inh.v = initialize_membrane_potentials(1000,-65,-40)
        self.neurons_inh.u = 'b*(v - vr)'
        self.neurons_inh.spike_trace = 0*Hz

        


        #call synapse set up function and add neurons and synapses to network
        self.setup_synapses()
        self.net.add(self.neurons_exc, self.neurons_inh)
        self.net.add(self.S_ee, self.S_ei, self.S_ie, self.S_ii)

        #call poisson input function
        self.create_poisson_input()

        #call synaptic scaling function if enabled
        if self.homeo_enabled:
            self.setup_synaptic_scaling()

        #setup monitors
        self.setup_monitors()
        self.net.add(self.spike_mon_exc, self.spike_mon_inh,
                    self.state_mon_exc, self.state_mon_inh)

    
    def setup_synapses(self):
        #parameters for synapses
        synapse_namespace = {**self.homeo_params, **self.syn_params, **self.plasticity_params}


        #E->E connections
        self.S_ee = Synapses(
            self.neurons_exc, self.neurons_exc,
            model='''
            w_syn : 1
            dapre/dt = -apre/stdp_tau_plus : 1 (event-driven)
            dapost/dt = -apost/stdp_tau_minus : 1 (event-driven)
            ''',
            on_pre='''
            g_ampa += w_syn*0.2 * nS
            g_nmda += w_syn * 0.012 * nS
            apre += 1
            ''',
            on_post='''
            apost += 1
            ''',
            method='euler',
            namespace=synapse_namespace
        )
        self.S_ee.connect(p=self.connection_probs['S_ee'])  #connection probability
        #initialize w_syn with defined mean, std and within bounds
        self.S_ee.w_syn = np.clip(
            np.abs(self.syn_weights_init['S_ee']['mean'] +
                   self.syn_weights_init['S_ee']['std'] * np.random.randn(len(self.S_ee))),
            self.syn_params['w_min_exc'],
            self.syn_params['w_max_exc']
        )

        #E->I connections
        self.S_ei = Synapses(
            self.neurons_exc, self.neurons_inh,
            model='''
            w_syn : 1
            dapre/dt = -apre/stdp_tau_plus : 1 (event-driven)
            dapost/dt = -apost/stdp_tau_minus : 1 (event-driven)
            ''',
            on_pre='''
            g_ampa += w_syn *0.2* nS
            g_nmda += w_syn * 0.12 * nS
            apre += 1
            ''',
            on_post='''
            apost += 1
            ''',
            method='euler',
            namespace=synapse_namespace
        )
        self.S_ei.connect(p=self.connection_probs['S_ei'])  # connection probability
        # initialize w_syn with defined mean, std and within bounds
        self.S_ei.w_syn = np.clip(
            np.abs(self.syn_weights_init['S_ei']['mean'] +
                   self.syn_weights_init['S_ei']['std'] * np.random.randn(len(self.S_ei))),
            self.syn_params['w_min_exc'],
            self.syn_params['w_max_exc']
        )

        #I->E connections
        self.S_ie = Synapses(
            self.neurons_inh, self.neurons_exc,
            model='''
            w_syn : 1
            ''',
            on_pre='''
            g_gaba += w_syn*0.24 * nS
            ''',
            on_post='''
            ''',
            method='euler',
            namespace=synapse_namespace
        )
        self.S_ie.connect(p=self.connection_probs['S_ie'])  #connection probability
        #initialize w_syn with defined mean, std and within bounds
        self.S_ie.w_syn = np.clip(
            np.abs(self.syn_weights_init['S_ie']['mean'] +
                   self.syn_weights_init['S_ie']['std'] * np.random.randn(len(self.S_ie))),
            self.syn_params['w_min_inh'],
            self.syn_params['w_max_inh']
        )

        #I->I connections
        self.S_ii = Synapses(
            self.neurons_inh, self.neurons_inh,
            model='''
            w_syn : 1
            ''',
            on_pre='''
            g_gaba += w_syn*0.4 * nS
            ''',
            on_post='''
            ''',
            method='euler',
            namespace=synapse_namespace
        )
        self.S_ii.connect(p=self.connection_probs['S_ii'])  #connection probability
        #initialize w_syn with defined mean, std and within bounds
        self.S_ii.w_syn = np.clip(
            np.abs(self.syn_weights_init['S_ii']['mean'] +
                   self.syn_weights_init['S_ii']['std'] * np.random.randn(len(self.S_ii))),
            self.syn_params['w_min_inh'],
            self.syn_params['w_max_inh']
        )

        #set synaptic delays
        for syn in [self.S_ee, self.S_ei, self.S_ie, self.S_ii]:
            syn.delay = 'rand() * 2*ms'

        #implement STDP, remove "#" to implement
        #self.setup_stdp(self.S_ee)
        #self.setup_stdp(self.S_ei)

    def setup_stdp(self, synapses):
        synapses.run_regularly('''w_syn += stdp_A_plus * apost - stdp_A_minus * apre
                                      w_syn = clip(w_syn, w_min_exc, w_max_exc)''',
                                   dt=defaultclock.dt)

    def create_poisson_input(self):
        #create a PoissonGroup for background input
        self.poisson_input = PoissonGroup(
            self.poisson_input_params['N_poisson'],
            rates=self.poisson_input_params['rate']
        )
        #poisson to exc
        self.S_pe = Synapses(self.poisson_input, self.neurons_exc,
                             on_pre='g_ampa += 0.05*nS')
        self.S_pe.connect(p=self.poisson_input_params['S_pe_p'])

        #poisson to inh
        self.S_pi = Synapses(self.poisson_input, self.neurons_inh,
                             on_pre='g_ampa += 0.05*nS')
        self.S_pi.connect(p=self.poisson_input_params['S_pi_p'])

        self.net.add(self.poisson_input, self.S_pe, self.S_pi)

    #synaptic scaling and spiking trace decay
    def setup_synaptic_scaling(self):
        @network_operation(dt=self.scaling_params['scale_dt'])
        def scale_synaptic_weights():
            def sigmoid(x, k=0.5):
                return (1 / (1 + np.exp(-k * x)))+0.5
            #clipped scaling E->E synapses based on postsynaptic excitatory rate
            postsyn_rates_ee = self.neurons_exc.spike_trace[self.S_ee.j]
            theta_v_ee = 4*Hz
            norm_input_ee = (theta_v_ee - postsyn_rates_ee) / theta_v_ee
            scale_ee = sigmoid(norm_input_ee)
            scale_ee = np.clip(scale_ee, self.scaling_params['scale_factor_min'], 
                                 self.scaling_params['scale_factor_max'])
            self.S_ee.w_syn *= scale_ee * self.homeo_params['eta_exc']
            self.S_ee.w_syn = np.clip(self.S_ee.w_syn, 
                                  self.syn_params['w_min_exc'], 
                                  self.syn_params['w_max_exc'])
            
            #clipped scaling I->E synapses based on postsynaptic excitatory rate
            postsyn_rates_ie = self.neurons_exc.spike_trace[self.S_ie.j] 
            theta_v_ie = 4*Hz    
            norm_input_ie = (postsyn_rates_ie - theta_v_ie) / theta_v_ie
            scale_ie = sigmoid(norm_input_ie)
            scale_ie = np.clip(scale_ie, self.scaling_params['scale_factor_min'], 
                                 self.scaling_params['scale_factor_max'])
            self.S_ie.w_syn *= scale_ie * self.homeo_params['eta_exc']
            self.S_ie.w_syn = np.clip(self.S_ie.w_syn, 
                                  self.syn_params['w_min_exc'], 
                                  self.syn_params['w_max_exc'])
            
            #clipped scaling E-I synapses based on postsynaptci inhibitory rate
            postsyn_rates_ei = self.neurons_inh.spike_trace[self.S_ei.j]
            theta_v_ei = 14*Hz 
            norm_input_ei = (theta_v_ei - postsyn_rates_ei) / theta_v_ei
            scale_ei = sigmoid(norm_input_ei)
            scale_ei = np.clip(scale_ei, self.scaling_params['scale_factor_min'], 
                                 self.scaling_params['scale_factor_max'])
            self.S_ei.w_syn *= scale_ei * self.homeo_params['eta_exc'] 
            self.S_ei.w_syn = np.clip(self.S_ei.w_syn, 
                                  self.syn_params['w_min_exc'], 
                                  self.syn_params['w_max_exc'])
            
            #clipped scaling I->I synapses based on postsynaptic inhibitory rate
            postsyn_rates_ii = self.neurons_inh.spike_trace[self.S_ii.j]
            theta_v_ii = 14*Hz  
            norm_input_ii = (postsyn_rates_ii-theta_v_ii) / theta_v_ii
            scale_ii = sigmoid(norm_input_ii)
            scale_ii = np.clip(scale_ii, self.scaling_params['scale_factor_min'], 
                                 self.scaling_params['scale_factor_max'])
            self.S_ii.w_syn *= scale_ii * self.homeo_params['eta_inh']
            self.S_ii.w_syn = np.clip(self.S_ii.w_syn, 
                                  self.syn_params['w_min_inh'], 
                                  self.syn_params['w_max_inh'])
        self.net.add(scale_synaptic_weights)

        #continuously decaying spike trace
        @network_operation(dt=self.scaling_params['decay_dt'])
        def decay_spike_trace():
            decay_factor = exp(-defaultclock.dt / self.tau_rate)
            self.neurons_exc.spike_trace = self.neurons_exc.spike_trace * decay_factor
            self.neurons_inh.spike_trace = self.neurons_inh.spike_trace * decay_factor
        self.net.add(decay_spike_trace)


    def setup_monitors(self):
        #spike monitors
        self.spike_mon_exc = SpikeMonitor(self.neurons_exc)
        self.spike_mon_inh = SpikeMonitor(self.neurons_inh)

        #state monitors
        self.state_mon_exc = StateMonitor(
            self.neurons_exc,
            ['v', 'u', 'g_ampa', 'g_nmda', 'g_gaba', 'spike_trace'],
            record=range(0, 4000, 100)
        )
        self.state_mon_inh = StateMonitor(
            self.neurons_inh,
            ['v', 'u', 'g_ampa', 'g_nmda', 'g_gaba', 'spike_trace'],
            record=range(0, 1000, 25)
        )

        recorded_synapses = 800 

        #synaptic weight monitors for each connection type
        self.syn_mon_ee = StateMonitor(
            self.S_ee,
            'w_syn',
            record=range(0, len(self.S_ee), max(1, len(self.S_ee)//recorded_synapses))
        )
        self.syn_mon_ei = StateMonitor(
            self.S_ei,
            'w_syn',
            record=range(0, len(self.S_ei), max(1, len(self.S_ei)//recorded_synapses))
        )
        self.syn_mon_ie = StateMonitor(
            self.S_ie,
            'w_syn',
            record=range(0, len(self.S_ie), max(1, len(self.S_ie)//recorded_synapses))
        )
        self.syn_mon_ii = StateMonitor(
            self.S_ii,
            'w_syn',
            record=range(0, len(self.S_ii), max(1, len(self.S_ii)//recorded_synapses))
        )

        #add monitors to the network
        self.net.add(
            self.spike_mon_exc, self.spike_mon_inh,
            self.state_mon_exc, self.state_mon_inh,
            self.syn_mon_ee, self.syn_mon_ei, self.syn_mon_ie, self.syn_mon_ii
        )


    def run_simulation(self):
        indices = self.external_input_params['indices']
        #initial simulation period before applying external input,
        #applying small initial input for better initialisation
        self.neurons_exc.I_input = 2*pA
        self.neurons_inh.I_input = 5*pA
        self.net.run(0.001*ms)
        self.neurons_exc.I_input = 0*pA
        self.neurons_inh.I_input = 0*pA
        self.net.run(0.249*self.simulation_time)
        #apply external input current to a subset of excitatory or inhibitory neurons:
        self.neurons_exc[indices].I_input = self.external_input_params['I_input']
        #self.neurons_inh.I_input[indices] = -1*self.external_input_params['I_input']
        self.net.run(0.3*self.simulation_time)
        self.neurons_exc[indices].I_input = 0
        self.net.run(0.2*self.simulation_time)
        self.neurons_inh.I_input = 0*pA
        
        self.net.run(0.25*self.simulation_time)

    def analyze_results(self):
        #calculate population firing rates
        exc_rates = self.calculate_firing_rates(self.spike_mon_exc)
        inh_rates = self.calculate_firing_rates(self.spike_mon_inh)
        avg_w_ee = np.mean(self.syn_mon_ee.w_syn, axis=0)
        avg_w_ei = np.mean(self.syn_mon_ei.w_syn, axis=0)
        avg_w_ie = np.mean(self.syn_mon_ie.w_syn, axis=0)
        avg_w_ii = np.mean(self.syn_mon_ii.w_syn, axis=0)
        return {
            'exc_rates': exc_rates,
            'inh_rates': inh_rates,
            'avg_w_ee': avg_w_ee,
            'avg_w_ei': avg_w_ei,
            'avg_w_ie': avg_w_ie,
            'avg_w_ii': avg_w_ii,
            'time_bins_rate': np.arange(0, len(exc_rates)) * float(10*ms),
        }
    def calculate_firing_rates(self, spike_monitor, bin_size=10*ms):
        spikes = spike_monitor.spike_trains()
        simulation_duration = float(self.simulation_time)
        num_bins = int(simulation_duration / float(bin_size))

        #create time bins
        time_bins = np.linspace(0, simulation_duration, num_bins+1)

        #initialize rates
        rates = np.zeros(num_bins)
        total_spikes = np.concatenate([spike_train for spike_train in spikes.values()])

        #compute histogram
        spike_counts, _ = np.histogram(total_spikes, bins=time_bins)
        rates = spike_counts / (len(spikes) * float(bin_size))
        return rates


    def plot_results(self):
        results = self.analyze_results()
        figsize = (12, 6)
        dpi=300
        #plot 1: excitatory raster plot
        fig1, ax1 = plt.subplots(figsize=figsize,dpi=dpi)
        ax1.plot(self.spike_mon_exc.t, self.spike_mon_exc.i, '.',markersize="0.1", color='red')
        ax1.set_ylabel('Neuron Index')
        ax1.set_title('Excitatory Neurons Raster Plot')
        ax1.set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    
        #plot 2: inhibitory raster plot
        fig2, ax2 = plt.subplots(figsize=figsize, dpi=dpi)
        ax2.plot(self.spike_mon_inh.t, self.spike_mon_inh.i, '.',markersize="0.1", color='blue')
        ax2.set_ylabel('Neuron Index')
        ax2.set_title('Inhibitory Neurons Raster Plot')
        ax2.set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    
        #plot 3: voltage traces for excitatory neurons
        fig3, ax3 = plt.subplots(figsize=figsize,dpi=dpi)
        offset_exc = 100 
        for i in range(len(self.state_mon_exc.v)):
            ax3.plot(
                self.state_mon_exc.t, 
                self.state_mon_exc.v[i]/mV + (offset_exc * i),
                label='i',
                color='red',
                alpha=0.7,
                linewidth=0.3
            )
        yticks_exc = [i * offset_exc for i in range(len(self.state_mon_exc.v))]
        ax3.set_yticks(yticks_exc)
        ax3.set_yticklabels([f'Neuron {i}' for i in range(0, len(self.state_mon_exc.v))]) 
        for i in range(len(self.state_mon_exc.v)):
            ax3.axhline(y=i * offset_exc, color='gray', alpha=0.2, linestyle='--')
        ax3.set_ylabel('Membrane Potential (mV)')
        ax3.set_title('Excitatory Neuron Membrane Potentials')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        #plot 4: average firing rate per population
        fig4, ax4 = plt.subplots(figsize=figsize,dpi=dpi)
        time_rate = results['time_bins_rate']
        ax4.plot(time_rate, results['exc_rates'], 'r-', label='Excitatory')
        ax4.plot(time_rate, results['inh_rates'], 'b-', label='Inhibitory')
        ax4.set_ylabel('Firing Rate (Hz)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Population Firing Rates')
        ax4.set_ylim(0, 60)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        #plot 5: average synaptic weights for each synapse type
        fig5, ax5 = plt.subplots(figsize=figsize,dpi=dpi)
        ax5.plot(self.syn_mon_ee.t, results['avg_w_ee'], label='E→E', color='red')
        ax5.plot(self.syn_mon_ei.t, results['avg_w_ei'], label='E→I', color='firebrick')
        ax5.plot(self.syn_mon_ie.t, results['avg_w_ie'], label='I→E', color='royalblue')
        ax5.plot(self.syn_mon_ii.t, results['avg_w_ii'], label='I→I', color='blue')
        ax5.set_ylabel("Average Synaptic Weight")
        ax5.set_xlabel("Time (s)")
        ax5.set_title("Average Synaptic Weights per Connection Type")
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        #plot 6: spike traces of inhibitory and excitatory neurons    
        fig6, ax6 = plt.subplots(figsize=figsize,dpi=dpi)
        exc_spiketrace = self.state_mon_exc.spike_trace.T
        inh_spiketrace = self.state_mon_inh.spike_trace.T 
        time_array = self.state_mon_exc.t
        ax6.plot(time_array, exc_spiketrace, label='Excitatory Neurons', color='red')
        ax6.plot(time_array, inh_spiketrace, label='Inhibitory Neurons', color='blue')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Average Spike Trace (Hz)')
        ax6.set_title('Average Spike Trace of Excitatory and Inhibitory Neuron Populations')
        ax6.grid(True)
        plt.tight_layout()
        plt.show()
    
        #plot 7: layered voltage traces for inhibitory neurons
        fig7, ax7 = plt.subplots(figsize=figsize,dpi=dpi)
        offset_inh = 100  # mV between traces
        for i in range(len(self.state_mon_inh.v)):
            ax7.plot(
                self.state_mon_inh.t, 
                self.state_mon_inh.v[i]/mV + (offset_inh * i),
                label='i',
                color='blue',
                alpha=0.7,
                linewidth=0.3
            )
        yticks_inh = [i * offset_inh for i in range(len(self.state_mon_inh.v))]
        ax7.set_yticks(yticks_inh)
        ax7.set_yticklabels([f'Neuron {i}' for i in range(0, len(self.state_mon_inh.v))]) 
        for i in range(len(self.state_mon_inh.v)):
            ax7.axhline(y=i * offset_inh, color='gray', alpha=0.2, linestyle='--')
        ax7.set_ylabel('Membrane Potential (mV)')
        ax7.set_title('Inhibitory Neuron Membrane Potentials')
        ax7.set_xlabel('Time (s)')
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


#run the model with config
def run_epilepsy_simulation():
    start_scope()
    model = EpilepsyModel(CONFIG)
    print("Starting simulation...")
    model.run_simulation()
    print("Simulation completed!")
    results = model.analyze_results()
    print(f"Average excitatory rate: {np.mean(results['exc_rates']): .2f} Hz")
    print(f"Average inhibitory rate: {np.mean(results['inh_rates']): .2f} Hz")
    model.plot_results()
    return model, results


#run the model in IDE
if __name__ == "__main__":
    model, results = run_epilepsy_simulation()
