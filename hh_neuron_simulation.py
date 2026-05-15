"""
hh_neuron_simulation_segev.py
==============================
A single-compartment Hodgkin-Huxley (HH) neuron implemented in NEURON,
updated to match Segev et al. 2021's spike train generation methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import fftconvolve
from neuron import h, gui

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SIM_DURATION_MS = 1000
DT_MS = 0.025

# one excitatory and one inhibitory synapse per segment (currently single compartment)
N_SEGMENTS = 1

# soma geometry (µm), used for segment length scaling (Segev's approach)
SOMA_LENGTH_UM = 20.0
SOMA_DIAM_UM   = 20.0

# ── Segev-style rate parameters ──────────────────────────────────────────────
# units are "total spikes per compartment-tree per 100 ms".
# for a single compartment, "tree" is just the soma itself.
# these are sampled uniformly from the given ranges each simulation.
# I used Segev's NMDA ranges (from his paper, single-compartment adaptation)
NUM_EX_SPIKES_PER_100MS_RANGE  = [0, 80]    # max ≈ 800 Hz for single seg
NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE = [-60, 20]  # inh relative to exc

# Rate envelope timing (ms): how long each "epoch" of constant rate lasts
# before a new rate is drawn. Segev used a broad discrete options list.
INST_RATE_INTERVAL_OPTIONS_MS = [25, 30, 35, 40, 45, 55, 60, 65, 70,
                                  75, 80, 85, 90, 100, 150, 200, 300, 450]
INST_RATE_INTERVAL_JITTER     = 20   # ± ms of uniform jitter added to interval

# Gaussian temporal smoothing sigma options (ms) applied to the rate envelope.
TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS = [25, 30, 35, 40, 50, 60, 80,
                                        100, 150, 200, 300, 400, 500, 600]
TEMPORAL_SMOOTHING_SIGMA_JITTER     = 20   # ± ms of uniform jitter

# "Regularization" floor for segment length (Segev's min_seg_length_um).
# Prevents division by very small numbers in multi-compartment extensions.
MIN_SEG_LENGTH_UM = 10.0

# Weight scale — kept as 1.0 for single compartment.
# In multi-compartment: weight_scale = seg_area_um2 / reference_area_um2.
WEIGHT_SCALE = 1.0

# Base synaptic weights (µS)
AMPA_WEIGHT_US  = 0.001
GABAA_WEIGHT_US = 0.002


# ─────────────────────────────────────────────────────────────────────────────
# SPIKE TRAIN GENERATION (based on Segev's version)
#
#  tldr of how it works:
#    1) coarse rate epochs:
#      num_ex_spikes_per_100ms = np.random.uniform(low=..., high=...,
#                                                   size=(1, num_inst_rate_samples))
#      sampled freshly every ~50–300 ms interval which creates non-stationarity
#
#    2) inhibitory rate locked to excitatory:
#      num_inh_low  = max(0, num_ex + diff_range[0])
#      num_inh_high = num_ex + diff_range[1]
#      num_inh = np.random.uniform(low=num_inh_low, high=num_inh_high, ...)
#      inhibitory is always near excitatory (E/I balance), not independent!
#
#    3) convert to per-segment per-ms rate, scaled by segment length:
#      rate_per_seg = (num_per_100ms / total_tree_length) * seg_length
#      thus larger segments get proportionally higher drive
#
#    4) Gaussian temporal smoothing, which makes the rate envelope vary smoothly over time
#      smoothed = signal.convolve(rate_per_seg, gaussian_window, mode='same')
#
#    5) over-dispersed instantaneous probability:
#      inst_prob = np.random.exponential(scale=smoothed_rate) <--- VERY IMPORTANT!!!
#      spikes    = np.random.rand(...) < inst_prob
#      exponential draw makes Fano factor > 1, matching cortical in-vivo data
#
#    6) returns TWO SEPARATE matrices:
#      return ex_spikes_bin, inh_spikes_bin
#      shape of each is (N_segments, T), vs one combined matrix that chad has
# ─────────────────────────────────────────────────────────────────────────────

def generate_input_spike_trains_for_simulation(
    sim_duration_ms: int,
    seg_length_um: np.ndarray,
    min_seg_length_um: float,
    num_ex_spikes_per_100ms_range: list,
    num_ex_inh_spike_diff_per_100ms_range: list,
    inst_rate_interval_options_ms: list,
    temporal_smoothing_sigma_options_ms: list,
    inst_rate_interval_jitter: int = 20,
    temporal_smoothing_jitter: int = 20,
    random_seed: int = None,
) -> tuple:
    """
    Generate excitatory and inhibitory spike train matrices directlyi following 
    Segev's generate_input_spike_trains_for_simulation() methodology.

    Adapted for a general list of segments (rather than separate basal/apical 
    trees). For a single-compartment soma, seg_length_um = np.array([SOMA_LENGTH_UM]).

    Parameters
    ----------
    sim_duration_ms           : total simulation length (ms)
    seg_length_um             : 1-D array of segment lengths (µm)
    min_seg_length_um         : regularization floor added to each seg length
    num_ex_spikes_per_100ms_range    : [low, high] for exc rate epochs
    num_ex_inh_spike_diff_per_100ms_range : [low, high] offset from exc rate
                                            to set inh rate range
    inst_rate_interval_options_ms    : list of possible epoch durations (ms)
    temporal_smoothing_sigma_options_ms  : list of possible Gaussian sigmas (ms)
    inst_rate_interval_jitter : +/- ms added to sampled epoch duration
    temporal_smoothing_jitter : +/- ms added to sampled Gaussian sigma
    random_seed               : for reproducibility

    Returns
    -------
    ex_spikes_bin  : uint8 array of shape (N_segments, sim_duration_ms)
    inh_spikes_bin : uint8 array of shape (N_segments, sim_duration_ms)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_segments = len(seg_length_um)
    
    

    # segment length regularization, Segev adds min_seg_length_um to prevent 
    # very short segments from getting negligible drive and to avoid division 
    # by near-zero values
    adjusted_length_um    = min_seg_length_um + seg_length_um
    total_tree_length_um  = adjusted_length_um.sum()

    # sample epoch duration and smoothing sigma, where both are drawn from
    # discrete option lists with added uniform jitter. this prevents the 
    # synthetic data from having stereotyped temporal structure.
    keep_rate_const_ms = inst_rate_interval_options_ms[
        np.random.randint(len(inst_rate_interval_options_ms))
    ]
    keep_rate_const_ms += int(
        2 * inst_rate_interval_jitter * np.random.rand() - inst_rate_interval_jitter
    )
    keep_rate_const_ms = max(keep_rate_const_ms, 1)   # ensure >= 1 ms

    smoothing_sigma_ms = temporal_smoothing_sigma_options_ms[
        np.random.randint(len(temporal_smoothing_sigma_options_ms))
    ]
    smoothing_sigma_ms += int(
        2 * temporal_smoothing_jitter * np.random.rand() - temporal_smoothing_jitter
    )
    smoothing_sigma_ms = max(smoothing_sigma_ms, 1)

    # coarse rate epochs, sample the number of spikes per 100 ms for the 
    # WHOLE tree at each epoch. could think of unit as: "total spikes across 
    # the entire compartment tree per 100 ms".
    num_epochs = int(np.ceil(sim_duration_ms / keep_rate_const_ms))

    num_ex_per_100ms = np.random.uniform(
        low=num_ex_spikes_per_100ms_range[0],
        high=num_ex_spikes_per_100ms_range[1],
        size=(1, num_epochs),
    )

    # inhibitory range is defined RELATIVE to excitatory (E/I balance)
    num_inh_low  = np.maximum(0, num_ex_per_100ms + num_ex_inh_spike_diff_per_100ms_range[0])
    num_inh_high = num_ex_per_100ms + num_ex_inh_spike_diff_per_100ms_range[1]
    num_inh_per_100ms = np.random.uniform(low=num_inh_low, high=num_inh_high,
                                          size=(1, num_epochs))

    # convert to per-segment per-ms rate
    ex_rate_per_um_per_ms  = num_ex_per_100ms  / (total_tree_length_um * 100.0)
    inh_rate_per_um_per_ms = num_inh_per_100ms / (total_tree_length_um * 100.0)

    ex_rate_per_seg  = np.kron(ex_rate_per_um_per_ms,  np.ones((num_segments, 1)))
    inh_rate_per_seg = np.kron(inh_rate_per_um_per_ms, np.ones((num_segments, 1)))

    # multiply each segment by its length where larger segments get more drive
    ex_rate_per_seg  *= np.tile(adjusted_length_um[:, np.newaxis],
                                [1, ex_rate_per_seg.shape[1]])
    inh_rate_per_seg *= np.tile(adjusted_length_um[:, np.newaxis],
                                [1, inh_rate_per_seg.shape[1]])

    # add spatial multiplicative noise (+/- 50% uniform)
    ex_rate_per_seg  *= np.random.uniform(0.5, 1.5, size=ex_rate_per_seg.shape)
    inh_rate_per_seg *= np.random.uniform(0.5, 1.5, size=inh_rate_per_seg.shape)

    # kron to 1-ms time bins (then crop to sim_duration_ms)
    ex_rate_per_seg  = np.kron(ex_rate_per_seg,
                               np.ones((1, keep_rate_const_ms)))[:, :sim_duration_ms]
    inh_rate_per_seg = np.kron(inh_rate_per_seg,
                               np.ones((1, keep_rate_const_ms)))[:, :sim_duration_ms]

    # Gaussian temporal smoothing
    win_len = 1 + 7 * smoothing_sigma_ms
    smoothing_window = signal.windows.gaussian(win_len, std=smoothing_sigma_ms)[np.newaxis, :]
    smoothing_window /= smoothing_window.sum()

    ex_rate_smoothed  = signal.convolve(ex_rate_per_seg,  smoothing_window, mode='same')
    inh_rate_smoothed = signal.convolve(inh_rate_per_seg, smoothing_window, mode='same')
    ex_rate_smoothed  = np.clip(ex_rate_smoothed,  0, None)
    inh_rate_smoothed = np.clip(inh_rate_smoothed, 0, None)

    # sample spikes with exponential over-dispersion
    ex_inst_prob  = np.random.exponential(scale=ex_rate_smoothed)
    inh_inst_prob = np.random.exponential(scale=inh_rate_smoothed)

    ex_spikes_bin  = (np.random.rand(*ex_inst_prob.shape)  < ex_inst_prob).astype(np.uint8)
    inh_spikes_bin = (np.random.rand(*inh_inst_prob.shape) < inh_inst_prob).astype(np.uint8)

    return ex_spikes_bin, inh_spikes_bin


# ─────────────────────────────────────────────────────────────────────────────
# SYNAPTIC KERNEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

SYNAPSE_PARAMS = {
    'AMPA': {
        'tau_rise_ms':  0.2,
        'tau_decay_ms': 2.0,
        'E_rev_mV':     0.0,
    },
    'NMDA': {
        'tau_rise_ms':  1.0,
        'tau_decay_ms': 50.0,
        'E_rev_mV':     0.0,
    },
    'GABA_A': {
        'tau_rise_ms':  0.5,
        'tau_decay_ms': 8.0,
        'E_rev_mV':    -75.0,
    },
}


def double_exp_kernel(tau_rise_ms, tau_decay_ms, dt_ms=0.025,
                      kernel_duration_ms=150.0):
    t   = np.arange(0, kernel_duration_ms, dt_ms)
    raw = np.exp(-t / tau_decay_ms) - np.exp(-t / tau_rise_ms)
    raw[raw < 0] = 0
    peak = raw.max()
    return raw / peak if peak > 0 else raw


# ─────────────────────────────────────────────────────────────────────────────
# NEURON MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_hh_cell():
    soma = h.Section(name='soma')
    soma.L    = SOMA_LENGTH_UM
    soma.diam = SOMA_DIAM_UM
    soma.nseg = 1
    soma.cm   = 1.0
    soma.Ra   = 100.0
    soma.insert('hh')
    return soma


class PresynapticSpikeTrain:
    """
    Delivers a precomputed spike train via h.NetCon.event() + FInitializeHandler.
    No VecStim / vecevent.mod required.
    """
    def __init__(self, spike_times_ms, netcon):
        self.spike_times_ms = spike_times_ms
        self.nc = netcon
        self._fih = h.FInitializeHandler(self._schedule_events)

    def _schedule_events(self):
        for t in self.spike_times_ms:
            self.nc.event(t)


def add_synapses_segev(soma, ex_spikes_bin, inh_spikes_bin,
                       exc_synapse_type='AMPA', weight_scale=1.0):
    """
    Attach synapses to the soma following Segev's per-segment approach.

    For the single-compartment case there is only one segment (the soma),
    so this collapses to 1 excitatory + 1 inhibitory synapse, but the loop structure
    matches Segev's so I should be able to extend to multi-compartment directly.

    Parameters
    ----------
    soma             : NEURON Section
    ex_spikes_bin    : uint8 (N_segments, T) — Segev's separate exc matrix
    inh_spikes_bin   : uint8 (N_segments, T) — Segev's separate inh matrix
    exc_synapse_type : 'AMPA' or 'NMDA'
    weight_scale     : area-based weight multiplier

    Returns
    -------
    synapses, spike_trains, netcons : lists (keep refs to prevent GC)
    """
    n_segments = ex_spikes_bin.shape[0]
    assert inh_spikes_bin.shape[0] == n_segments

    synapses     = []
    spike_trains = []
    netcons      = []

    # Segev's loop: "for segInd, segment in enumerate(allSegments)"
    # however we set allSegments = [soma(0.5)] since we have one compartment
    for segInd in range(n_segments):
        segment_loc = soma(0.5)   # midpoint of soma, I'll extend for multi-compartment

        # excitatory synapse (AMPAR or NMDAR)
        exc_params = SYNAPSE_PARAMS[exc_synapse_type]
        exc_syn = h.Exp2Syn(segment_loc)
        exc_syn.tau1 = exc_params['tau_rise_ms']
        exc_syn.tau2 = exc_params['tau_decay_ms']
        exc_syn.e    = exc_params['E_rev_mV']
        synapses.append(exc_syn)

        exc_bins       = np.where(ex_spikes_bin[segInd, :] == 1)[0]
        exc_times_ms   = np.maximum(exc_bins.astype(float) + 0.5, 0.1)
        exc_nc = h.NetCon(None, exc_syn)
        exc_nc.delay    = 0.0
        exc_nc.weight[0] = AMPA_WEIGHT_US * weight_scale
        netcons.append(exc_nc)
        spike_trains.append(PresynapticSpikeTrain(exc_times_ms, exc_nc))

        # inhibitory synapse (GABA)
        inh_params = SYNAPSE_PARAMS['GABA_A']
        inh_syn = h.Exp2Syn(segment_loc)
        inh_syn.tau1 = inh_params['tau_rise_ms']
        inh_syn.tau2 = inh_params['tau_decay_ms']
        inh_syn.e    = inh_params['E_rev_mV']
        synapses.append(inh_syn)

        inh_bins       = np.where(inh_spikes_bin[segInd, :] == 1)[0]
        inh_times_ms   = np.maximum(inh_bins.astype(float) + 0.5, 0.1)
        inh_nc = h.NetCon(None, inh_syn)
        inh_nc.delay    = 0.0
        inh_nc.weight[0] = GABAA_WEIGHT_US * weight_scale
        netcons.append(inh_nc)
        spike_trains.append(PresynapticSpikeTrain(inh_times_ms, inh_nc))

    return synapses, spike_trains, netcons


# ─────────────────────────────────────────────────────────────────────────────
# RECORDING
# ─────────────────────────────────────────────────────────────────────────────

def setup_recording(soma):
    t_vec = h.Vector()
    v_vec = h.Vector()
    t_vec.record(h._ref_t)
    v_vec.record(soma(0.5)._ref_v)

    apc = h.APCount(soma(0.5))
    apc.thresh = -30.0
    spike_vec  = h.Vector()
    apc.record(spike_vec)

    return t_vec, v_vec, spike_vec, apc


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(sim_duration_ms, dt_ms):
    h.load_file('stdrun.hoc')
    h.dt    = dt_ms
    h.tstop = sim_duration_ms
    h.v_init = -65.0
    cvode = h.CVode()
    cvode.active(0)
    h.finitialize(h.v_init)
    h.run()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICAL KERNEL CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_conductance_waveform(spike_train_1ms, synapse_type, dt_ms=0.025):
    params = SYNAPSE_PARAMS[synapse_type]
    kernel = double_exp_kernel(
        tau_rise_ms=params['tau_rise_ms'],
        tau_decay_ms=params['tau_decay_ms'],
        dt_ms=dt_ms,
    )
    upsample = int(round(1.0 / dt_ms))
    spike_up = np.repeat(spike_train_1ms, upsample).astype(float)
    conv = fftconvolve(spike_up, kernel, mode='full')
    return conv[:len(spike_up)]


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(t_ms, v_mv, spike_times, ex_spikes_bin, inh_spikes_bin,
                 exc_synapse_type='AMPA', dt_ms=0.025):
    """
    Four-panel figure adapted for Segev's two-matrix input structure.
    Panels 1 and 3 now show the exc and inh matrices separately,
    matching Segev's CreateCombinedColorImage() convention.
    """
    t_ms   = np.array(t_ms)
    v_mv   = np.array(v_mv)
    spikes = np.array(spike_times)
    sim_dur = ex_spikes_bin.shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14),
                             gridspec_kw={'height_ratios': [2, 1.5, 1, 2]})
    fig.suptitle('HH Neuron Simulation (Segev-style inputs)\n'
                 f'({N_SEGMENTS} segment(s), {exc_synapse_type} + GABA_A, '
                 f'over-dispersed Poisson, E/I correlated)', fontsize=12)

    # panel 1: input raster (separated excitatory/inhibitory matrices)
    ax = axes[0]
    t_bins = np.arange(sim_dur)
    # excitatory
    for segInd in range(N_SEGMENTS):
        spk_t = t_bins[ex_spikes_bin[segInd, :] == 1]
        ax.plot(spk_t, np.full_like(spk_t, segInd), '|',
                color='#2196F3', markersize=3, linewidth=0.6, alpha=0.8)
    # inhibitory (offset by N_SEGMENTS in y for clarity)
    for segInd in range(N_SEGMENTS):
        spk_t = t_bins[inh_spikes_bin[segInd, :] == 1]
        ax.plot(spk_t, np.full_like(spk_t, segInd + N_SEGMENTS), '|',
                color='#F44336', markersize=3, linewidth=0.6, alpha=0.8)

    ax.axhline(N_SEGMENTS - 0.5, color='k', lw=0.8, linestyle='--')
    ax.set_ylabel('Segment index', fontsize=10)
    ax.set_xlim(0, sim_dur)
    ax.set_ylim(-1, 2 * N_SEGMENTS)
    ax.text(5, N_SEGMENTS * 0.3, 'Excitatory (ex_spikes_bin)', fontsize=8, color='#2196F3')
    ax.text(5, N_SEGMENTS + N_SEGMENTS * 0.3, 'Inhibitory (inh_spikes_bin)',
            fontsize=8, color='#F44336')
    ax.set_title('Input Spike Raster — Segev-style (two separate matrices)', fontsize=10)
    ax.set_xlabel('Time (ms)')

    # panel 2: synaptic conductance kernels
    ax = axes[1]
    t_k = np.arange(0, 100, dt_ms)
    ampa_k  = double_exp_kernel(
        SYNAPSE_PARAMS['AMPA']['tau_rise_ms'],
        SYNAPSE_PARAMS['AMPA']['tau_decay_ms'],
        dt_ms=dt_ms, kernel_duration_ms=100.0)
    gabaa_k = double_exp_kernel(
        SYNAPSE_PARAMS['GABA_A']['tau_rise_ms'],
        SYNAPSE_PARAMS['GABA_A']['tau_decay_ms'],
        dt_ms=dt_ms, kernel_duration_ms=100.0)
    ax.plot(t_k[:len(ampa_k)],  ampa_k,  '#2196F3', lw=2, label='AMPA kernel')
    ax.plot(t_k[:len(gabaa_k)], gabaa_k, '#F44336', lw=2, label='GABA_A kernel')
    ax.set_xlabel('Time after spike (ms)')
    ax.set_ylabel('Normalized conductance')
    ax.set_title('Synaptic Conductance Kernels (double-exponential)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 80)

    # panel 3: instantaneous firing rate (smoothed) from each matrix
    ax = axes[2]
    from scipy.ndimage import gaussian_filter1d
    exc_rate = ex_spikes_bin[0, :].astype(float)
    inh_rate = inh_spikes_bin[0, :].astype(float)
    sigma_ms = 50
    ax.plot(np.arange(sim_dur), gaussian_filter1d(exc_rate, sigma_ms) * 1000,
            '#2196F3', lw=1.5, label='Exc inst. rate (Hz)')
    ax.plot(np.arange(sim_dur), gaussian_filter1d(inh_rate, sigma_ms) * 1000,
            '#F44336', lw=1.5, label='Inh inst. rate (Hz)')
    ax.set_ylabel('Inst. rate (Hz)')
    ax.set_xlim(0, sim_dur)
    ax.set_title('Smoothed Instantaneous Firing Rates (non-stationary, over-dispersed)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel('Time (ms)')

    # panel 4: somatic membrane potential
    ax = axes[3]
    ax.plot(t_ms, v_mv, 'k', lw=0.8, label='V_soma (mV)')
    if len(spikes) > 0:
        ax.plot(spikes, np.full_like(spikes, 40), 'r|', markersize=10,
                label=f'{len(spikes)} spikes')
    ax.axhline(-65, color='grey', lw=0.5, linestyle='--', label='V_rest')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane potential (mV)')
    ax.set_title('Somatic Membrane Potential (HH Model Output)', fontsize=10)
    ax.set_xlim(0, sim_dur)
    ax.set_ylim(-80, 50)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('hh_simulation_results_segev.png', dpi=150, bbox_inches='tight')
    print("Figure saved: hh_simulation_results_segev.png")
    plt.show()


def main():
    print("=" * 60)
    print("HH Neuron Simulation — Segev-style inputs")
    print("=" * 60)

    # segment lengths for single compartment (just the soma)
    # for multi-compartment, I'll prob replace with the full allSegmentsLength array
    seg_length_um = np.array([SOMA_LENGTH_UM])

    print(f"\n[1] Generating Segev-style spike trains "
          f"({N_SEGMENTS} segment(s), {SIM_DURATION_MS} ms)")
    print(f"    Exc spikes/100ms range: {NUM_EX_SPIKES_PER_100MS_RANGE}")
    print(f"    E/I diff range:         {NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE}")

    ex_spikes_bin, inh_spikes_bin = generate_input_spike_trains_for_simulation(
        sim_duration_ms=SIM_DURATION_MS,
        seg_length_um=seg_length_um,
        min_seg_length_um=MIN_SEG_LENGTH_UM,
        num_ex_spikes_per_100ms_range=NUM_EX_SPIKES_PER_100MS_RANGE,
        num_ex_inh_spike_diff_per_100ms_range=NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE,
        inst_rate_interval_options_ms=INST_RATE_INTERVAL_OPTIONS_MS,
        temporal_smoothing_sigma_options_ms=TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS,
        inst_rate_interval_jitter=INST_RATE_INTERVAL_JITTER,
        temporal_smoothing_jitter=TEMPORAL_SMOOTHING_SIGMA_JITTER,
        random_seed=42,
    )

    print(f"    ex_spikes_bin shape:  {ex_spikes_bin.shape}   "
          f"(total spikes: {ex_spikes_bin.sum()})")
    print(f"    inh_spikes_bin shape: {inh_spikes_bin.shape}  "
          f"(total spikes: {inh_spikes_bin.sum()})")
    print(f"    Note: two separate matrices — not one combined matrix (Chad)")

    print("\n[2] Building HH soma in NEURON...")
    soma = build_hh_cell()

    print("[3] Adding synapses (Segev per-segment: 1 exc + 1 inh per segment)...")
    synapses, spike_trains, netcons = add_synapses_segev(
        soma=soma,
        ex_spikes_bin=ex_spikes_bin,
        inh_spikes_bin=inh_spikes_bin,
        exc_synapse_type='AMPA',
        weight_scale=WEIGHT_SCALE,
    )
    print(f"    {len(synapses)} synapses attached ({N_SEGMENTS} exc + {N_SEGMENTS} inh).")

    print("[4] Setting up recording vectors...")
    t_vec, v_vec, spike_vec, apc = setup_recording(soma)

    print(f"[5] Running simulation for {SIM_DURATION_MS} ms (dt={DT_MS} ms)...")
    run_simulation(SIM_DURATION_MS, DT_MS)
    print(f"    Done. Somatic spikes: {int(spike_vec.size())}")

    t_ms  = np.array(t_vec)
    v_mv  = np.array(v_vec)
    spk_t = np.array(spike_vec)
    print(f"    Output firing rate: {1000.0 * len(spk_t) / SIM_DURATION_MS:.2f} Hz")

    print("[6] Plotting results...")
    plot_results(
        t_ms=t_ms,
        v_mv=v_mv,
        spike_times=spk_t,
        ex_spikes_bin=ex_spikes_bin,
        inh_spikes_bin=inh_spikes_bin,
        exc_synapse_type='AMPA',
        dt_ms=DT_MS,
    )

    print("[7] Saving data to hh_simulation_data_segev.npz ...")
    np.savez_compressed(
        'hh_simulation_data_segev.npz',
        t_ms=t_ms,
        v_mv=v_mv,
        somatic_spike_times=spk_t,
        ex_spikes_bin=ex_spikes_bin,
        inh_spikes_bin=inh_spikes_bin,
    )
    print("    Saved.")

    print("\nDone. Key outputs:")
    print("  ex_spikes_bin   — Segev exc input matrix (N_segments x T)")
    print("  inh_spikes_bin  — Segev inh input matrix (N_segments x T)")
    print("  t_ms, v_mv      — time and somatic voltage traces")
    print("  somatic_spike_times — detected output spike times")
    return ex_spikes_bin, inh_spikes_bin, t_ms, v_mv, spk_t


if __name__ == '__main__':
    ex_spikes_bin, inh_spikes_bin, t_ms, v_mv, somatic_spikes = main()