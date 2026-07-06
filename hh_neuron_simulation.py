"""
hh_neuron_simulation_segev.py
==============================
A single-compartment Hodgkin-Huxley (HH) neuron implemented in NEURON,
updated to match Segev et al. 2021's spike train generation methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from segev_inputs import (generate_input_spike_trains, check_per_input_rates,
                           build_population_inputs, DEFAULT_SYNAPSE_KINETICS)
from presynaptic_spike_train import PresynapticSpikeTrain
from scipy.ndimage import gaussian_filter1d


from neuron import h

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SIM_DURATION_MS = 10000
DT_MS           = 0.025
CELSIUS         = 34.0
V_INIT          = -65.0


NUM_EXC_INPUTS = 500
NUM_INH_INPUTS = 125

# soma geometry (µm), used for segment length scaling (Segev's approach)
SOMA_LENGTH_UM = 20.0
SOMA_DIAM_UM   = 20.0

SYNAPSE_PARAMS = {
    "AMPA":   DEFAULT_SYNAPSE_KINETICS["AMPA"],
    "GABA_A": DEFAULT_SYNAPSE_KINETICS["GABA_A"],
}

AMPA_WEIGHT_US  = 0.040
GABAA_WEIGHT_US = 0.003

# ── Segev-style rate parameters ──────────────────────────────────────────────

MIN_SEG_LENGTH_UM = 10.0

# units are "total spikes per compartment-tree per 100 ms".
# for a single compartment, "tree" is just the soma itself.
# these are sampled uniformly from the given ranges each simulation.
# I used Segev's NMDA ranges (from his paper, single-compartment adaptation)
NUM_EX_SPIKES_PER_100MS_RANGE  = [20, 200]
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


def build_inputs(random_seed=None):
    return build_population_inputs(
        sim_duration_ms=SIM_DURATION_MS, num_exc_inputs=NUM_EXC_INPUTS,
        num_inh_inputs=NUM_INH_INPUTS, min_seg_length_um=MIN_SEG_LENGTH_UM,
        num_ex_spikes_per_100ms_range=NUM_EX_SPIKES_PER_100MS_RANGE,
        num_ex_inh_spike_diff_per_100ms_range=NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE,
        inst_rate_interval_options_ms=INST_RATE_INTERVAL_OPTIONS_MS,
        temporal_smoothing_sigma_options_ms=TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS,
        inst_rate_interval_jitter=INST_RATE_INTERVAL_JITTER,
        temporal_smoothing_jitter=TEMPORAL_SMOOTHING_SIGMA_JITTER,
        random_seed=random_seed,
    )

def add_synapses(soma, ex_bin, inh_bin):
    synapses, spike_trains, netcons = [], [], []
    ampa_p, gabaa_p = SYNAPSE_PARAMS["AMPA"], SYNAPSE_PARAMS["GABA_A"]

    for i in range(ex_bin.shape[0]):
        syn = h.Exp2Syn(soma(0.5))
        syn.tau1, syn.tau2, syn.e = ampa_p["tau_rise_ms"], ampa_p["tau_decay_ms"], ampa_p["E_rev_mV"]
        synapses.append(syn)
        times = np.maximum(np.where(ex_bin[i, :] == 1)[0].astype(float) + 0.5, 0.1)
        nc = h.NetCon(None, syn)
        nc.delay, nc.weight[0] = 0.0, AMPA_WEIGHT_US
        netcons.append(nc)
        spike_trains.append(PresynapticSpikeTrain(times, nc))

    for i in range(inh_bin.shape[0]):
        syn = h.Exp2Syn(soma(0.5))
        syn.tau1, syn.tau2, syn.e = gabaa_p["tau_rise_ms"], gabaa_p["tau_decay_ms"], gabaa_p["E_rev_mV"]
        synapses.append(syn)
        times = np.maximum(np.where(inh_bin[i, :] == 1)[0].astype(float) + 0.5, 0.1)
        nc = h.NetCon(None, syn)
        nc.delay, nc.weight[0] = 0.0, GABAA_WEIGHT_US
        netcons.append(nc)
        spike_trains.append(PresynapticSpikeTrain(times, nc))

    return synapses, spike_trains, netcons


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(soma):
    t_vec = h.Vector(); t_vec.record(h._ref_t)
    v_vec = h.Vector(); v_vec.record(soma(0.5)._ref_v)
    apc = h.APCount(soma(0.5)); apc.thresh = -20.0
    spike_vec = h.Vector(); apc.record(spike_vec)

    h.load_file("stdrun.hoc")
    h.celsius = CELSIUS
    h.dt = DT_MS
    h.tstop = SIM_DURATION_MS
    h.v_init = V_INIT
    cvode = h.CVode(); cvode.active(0)
    h.finitialize(h.v_init)
    h.run()
    return np.array(t_vec), np.array(v_vec), np.array(spike_vec)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(t_ms, v_mv, spike_times, ex_bin, inh_bin):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 2]})
    fig.suptitle(f'HH Point Neuron -- {ex_bin.shape[0]} exc + {inh_bin.shape[0]} inh Segev-style inputs '
                 f'(NEURON built-in hh mechanism, T={CELSIUS}C)', fontsize=12)

    ax = axes[0]
    n_show = min(ex_bin.shape[0], 150)
    for i in range(n_show):
        spk_t = np.where(ex_bin[i, :] == 1)[0]
        ax.plot(spk_t, np.full_like(spk_t, i), '|', color='#2196F3', markersize=1.5, alpha=0.5)
    n_show_i = min(inh_bin.shape[0], 50)
    for i in range(n_show_i):
        spk_t = np.where(inh_bin[i, :] == 1)[0]
        ax.plot(spk_t, np.full_like(spk_t, n_show + i), '|', color='#F44336', markersize=1.5, alpha=0.5)
    ax.set_ylabel('Input index (subsample)')
    ax.set_xlim(0, ex_bin.shape[1])
    ax.set_title('Input Spike Raster (subsampled)', fontsize=10)

    ax = axes[1]
    exc_mean = ex_bin.mean(axis=0).astype(float)
    inh_mean = inh_bin.mean(axis=0).astype(float)
    # ax.plot(gaussian_filter1d(exc_mean, 50) * 1000, '#2196F3', lw=1, label='Exc mean rate (Hz)')
    # ax.plot(gaussian_filter1d(inh_mean, 50) * 1000, '#F44336', lw=1, label='Inh mean rate (Hz)')
    ax.plot(gaussian_filter1d(exc_mean - inh_mean, 50) * 1000, '#9E9E9E', lw=1, label='Exc mean rate - Inh mean rate (Hz)')
    ax.axhline(y=0, linestyle=':')
    ax.set_ylabel('Rate (Hz)'); ax.legend(fontsize=8); ax.set_xlim(0, ex_bin.shape[1])
    ax.set_title('Mean Instantaneous Input Rate', fontsize=10)

    ax = axes[2]
    ax.plot(t_ms, v_mv, 'k', lw=0.6, label='V_soma')
    if len(spike_times) > 0:
        ax.plot(spike_times, np.full_like(spike_times, 40), 'r|', markersize=10, label=f'{len(spike_times)} APs')
    ax.axhline(-65, color='grey', lw=0.5, linestyle='--', label='V_rest')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Vm (mV)')
    ax.set_xlim(0, ex_bin.shape[1]); ax.set_ylim(-80, 50); ax.legend(fontsize=8)
    ax.set_title('HH Somatic Membrane Potential', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
    out_path = 'hh_point_neuron_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    Figure saved: {out_path}")
    plt.close(fig)


def main(random_seed=7):
    print("=" * 70)
    print("HH Point-Neuron Simulation (Segev-style inputs)")
    print("=" * 70)

    print(f"\n[1] Inputs: {NUM_EXC_INPUTS} excitatory + {NUM_INH_INPUTS} inhibitory, "
          f"{SIM_DURATION_MS} ms")
    ex_bin, inh_bin = build_inputs(random_seed=random_seed)
    check_per_input_rates(ex_bin, inh_bin, SIM_DURATION_MS)

    print("\n[2] Building single-compartment HH cell ...")
    soma = build_hh_cell()

    print("\n[3] Placing synapses ...")
    synapses, spike_trains, netcons = add_synapses(soma, ex_bin, inh_bin)
    print(f"    {len(synapses)} total synapse objects "
          f"({ex_bin.shape[0]} AMPA + {inh_bin.shape[0]} GABA-A)")

    print(f"\n[4] Running simulation ({SIM_DURATION_MS} ms, dt={DT_MS} ms, T={CELSIUS}C) ...")
    t_ms, v_mv, spike_times = run_simulation(soma)
    n_aps = len(spike_times)
    print(f"    Done. Somatic APs: {n_aps} (output rate: {1000.0 * n_aps / SIM_DURATION_MS:.2f} Hz)")

    print("\n[5] Plotting ...")
    plot_results(t_ms, v_mv, spike_times, ex_bin, inh_bin)

    np.savez_compressed(
        'hh_point_neuron_data.npz', t_ms=t_ms, soma_v_mv=v_mv, spike_times_ms=spike_times,
        ex_spikes_bin=ex_bin, inh_spikes_bin=inh_bin,
    )
    print("    Saved: hh_point_neuron_data.npz")
    print("\n" + "=" * 70 + "\nDone.\n" + "=" * 70)
    return t_ms, v_mv, spike_times, ex_bin, inh_bin


if __name__ == '__main__':
    main(random_seed=7)