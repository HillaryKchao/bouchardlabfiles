"""
lif_point_neuron_simulation.py
================================
Leaky integrate-and-fire (LIF) point neuron, driven by the same Segev-style
over-dispersed input spike trains as the HH point-neuron and multi-compartment
models.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from segev_inputs import check_per_input_rates, build_population_inputs

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SIM_DURATION_MS = 10000
DT_MS           = 0.1

NUM_EXC_INPUTS = 500
NUM_INH_INPUTS = 125

# LIF membrane parameters
C_M_NF    = 0.2      # nF, membrane capacitance
G_L_US    = 0.01      # uS, leak conductance  (tau_m = C_M/G_L = 20 ms)
E_L_MV    = -65.0
V_TH_MV   = -50.0
V_RESET_MV = -65.0
T_REFRAC_MS = 2.0
E_EXC_MV  = 0.0
E_INH_MV  = -75.0

# synaptic kinetics (single-exponential conductance, per-input peak in uS)
TAU_EXC_MS = 2.0
TAU_INH_MS = 8.0
W_EXC_US = 0.0025
W_INH_US = 0.010

# Segev input-statistics knobs (identical framework to the multicompartment model)
MIN_SEG_LENGTH_UM = 10.0
NUM_EX_SPIKES_PER_100MS_RANGE         = [20, 200]
NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE = [-60, 20]
INST_RATE_INTERVAL_OPTIONS_MS = [25, 30, 35, 40, 45, 55, 60, 65, 70,
                                  75, 80, 85, 90, 100, 150, 200, 300, 450]
INST_RATE_INTERVAL_JITTER = 20
TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS = [25, 30, 35, 40, 50, 60, 80,
                                        100, 150, 200, 300, 400, 500, 600]
TEMPORAL_SMOOTHING_SIGMA_JITTER = 20


def build_inputs(random_seed=None):
    """Generate 1-ms-binned spike trains for the exc/inh input populations."""
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


def run_lif(ex_bin_1ms, inh_bin_1ms):
    n_steps = int(SIM_DURATION_MS / DT_MS)
    t_ms = np.arange(n_steps) * DT_MS

    # population spike counts per 1-ms bin -> upsample (repeat) onto the DT grid
    steps_per_ms = int(round(1.0 / DT_MS))
    ex_count_1ms = ex_bin_1ms.sum(axis=0).astype(float)   # (SIM_DURATION_MS,)
    inh_count_1ms = inh_bin_1ms.sum(axis=0).astype(float)
    ex_count = np.repeat(ex_count_1ms, steps_per_ms)[:n_steps] / steps_per_ms
    inh_count = np.repeat(inh_count_1ms, steps_per_ms)[:n_steps] / steps_per_ms

    v = np.full(n_steps, E_L_MV)
    g_exc = 0.0
    g_inh = 0.0
    spike_times = []
    refrac_until = -1.0
    decay_exc = np.exp(-DT_MS / TAU_EXC_MS)
    decay_inh = np.exp(-DT_MS / TAU_INH_MS)

    v_now = E_L_MV
    for i in range(1, n_steps):
        g_exc = g_exc * decay_exc + W_EXC_US * ex_count[i]
        g_inh = g_inh * decay_inh + W_INH_US * inh_count[i]

        if t_ms[i] < refrac_until:
            v_now = V_RESET_MV
        else:
            i_leak = G_L_US * (E_L_MV - v_now)
            i_syn = g_exc * (E_EXC_MV - v_now) + g_inh * (E_INH_MV - v_now)
            dv = (i_leak + i_syn) / C_M_NF * DT_MS
            v_now = v_now + dv
            if v_now >= V_TH_MV:
                spike_times.append(t_ms[i])
                v_now = V_RESET_MV
                refrac_until = t_ms[i] + T_REFRAC_MS
        v[i] = v_now

    return t_ms, v, np.array(spike_times), ex_count, inh_count


def plot_results(t_ms, v, spike_times, ex_bin_1ms, inh_bin_1ms, ex_count, inh_count):
    fig, axes = plt.subplots(5, 1, figsize=(14, 12),
                              gridspec_kw={'height_ratios': [2, 1, 1, 2, 2.5]})
    fig.suptitle(f'LIF Point Neuron -- {NUM_EXC_INPUTS} exc + {NUM_INH_INPUTS} inh Segev-style inputs',
                  fontsize=12)

    ax = axes[0]
    n_show = min(NUM_EXC_INPUTS, 150)
    for i in range(n_show):
        spk_t = np.where(ex_bin_1ms[i, :] == 1)[0]
        ax.plot(spk_t, np.full_like(spk_t, i), '|', color='#2196F3', markersize=1.5, alpha=0.5)
    n_show_i = min(NUM_INH_INPUTS, 50)
    for i in range(n_show_i):
        spk_t = np.where(inh_bin_1ms[i, :] == 1)[0]
        ax.plot(spk_t, np.full_like(spk_t, n_show + i), '|', color='#F44336', markersize=1.5, alpha=0.5)
    ax.set_ylabel('Input index (subsample)')
    ax.set_xlim(0, SIM_DURATION_MS)
    ax.set_title('Input Spike Raster (subsampled)', fontsize=10)

    ax = axes[1]
    t_bins = np.arange(SIM_DURATION_MS)
    ax.plot(t_bins, ex_count[::int(round(1/DT_MS))][:SIM_DURATION_MS] * 1000, '#2196F3', lw=1, label='Exc pop. rate (Hz-equiv)')
    ax.set_ylabel('Rate proxy'); ax.legend(fontsize=8); ax.set_xlim(0, SIM_DURATION_MS)
    ax.set_title('Excitatory Population Drive', fontsize=10)

    ax = axes[2]
    ax.plot(t_bins, inh_count[::int(round(1/DT_MS))][:SIM_DURATION_MS] * 1000, '#F44336', lw=1, label='Inh pop. rate (Hz-equiv)')
    ax.set_ylabel('Rate proxy'); ax.legend(fontsize=8); ax.set_xlim(0, SIM_DURATION_MS)
    ax.set_xlabel('Time (ms)')
    ax.set_title('Inhibitory Population Drive', fontsize=10)
    
    ax = axes[3]
    exc_mean = ex_bin_1ms.mean(axis=0).astype(float)
    inh_mean = inh_bin_1ms.mean(axis=0).astype(float)
    ax.plot(gaussian_filter1d(exc_mean - inh_mean, 50) * 1000, '#9E9E9E', lw=1,
            label='Exc mean rate - Inh mean rate (Hz)')
    ax.set_ylabel('Rate (Hz)')
    ax.axhline(y=0, linestyle=':')
    ax.legend(fontsize=8)
    ax.set_xlim(0, SIM_DURATION_MS)
    ax.set_title('Mean Instantaneous Input Rate', fontsize=10)

    ax = axes[4]
    ax.plot(t_ms, v, 'k', lw=0.6, label='V_m')
    if len(spike_times) > 0:
        ax.plot(spike_times, np.full_like(spike_times, V_TH_MV + 10), 'r|', markersize=10,
                label=f'{len(spike_times)} APs')
    ax.axhline(V_TH_MV, color='grey', lw=0.5, linestyle='--', label='threshold')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Vm (mV)')
    ax.set_xlim(0, SIM_DURATION_MS); ax.legend(fontsize=8)
    ax.set_title('LIF Somatic Membrane Potential', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
    out_path = 'lif_point_neuron_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    Figure saved: {out_path}")
    plt.close(fig)


def main(random_seed=7):
    print("=" * 70)
    print("LIF Point-Neuron Simulation (Segev-style inputs)")
    print("=" * 70)
    print(f"\n[1] Inputs: {NUM_EXC_INPUTS} excitatory + {NUM_INH_INPUTS} inhibitory, "
          f"{SIM_DURATION_MS} ms")
    ex_bin_1ms, inh_bin_1ms = build_inputs(random_seed=random_seed)
    check_per_input_rates(ex_bin_1ms, inh_bin_1ms, SIM_DURATION_MS)

    print("\n[2] Running LIF dynamics ...")
    t_ms, v, spike_times, ex_count, inh_count = run_lif(ex_bin_1ms, inh_bin_1ms)
    n_aps = len(spike_times)
    out_hz = 1000.0 * n_aps / SIM_DURATION_MS
    print(f"    Done. Somatic APs: {n_aps} (output rate: {out_hz:.2f} Hz)")

    print("\n[3] Plotting ...")
    plot_results(t_ms, v, spike_times, ex_bin_1ms, inh_bin_1ms, ex_count, inh_count)

    np.savez_compressed(
        'lif_point_neuron_data.npz', t_ms=t_ms, v_mv=v, spike_times_ms=spike_times,
        ex_spikes_bin=ex_bin_1ms, inh_spikes_bin=inh_bin_1ms,
    )
    print("    Saved: lif_point_neuron_data.npz")
    print("\n" + "=" * 70 + "\nDone.\n" + "=" * 70)
    return t_ms, v, spike_times, ex_bin_1ms, inh_bin_1ms


if __name__ == '__main__':
    main(random_seed=7)