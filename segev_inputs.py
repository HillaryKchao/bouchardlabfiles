"""
segev_inputs.py
================
Shared Segev-style (over-dispersed, temporally correlated, E/I-correlated)
input spike train generator.

`num_inputs` replaces the multicompartment model's per-segment dendritic
length array: pass real segment lengths for the multicompartment cell,
or `np.ones(num_inputs)` for a point neuron where every input is nominally
equivalent (no dendritic weighting).
"""

import numpy as np
from scipy import signal
from scipy.signal.windows import gaussian

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
def generate_input_spike_trains(
    sim_duration_ms, seg_length_um, min_seg_length_um,
    num_ex_spikes_per_100ms_range, num_ex_inh_spike_diff_per_100ms_range,
    inst_rate_interval_options_ms, temporal_smoothing_sigma_options_ms,
    inst_rate_interval_jitter=20, temporal_smoothing_jitter=20, random_seed=None,
    total_tree_length_um_override=None,
):
    """
    Returns (ex_spikes_bin, inh_spikes_bin), each shape (num_inputs, sim_duration_ms).

    Added 'total_tree_length_um_override' so that two separate calls 
    (one for an exc pool, one for an inh pool of a different size) 
    share the same denominator and aren't biased relative to each other 
    purely by population size. See 'build_population_inputs()'.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_segments = len(seg_length_um)
    adjusted_length_um = min_seg_length_um + seg_length_um
    total_tree_length_um = (total_tree_length_um_override
                             if total_tree_length_um_override is not None
                             else adjusted_length_um.sum())

    keep_rate_const_ms = inst_rate_interval_options_ms[
        np.random.randint(len(inst_rate_interval_options_ms))]
    keep_rate_const_ms += int(2 * inst_rate_interval_jitter * np.random.rand()
                               - inst_rate_interval_jitter)
    keep_rate_const_ms = max(keep_rate_const_ms, 1)

    sigma_cap_ms = sim_duration_ms // 3
    smoothing_sigma_ms = temporal_smoothing_sigma_options_ms[
        np.random.randint(len(temporal_smoothing_sigma_options_ms))]
    smoothing_sigma_ms += int(2 * temporal_smoothing_jitter * np.random.rand()
                               - temporal_smoothing_jitter)
    smoothing_sigma_ms = max(1, min(smoothing_sigma_ms, sigma_cap_ms))

    num_epochs = int(np.ceil(sim_duration_ms / keep_rate_const_ms))
    num_ex_per_100ms = np.random.uniform(
        low=num_ex_spikes_per_100ms_range[0], high=num_ex_spikes_per_100ms_range[1],
        size=(1, num_epochs))
    num_inh_low = np.maximum(0, num_ex_per_100ms + num_ex_inh_spike_diff_per_100ms_range[0])
    num_inh_high = num_ex_per_100ms + num_ex_inh_spike_diff_per_100ms_range[1]
    num_inh_per_100ms = np.random.uniform(low=num_inh_low, high=num_inh_high,
                                           size=(1, num_epochs))

    ex_rate_per_um_per_ms = num_ex_per_100ms / (total_tree_length_um * 100.0)
    inh_rate_per_um_per_ms = num_inh_per_100ms / (total_tree_length_um * 100.0)

    ex_rate_per_seg = np.kron(ex_rate_per_um_per_ms, np.ones((num_segments, 1)))
    inh_rate_per_seg = np.kron(inh_rate_per_um_per_ms, np.ones((num_segments, 1)))
    ex_rate_per_seg *= np.tile(adjusted_length_um[:, np.newaxis], [1, ex_rate_per_seg.shape[1]])
    inh_rate_per_seg *= np.tile(adjusted_length_um[:, np.newaxis], [1, inh_rate_per_seg.shape[1]])
    ex_rate_per_seg *= np.random.uniform(0.5, 1.5, size=ex_rate_per_seg.shape)
    inh_rate_per_seg *= np.random.uniform(0.5, 1.5, size=inh_rate_per_seg.shape)

    ex_rate_per_seg = np.kron(ex_rate_per_seg, np.ones((1, keep_rate_const_ms)))[:, :sim_duration_ms]
    inh_rate_per_seg = np.kron(inh_rate_per_seg, np.ones((1, keep_rate_const_ms)))[:, :sim_duration_ms]

    win_len = 1 + 7 * smoothing_sigma_ms
    smoothing_window = gaussian(win_len, std=smoothing_sigma_ms)[np.newaxis, :]
    smoothing_window /= smoothing_window.sum()

    ex_rate_smoothed = np.clip(signal.convolve(ex_rate_per_seg, smoothing_window, mode='same'), 0, None)
    inh_rate_smoothed = np.clip(signal.convolve(inh_rate_per_seg, smoothing_window, mode='same'), 0, None)

    ex_inst_prob = np.random.exponential(scale=ex_rate_smoothed)
    inh_inst_prob = np.random.exponential(scale=inh_rate_smoothed)
    ex_spikes_bin = (np.random.rand(*ex_inst_prob.shape) < ex_inst_prob).astype(np.uint8)
    inh_spikes_bin = (np.random.rand(*inh_inst_prob.shape) < inh_inst_prob).astype(np.uint8)
    return ex_spikes_bin, inh_spikes_bin


def check_per_input_rates(ex_spikes_bin, inh_spikes_bin, sim_duration_ms, label="input"):
    ex_rates = ex_spikes_bin.sum(axis=1) / (sim_duration_ms / 1000.0)
    inh_rates = inh_spikes_bin.sum(axis=1) / (sim_duration_ms / 1000.0)
    print(f"    Per-{label} exc rate: mean={ex_rates.mean():.1f} Hz, "
          f"max={ex_rates.max():.1f} Hz, min={ex_rates.min():.1f} Hz")
    print(f"    Per-{label} inh rate: mean={inh_rates.mean():.1f} Hz, "
          f"max={inh_rates.max():.1f} Hz, min={inh_rates.min():.1f} Hz")
    if ex_rates.max() > 300:
        print("    WARNING: peak per-input rate > 300 Hz -- "
              "consider reducing the excitatory rate range.")


def build_population_inputs(
    sim_duration_ms, num_exc_inputs, num_inh_inputs, min_seg_length_um,
    num_ex_spikes_per_100ms_range, num_ex_inh_spike_diff_per_100ms_range,
    inst_rate_interval_options_ms, temporal_smoothing_sigma_options_ms,
    inst_rate_interval_jitter=20, temporal_smoothing_jitter=20, random_seed=None,
):
    """
    Build two independent, unweighted input populations (no dendritic-length
    weighting -- every input is nominally equivalent) for a point neuron: one
    call sized for the excitatory pool, one (with an offset seed) sized for
    the inhibitory pool. Shared by the LIF and HH point-neuron models, which
    were previously carrying identical copies of this logic.

    The exc and inh calls share one combined 'total_tree_length_um_override'
    (based on num_exc_inputs + num_inh_inputs together) rather than each
    normalizing by its own pool size. Without this, an asymmetric split (e.g.
    500 exc vs. 125 inh) inflates the smaller pool's per-input rate purely
    because it's dividing by a smaller total -- a structural bias, not a
    reflection of num_ex_spikes_per_100ms_range vs. the inh diff range. With a
    shared denominator, the exc/inh balance reflects only the sampled rate
    epochs, matching how the multicompartment model (a single call, one
    shared seg_length_um array) behaves.
    """
    shared_total_length_um = (min_seg_length_um + 1.0) * (num_exc_inputs + num_inh_inputs)

    ex_bin, _ = generate_input_spike_trains(
        sim_duration_ms=sim_duration_ms, seg_length_um=np.ones(num_exc_inputs),
        min_seg_length_um=min_seg_length_um,
        num_ex_spikes_per_100ms_range=num_ex_spikes_per_100ms_range,
        num_ex_inh_spike_diff_per_100ms_range=num_ex_inh_spike_diff_per_100ms_range,
        inst_rate_interval_options_ms=inst_rate_interval_options_ms,
        temporal_smoothing_sigma_options_ms=temporal_smoothing_sigma_options_ms,
        inst_rate_interval_jitter=inst_rate_interval_jitter,
        temporal_smoothing_jitter=temporal_smoothing_jitter,
        random_seed=random_seed,
        total_tree_length_um_override=shared_total_length_um,
    )
    _, inh_bin = generate_input_spike_trains(
        sim_duration_ms=sim_duration_ms, seg_length_um=np.ones(num_inh_inputs),
        min_seg_length_um=min_seg_length_um,
        num_ex_spikes_per_100ms_range=num_ex_spikes_per_100ms_range,
        num_ex_inh_spike_diff_per_100ms_range=num_ex_inh_spike_diff_per_100ms_range,
        inst_rate_interval_options_ms=inst_rate_interval_options_ms,
        temporal_smoothing_sigma_options_ms=temporal_smoothing_sigma_options_ms,
        inst_rate_interval_jitter=inst_rate_interval_jitter,
        temporal_smoothing_jitter=temporal_smoothing_jitter,
        random_seed=None if random_seed is None else random_seed + 1,
        total_tree_length_um_override=shared_total_length_um,
    )
    return ex_bin, inh_bin


# Shared synaptic kinetics -- both point-neuron models and the multicompartment
# model reference the same AMPA/GABA-A (and, for the multicompartment
# deterministic path, NMDA) time constants. Kept here as plain data (no NEURON
# dependency) so every script can import the same numbers instead of carrying
# separate hand-copied dicts.
DEFAULT_SYNAPSE_KINETICS = {
    "AMPA":   {"tau_rise_ms": 0.2, "tau_decay_ms": 2.0,  "E_rev_mV": 0.0},
    "NMDA":   {"tau_rise_ms": 1.0, "tau_decay_ms": 50.0, "E_rev_mV": 0.0, "mg_mM": 1.0},
    "GABA_A": {"tau_rise_ms": 0.5, "tau_decay_ms": 8.0,  "E_rev_mV": -75.0},
}