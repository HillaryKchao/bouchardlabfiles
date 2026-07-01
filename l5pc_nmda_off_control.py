"""
l5pc_nmda_off_control.py
=========================
Forces NMDA synapse's weight to 0, then runs the same simulation as 
l5pc_real_simulation.py's main(random_seed=7) run. 

Doing this to see how much of  the apical dendrite's depolarization 
and somatic firing is actually NMDA-driven, as opposed to 
AMPA/GABA/backpropagating-AP activity.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l5pc_real_simulation as m   # noqa: E402  (must come after path setup)

def is_nmda_synapse(syn):
    """Return True for the NMDA synapse objects created by the real model."""
    return hasattr(syn, 'mg')

def main(random_seed=7):
    np.random.seed(random_seed)

    print("[1] Building cell...")
    cell = m.build_cell()
    all_segs, seg_length_um = m.collect_all_segments(cell, include_soma=True)

    print("[2] Generating the SAME spike trains as the original run "
          f"(random_seed={random_seed})...")
    ex_bin, inh_bin = m.generate_input_spike_trains_for_simulation(
        sim_duration_ms=m.SIM_DURATION_MS, seg_length_um=seg_length_um,
        min_seg_length_um=m.MIN_SEG_LENGTH_UM,
        num_ex_spikes_per_100ms_range=m.NUM_EX_SPIKES_PER_100MS_RANGE,
        num_ex_inh_spike_diff_per_100ms_range=m.NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE,
        inst_rate_interval_options_ms=m.INST_RATE_INTERVAL_OPTIONS_MS,
        temporal_smoothing_sigma_options_ms=m.TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS,
        inst_rate_interval_jitter=m.INST_RATE_INTERVAL_JITTER,
        temporal_smoothing_jitter=m.TEMPORAL_SMOOTHING_SIGMA_JITTER,
        random_seed=random_seed,
    )

    print("[3] Placing synapses, then forcing every NMDA weight to 0...")
    synapses, spike_trains, netcons = m.add_synapses_multicompartment(
        all_segs, ex_bin, inh_bin, weight_scale=1.0)

    n_nmda_zeroed = 0
    for i, syn in enumerate(synapses):
        if is_nmda_synapse(syn):
            netcons[i].weight[0] = 0.0
            n_nmda_zeroed += 1
    print(f"    Zeroed {n_nmda_zeroed} NMDA synapse weights "
          f"(expected {len(all_segs)}, one per synaptic segment)")
    # only NMDA disabled, AMPA/GABA synapses/netcons are untouched

    print("[4] Setting up recording (soma + apical + basal)...")
    t_vec, soma_v, apical_v, basal_v, spike_vec, apc = m.setup_recording(cell)

    print(f"[5] Running simulation ({m.SIM_DURATION_MS} ms, NMDA disabled)...")
    t0 = time.time()
    m.run_simulation(m.SIM_DURATION_MS, m.DT_MS)
    print(f"    Done in {time.time() - t0:.1f} s")

    t_ms_arr = np.array(t_vec)
    soma_v_arr = np.array(soma_v)
    apical_v_arr = np.array(apical_v)
    basal_v_arr = np.array(basal_v)
    spk_t_arr = np.array(spike_vec)

    above = apical_v_arr > -40
    print(f"\n    Apical samples above -40 mV: {above.sum()}/{len(apical_v_arr)} "
          f"({100 * above.mean():.1f}%)")
    print(f"    Apical V min/max: {apical_v_arr.min():.1f} / {apical_v_arr.max():.1f} mV")
    print(f"    Somatic APs: {len(spk_t_arr)} "
          f"(vs. 54 in the original NMDA-on run)")

    np.savez_compressed(
        'l5pc_nmda_off_data.npz',
        t_ms=t_ms_arr, soma_v_mv=soma_v_arr, apical_v_mv=apical_v_arr,
        basal_v_mv=basal_v_arr, somatic_spike_times=spk_t_arr,
    )
    print("\nSaved: l5pc_nmda_off_data.npz")
    return t_ms_arr, soma_v_arr, apical_v_arr, basal_v_arr, spk_t_arr


if __name__ == '__main__':
    main(random_seed=7)