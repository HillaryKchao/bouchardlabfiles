"""
miso_dataset_gen.py
=====================
Builds a MISO dataset from L5PC multicompartment simulation.

INPUT  (X):  per-segment presynaptic spike trains delivered onto the cell
             (excitatory + inhibitory), binary, 1 ms bins.
             shape = (N_in, T)   where N_in = 2 * n_synaptic_segments
             (channel 0:n_segs = excitatory (AMPA+NMDA) bins,
              channel n_segs:2*n_segs = inhibitory (GABA-A) bins)

OUTPUT (Y):  somatic output spike train, binary, 1 ms bins (same dt as X).
             shape = (N_out, T) with N_out = 1

"""

import numpy as np
import l5pc_real_simulation as sim

def bin_output_spikes(spike_times_ms, sim_duration_ms):
    """
    Convert continuous-time somatic AP times into the same 1ms-bin format
    used for ex_spikes_bin / inh_spikes_bin, so X and Y are on one time axis.
    """
    
    y = np.zeros((1, int(sim_duration_ms)), dtype=np.uint8)
    bins = np.floor(np.asarray(spike_times_ms)).astype(int)
    bins = bins[(bins >= 0) & (bins < sim_duration_ms)]
    # extremely rare to get 2 somatic APs in the same 1ms bin, but precaution anyways
    np.add.at(y[0], bins, 1)
    y = np.clip(y, 0, 1)
    return y


def build_miso_dataset(random_seed=7, out_path="l5pc_miso_dataset.npz"):
    print("[1] Building real L5PC cell ...")
    cell = sim.build_cell()
    n_sections = sum(1 for _ in cell.all)

    print("[2] Collecting synaptic segments ...")
    all_segs, seg_length_um = sim.collect_all_segments(cell, include_soma=True)
    region_names = np.array([name for (_sec, _seg, name) in all_segs])

    print("[3] Generating input spike trains (the MISO input X) ...")
    ex_spikes_bin, inh_spikes_bin = sim.generate_input_spike_trains_for_simulation(
        sim_duration_ms=sim.SIM_DURATION_MS,
        seg_length_um=seg_length_um,
        min_seg_length_um=sim.MIN_SEG_LENGTH_UM,
        num_ex_spikes_per_100ms_range=sim.NUM_EX_SPIKES_PER_100MS_RANGE,
        num_ex_inh_spike_diff_per_100ms_range=sim.NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE,
        inst_rate_interval_options_ms=sim.INST_RATE_INTERVAL_OPTIONS_MS,
        temporal_smoothing_sigma_options_ms=sim.TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS,
        inst_rate_interval_jitter=sim.INST_RATE_INTERVAL_JITTER,
        temporal_smoothing_jitter=sim.TEMPORAL_SMOOTHING_SIGMA_JITTER,
        random_seed=random_seed,
    )
    sim.check_per_synapse_rates(ex_spikes_bin, inh_spikes_bin, sim.SIM_DURATION_MS)

    print("[4] Placing synapses on the cell ...")
    synapses, spike_trains, netcons = sim.add_synapses_multicompartment(
        all_segs, ex_spikes_bin, inh_spikes_bin, weight_scale=1.0)

    print("[5] Setting up recording ...")
    t_vec, soma_v, apical_v, basal_v, spike_vec, apc = sim.setup_recording(cell)

    print(f"[6] Running simulation ({sim.SIM_DURATION_MS} ms, dt={sim.DT_MS} ms) ...")
    sim.run_simulation(sim.SIM_DURATION_MS, sim.DT_MS)
    n_aps = int(spike_vec.size())
    print(f"    Done. Somatic APs: {n_aps} "
          f"({1000.0 * n_aps / sim.SIM_DURATION_MS:.2f} Hz)")

    spk_t_arr = np.array(spike_vec)
    t_ms_arr = np.array(t_vec)
    soma_v_arr = np.array(soma_v)

    print("[7] Assembling MISO arrays ...")
    X = np.concatenate([ex_spikes_bin, inh_spikes_bin], axis=0).astype(np.uint8)   # (N_in, T)
    Y = bin_output_spikes(spk_t_arr, sim.SIM_DURATION_MS)                          # (1, T)
    input_channel_kind = np.array(
        ["excitatory"] * ex_spikes_bin.shape[0] + ["inhibitory"] * inh_spikes_bin.shape[0]
    )

    print(f"    X (input):  shape={X.shape}  dtype={X.dtype}  "
          f"(N_in={X.shape[0]} segment-channels, T={X.shape[1]} ms bins)")
    print(f"    Y (output): shape={Y.shape}  dtype={Y.dtype}  "
          f"(N_out=1 somatic channel, {int(Y.sum())} spikes)")

    np.savez_compressed(
        out_path,
        X=X,                                   # (N_in, T) input spike trains = MISO input
        Y=Y,                                   # (1, T)   somatic spike train = MISO output
        dt_ms=np.array(1.0),                   # bin width shared by X and Y
        sim_duration_ms=np.array(sim.SIM_DURATION_MS),
        input_channel_kind=input_channel_kind,         # 'excitatory' / 'inhibitory' per row of X
        input_channel_region=np.concatenate([region_names, region_names]),  # soma/basal/apical per row of X
        soma_v_mv=soma_v_arr, t_ms=t_ms_arr, somatic_spike_times_ms=spk_t_arr,
        random_seed=np.array(random_seed if random_seed is not None else -1),
    )
    print(f"[8] Saved MISO dataset -> {out_path}")
    return X, Y


if __name__ == "__main__":
    build_miso_dataset(random_seed=7, out_path="l5pc_miso_dataset.npz")