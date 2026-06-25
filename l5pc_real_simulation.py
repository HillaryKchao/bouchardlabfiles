"""
l5pc_real_simulation.py
========================

SETUP NOTES
--------------------------------------------------------------------
l5pc_model/ folder is a CLEANED version of the BBP "L5_TTPC1_cADpyr232_1" 
package. Three things were removed/patched relative to the original download:
  1. mechanisms/ProbAMPANMDA_EMS.mod and ProbGABAAB_EMS.mod were moved to
     mechanisms_skipped/ -- they use nrn_random_arg()/scop_random() with a
     legacy signature that fails to compile on this NEURON version (a
     NEURON 9.x C++ API change). Not needed because I use my own synapses.
  2. mechanisms/vecevent.mod and vecstim.mod were moved to
     mechanisms_skipped/ -- both define a mechanism named "VecStim",
     which collide if compiled together. Not needed since this pipeline
     delivers spikes via NetCon(None, syn).event(), not VecStim.
  3. synapses/synapses.hoc was replaced with a stub (the original is saved
     alongside as synapses_full_original.hoc.bak). The real one
     instantiates ProbAMPANMDA_EMS/ProbGABAAB_EMS, and NEURON resolves
     "new Mechanism(location)" eagerly while parsing (even inside a proc
     that's never called!) so the file fails to load at all without those
     two mechanisms compiled. The stub satisfies the parser without
     pulling in BBP's anatomical connectivity loader (which isn't used anyways, 
     as synapses_enabled=0 is passed to the cell constructor).
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "l5pc_model")
MECH_DIR   = os.path.join(MODEL_DIR, "mechanisms")

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SIM_DURATION_MS = 2000
DT_MS           = 0.025

# documented BBP operating temperature for this exact cell (constants.hoc)
CELSIUS = 34.0
V_INIT  = -65.0

MIN_SEG_LENGTH_UM = 10.0
NUM_EX_SPIKES_PER_100MS_RANGE         = [20, 200]
NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE = [-60, 20]

INST_RATE_INTERVAL_OPTIONS_MS = [25, 30, 35, 40, 45, 55, 60, 65, 70,
                                  75, 80, 85, 90, 100, 150, 200, 300, 450]
INST_RATE_INTERVAL_JITTER     = 20

TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS = [25, 30, 35, 40, 50, 60, 80,
                                        100, 150, 200, 300, 400, 500, 600]
TEMPORAL_SMOOTHING_SIGMA_JITTER     = 20

SYNAPSE_PARAMS = {
    "AMPA":   {"tau_rise_ms": 0.2, "tau_decay_ms": 2.0,  "E_rev_mV": 0.0},
    "NMDA":   {"tau_rise_ms": 1.0, "tau_decay_ms": 50.0, "E_rev_mV": 0.0, "mg_mM": 1.0},
    "GABA_A": {"tau_rise_ms": 0.5, "tau_decay_ms": 8.0,  "E_rev_mV": -75.0},
}

# check check_per_synapse_rates() output and the resulting firing rate
# raise/lower if the cell is silent or saturating
AMPA_WEIGHT_US  = 0.008
NMDA_WEIGHT_US  = 0.008
GABAA_WEIGHT_US = 0.010


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — compile mechanisms and import NEURON
# ─────────────────────────────────────────────────────────────────────────────

def _get_arch_dir(mech_dir):
    for name in ("x86_64", "arm64", "aarch64"):
        path = os.path.join(mech_dir, name)
        if os.path.isdir(path):
            return path
    return None


def compile_mechanisms():
    arch_dir = _get_arch_dir(MECH_DIR)
    already_compiled = (
        arch_dir is not None and
        any(f.endswith((".so", ".dylib")) or f == "special" for f in os.listdir(arch_dir))
    )
    if already_compiled:
        print(f"[MOD] Mechanisms already compiled in {arch_dir}, skipping.")
        return
    print(f"[MOD] Compiling mechanisms in {MECH_DIR} with nrnivmodl ...")
    result = subprocess.run(["nrnivmodl", "."], capture_output=True, text=True, cwd=MECH_DIR)
    if result.returncode != 0:
        print("nrnivmodl stdout:\n", result.stdout)
        print("nrnivmodl stderr:\n", result.stderr)
        raise RuntimeError("nrnivmodl compilation failed -- see output above.")
    arch_dir = _get_arch_dir(MECH_DIR)
    if arch_dir is None:
        raise RuntimeError(f"nrnivmodl reported success but no arch dir found in {MECH_DIR}")
    print(f"[MOD] Compilation successful -> {arch_dir}")


compile_mechanisms()

import neuron                       # noqa: E402
from neuron import h                # noqa: E402

neuron.load_mechanisms(MECH_DIR)
if not hasattr(h, "NMDA_Mg"):
    raise RuntimeError("NMDA_Mg mechanism not found after load_mechanisms(). "
                        "Delete l5pc_model/mechanisms/x86_64 (or arm64) and rerun.")

# template.hoc / morphology.hoc / synapses/synapses.hoc all use paths
# relative to MODEL_DIR (e.g. load_file("morphology/....asc")), so NEURON
# must be pointed there before loading them.
_prev_cwd = os.getcwd()
os.chdir(MODEL_DIR)
h.load_file("import3d.hoc")
h.load_file("template.hoc")
os.chdir(_prev_cwd)


# ─────────────────────────────────────────────────────────────────────────────
# CELL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_cell():
    """
    Instantiate the real L5PC cell. Our synapses are added separately below
    via add_synapses_multicompartment().
    """
    prev_cwd = os.getcwd()
    os.chdir(MODEL_DIR)
    try:
        cell = h.cADpyr232_L5_TTPC1_0fb1ca4724(0)
    finally:
        os.chdir(prev_cwd)
    return cell


def collect_all_segments(cell, include_soma=True):
    """
    Flatten the cell's somatic/basal/apical SectionLists into a list of
    (section, segment, region_name) tuples plus a matching seg_length_um
    array, exactly like the toy model's collect_all_segments() but iterating
    real SectionLists instead of a 5-entry dict. Axon excluded (as before).
    """
    all_segs = []
    seg_lengths = []
    groups = [(cell.basal, "basal"), (cell.apical, "apical")]
    if include_soma:
        groups.insert(0, (cell.somatic, "soma"))
    for sl, name in groups:
        for sec in sl:
            for seg in sec:
                all_segs.append((sec, seg, name))
                seg_lengths.append(sec.L / sec.nseg)
    seg_lengths = np.array(seg_lengths)
    print(f"    Synaptic segments: {len(all_segs)} "
          f"(soma+basal+apical, axon excluded), "
          f"total tree length: {seg_lengths.sum():.1f} um")
    return all_segs, seg_lengths


# ─────────────────────────────────────────────────────────────────────────────
# SPIKE TRAIN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_input_spike_trains_for_simulation(
    sim_duration_ms, seg_length_um, min_seg_length_um,
    num_ex_spikes_per_100ms_range, num_ex_inh_spike_diff_per_100ms_range,
    inst_rate_interval_options_ms, temporal_smoothing_sigma_options_ms,
    inst_rate_interval_jitter=20, temporal_smoothing_jitter=20, random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    num_segments = len(seg_length_um)
    adjusted_length_um = min_seg_length_um + seg_length_um
    total_tree_length_um = adjusted_length_um.sum()

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
    smoothing_window = signal.windows.gaussian(win_len, std=smoothing_sigma_ms)[np.newaxis, :]
    smoothing_window /= smoothing_window.sum()

    ex_rate_smoothed = np.clip(signal.convolve(ex_rate_per_seg, smoothing_window, mode='same'), 0, None)
    inh_rate_smoothed = np.clip(signal.convolve(inh_rate_per_seg, smoothing_window, mode='same'), 0, None)

    ex_inst_prob = np.random.exponential(scale=ex_rate_smoothed)
    inh_inst_prob = np.random.exponential(scale=inh_rate_smoothed)
    ex_spikes_bin = (np.random.rand(*ex_inst_prob.shape) < ex_inst_prob).astype(np.uint8)
    inh_spikes_bin = (np.random.rand(*inh_inst_prob.shape) < inh_inst_prob).astype(np.uint8)
    return ex_spikes_bin, inh_spikes_bin


def check_per_synapse_rates(ex_spikes_bin, inh_spikes_bin, sim_duration_ms):
    ex_rates = ex_spikes_bin.sum(axis=1) / (sim_duration_ms / 1000.0)
    inh_rates = inh_spikes_bin.sum(axis=1) / (sim_duration_ms / 1000.0)
    print(f"    Per-synapse exc rate: mean={ex_rates.mean():.1f} Hz, "
          f"max={ex_rates.max():.1f} Hz, min={ex_rates.min():.1f} Hz")
    print(f"    Per-synapse inh rate: mean={inh_rates.mean():.1f} Hz, "
          f"max={inh_rates.max():.1f} Hz, min={inh_rates.min():.1f} Hz")
    if ex_rates.max() > 300:
        print("    WARNING: peak per-synapse rate > 300 Hz -- "
              "consider reducing NUM_EX_SPIKES_PER_100MS_RANGE.")


# ─────────────────────────────────────────────────────────────────────────────
# SYNAPSE PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────

class PresynapticSpikeTrain:
    def __init__(self, spike_times_ms, netcon):
        self.spike_times_ms = spike_times_ms
        self.nc = netcon
        self._fih = h.FInitializeHandler(self._schedule_events)

    def _schedule_events(self):
        for t in self.spike_times_ms:
            self.nc.event(t)


def add_synapses_multicompartment(all_segs, ex_spikes_bin, inh_spikes_bin, weight_scale=1.0):
    assert len(all_segs) == ex_spikes_bin.shape[0] == inh_spikes_bin.shape[0]
    synapses, spike_trains, netcons = [], [], []
    ampa_p, nmda_p, gabaa_p = SYNAPSE_PARAMS["AMPA"], SYNAPSE_PARAMS["NMDA"], SYNAPSE_PARAMS["GABA_A"]

    for seg_ind, (sec, seg, _name) in enumerate(all_segs):
        loc = seg.x

        ampa_syn = h.Exp2Syn(sec(loc))
        ampa_syn.tau1, ampa_syn.tau2, ampa_syn.e = (
            ampa_p["tau_rise_ms"], ampa_p["tau_decay_ms"], ampa_p["E_rev_mV"])
        synapses.append(ampa_syn)
        exc_bins = np.where(ex_spikes_bin[seg_ind, :] == 1)[0]
        exc_times_ms = np.maximum(exc_bins.astype(float) + 0.5, 0.1)
        ampa_nc = h.NetCon(None, ampa_syn)
        ampa_nc.delay, ampa_nc.weight[0] = 0.0, AMPA_WEIGHT_US * weight_scale
        netcons.append(ampa_nc)
        spike_trains.append(PresynapticSpikeTrain(exc_times_ms, ampa_nc))

        nmda_syn = h.NMDA_Mg(sec(loc))
        nmda_syn.tau1, nmda_syn.tau2, nmda_syn.e, nmda_syn.mg = (
            nmda_p["tau_rise_ms"], nmda_p["tau_decay_ms"], nmda_p["E_rev_mV"], nmda_p["mg_mM"])
        synapses.append(nmda_syn)
        nmda_nc = h.NetCon(None, nmda_syn)
        nmda_nc.delay, nmda_nc.weight[0] = 0.0, NMDA_WEIGHT_US * weight_scale
        netcons.append(nmda_nc)
        spike_trains.append(PresynapticSpikeTrain(exc_times_ms, nmda_nc))

        inh_syn = h.Exp2Syn(sec(loc))
        inh_syn.tau1, inh_syn.tau2, inh_syn.e = (
            gabaa_p["tau_rise_ms"], gabaa_p["tau_decay_ms"], gabaa_p["E_rev_mV"])
        synapses.append(inh_syn)
        inh_bins = np.where(inh_spikes_bin[seg_ind, :] == 1)[0]
        inh_times_ms = np.maximum(inh_bins.astype(float) + 0.5, 0.1)
        inh_nc = h.NetCon(None, inh_syn)
        inh_nc.delay, inh_nc.weight[0] = 0.0, GABAA_WEIGHT_US * weight_scale
        netcons.append(inh_nc)
        spike_trains.append(PresynapticSpikeTrain(inh_times_ms, inh_nc))

    return synapses, spike_trains, netcons


# ─────────────────────────────────────────────────────────────────────────────
# RECORDING & RUN
# ─────────────────────────────────────────────────────────────────────────────

def setup_recording(cell):
    t_vec = h.Vector(); t_vec.record(h._ref_t)
    soma_sec = list(cell.somatic)[0]
    soma_v = h.Vector(); soma_v.record(soma_sec(0.5)._ref_v)

    apical_secs = list(cell.apical)
    apic_sec = apical_secs[min(60, len(apical_secs) - 1)]
    apical_v = h.Vector(); apical_v.record(apic_sec(0.5)._ref_v)

    basal_secs = list(cell.basal)
    bas_sec = basal_secs[min(40, len(basal_secs) - 1)]
    basal_v = h.Vector(); basal_v.record(bas_sec(0.5)._ref_v)

    apc = h.APCount(soma_sec(0.5)); apc.thresh = -20.0
    spike_vec = h.Vector(); apc.record(spike_vec)

    return t_vec, soma_v, apical_v, basal_v, spike_vec, apc


def run_simulation(sim_duration_ms, dt_ms):
    h.load_file("stdrun.hoc")
    h.celsius = CELSIUS
    h.dt = dt_ms
    h.tstop = sim_duration_ms
    h.v_init = V_INIT
    cvode = h.CVode(); cvode.active(0)
    h.finitialize(h.v_init)
    h.run()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def double_exp_kernel(tau_rise_ms, tau_decay_ms, dt_ms=0.025, kernel_duration_ms=150.0):
    t = np.arange(0, kernel_duration_ms, dt_ms)
    raw = np.exp(-t / tau_decay_ms) - np.exp(-t / tau_rise_ms)
    raw[raw < 0] = 0
    peak = raw.max()
    return raw / peak if peak > 0 else raw


def plot_results(t_ms, soma_v, apical_v, basal_v, spike_times,
                  ex_spikes_bin, inh_spikes_bin, n_sections, dt_ms=0.025):
    t_ms_arr, soma_v_arr = np.array(t_ms), np.array(soma_v)
    api_v_arr, bas_v_arr = np.array(apical_v), np.array(basal_v)
    spikes = np.array(spike_times)
    sim_dur = ex_spikes_bin.shape[1]
    n_segs = ex_spikes_bin.shape[0]

    fig, axes = plt.subplots(6, 1, figsize=(16, 22),
                              gridspec_kw={'height_ratios': [2, 1.5, 1, 1.5, 1, 2]})
    fig.suptitle(
        'Multi-Compartment HH Neuron — Real L5PC (Hay et al. 2011 / BBP cADpyr232_L5_TTPC1)\n'
        f'({n_segs} synapse-bearing segments across {n_sections} sections, '
        f'AMPA+NMDA_Mg paired + GABA_A, T={CELSIUS}°C, over-dispersed Poisson, E/I correlated)',
        fontsize=11, y=0.995,
    )

    t_bins = np.arange(sim_dur)

    ax = axes[0]
    for seg_ind in range(n_segs):
        spk_t = t_bins[ex_spikes_bin[seg_ind, :] == 1]
        ax.plot(spk_t, np.full_like(spk_t, seg_ind), '|', color='#2196F3',
                markersize=1.5, linewidth=0.3, alpha=0.5)
    for seg_ind in range(n_segs):
        spk_t = t_bins[inh_spikes_bin[seg_ind, :] == 1]
        ax.plot(spk_t, np.full_like(spk_t, seg_ind + n_segs), '|', color='#F44336',
                markersize=1.5, linewidth=0.3, alpha=0.5)
    ax.axhline(n_segs - 0.5, color='k', lw=0.8, linestyle='--')
    ax.set_ylabel('Segment index')
    ax.set_xlim(0, sim_dur); ax.set_ylim(-1, 2 * n_segs)
    ax.text(10, n_segs * 0.05, 'Excitatory (AMPA+NMDA)', color='#2196F3', fontsize=8)
    ax.text(10, n_segs + n_segs * 0.05, 'Inhibitory (GABA-A)', color='#F44336', fontsize=8)
    ax.set_title('Input Spike Raster (Segev-style two-matrix format)', fontsize=10)
    ax.set_xlabel('Time (ms)')

    ax = axes[1]
    t_k = np.arange(0, 120, dt_ms)
    ampa = double_exp_kernel(SYNAPSE_PARAMS['AMPA']['tau_rise_ms'], SYNAPSE_PARAMS['AMPA']['tau_decay_ms'],
                              dt_ms=dt_ms, kernel_duration_ms=120.0)
    nmda = double_exp_kernel(SYNAPSE_PARAMS['NMDA']['tau_rise_ms'], SYNAPSE_PARAMS['NMDA']['tau_decay_ms'],
                              dt_ms=dt_ms, kernel_duration_ms=120.0)
    gabaa = double_exp_kernel(SYNAPSE_PARAMS['GABA_A']['tau_rise_ms'], SYNAPSE_PARAMS['GABA_A']['tau_decay_ms'],
                               dt_ms=dt_ms, kernel_duration_ms=120.0)
    ax.plot(t_k[:len(ampa)], ampa, '#2196F3', lw=2, label='AMPA (τr=0.2, τd=2 ms)')
    ax.plot(t_k[:len(nmda)], nmda, '#4CAF50', lw=2, label='NMDA (τr=1, τd=50 ms) — Mg²⁺ blocked')
    ax.plot(t_k[:len(gabaa)], gabaa, '#F44336', lw=2, label='GABA-A (τr=0.5, τd=8 ms)')
    ax.set_xlabel('Time after spike (ms)'); ax.set_ylabel('Norm. conductance')
    ax.set_title('Synaptic Conductance Kernels (kinetics; NMDA additionally Mg²⁺ gated)', fontsize=10)
    ax.legend(fontsize=8); ax.set_xlim(0, 110)

    ax = axes[2]
    exc_mean = ex_spikes_bin.mean(axis=0).astype(float)
    inh_mean = inh_spikes_bin.mean(axis=0).astype(float)
    ax.plot(t_bins, gaussian_filter1d(exc_mean, 50) * 1000, '#2196F3', lw=1.5, label='Exc mean inst. rate (Hz)')
    ax.plot(t_bins, gaussian_filter1d(inh_mean, 50) * 1000, '#F44336', lw=1.5, label='Inh mean inst. rate (Hz)')
    ax.set_ylabel('Inst. rate (Hz)'); ax.set_xlim(0, sim_dur)
    ax.set_title('Mean Instantaneous Firing Rate Across All Segments', fontsize=10)
    ax.legend(fontsize=8); ax.set_xlabel('Time (ms)')

    ax = axes[3]
    ax.plot(t_ms_arr, bas_v_arr, color='#9C27B0', lw=0.8, label='Basal dendrite (mid)')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Vm (mV)')
    ax.set_title('Basal Dendrite Membrane Potential', fontsize=10)
    ax.set_xlim(0, sim_dur); ax.set_ylim(-80, 50); ax.legend(fontsize=8)

    ax = axes[4]
    ax.plot(t_ms_arr, api_v_arr, color='#FF9800', lw=0.8, label='Apical dendrite (mid-tree)')
    ax.axhline(-40, color='k', lw=0.5, linestyle=':', label='NMDA spike threshold ≈ −40 mV')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Vm (mV)')
    ax.set_title('Apical Dendrite Vm — look for fast NMDA spike events', fontsize=10)
    ax.set_xlim(0, sim_dur); ax.set_ylim(-80, 50); ax.legend(fontsize=8)

    ax = axes[5]
    ax.plot(t_ms_arr, soma_v_arr, 'k', lw=0.8, label='V_soma')
    if len(spikes) > 0:
        ax.plot(spikes, np.full_like(spikes, 40), 'r|', markersize=10, label=f'{len(spikes)} APs')
    ax.axhline(-65, color='grey', lw=0.5, linestyle='--', label='V_rest')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Vm (mV)')
    ax.set_title('Somatic Membrane Potential (Real L5PC Model Output)', fontsize=10)
    ax.set_xlim(0, sim_dur); ax.set_ylim(-80, 50); ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = 'l5pc_real_simulation_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    Figure saved: {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(random_seed=None):
    print("=" * 70)
    print("Real L5PC Multi-Compartment Simulation (BBP cADpyr232_L5_TTPC1)")
    print("=" * 70)

    print("\n[1] Instantiating real L5PC cell ...")
    cell = build_cell()
    n_sections = sum(1 for _ in cell.all)
    n_segments_total = sum(sec.nseg for sec in cell.all)
    print(f"    {n_sections} total sections, {n_segments_total} total compartments "
          f"(incl. axon, which receives no synapses)")

    print("\n[2] Collecting synaptic segments (soma + basal + apical) ...")
    all_segs, seg_length_um = collect_all_segments(cell, include_soma=True)

    print(f"\n[3] Generating Segev-style spike trains "
          f"({len(all_segs)} segments, {SIM_DURATION_MS} ms) ...")
    ex_spikes_bin, inh_spikes_bin = generate_input_spike_trains_for_simulation(
        sim_duration_ms=SIM_DURATION_MS, seg_length_um=seg_length_um,
        min_seg_length_um=MIN_SEG_LENGTH_UM,
        num_ex_spikes_per_100ms_range=NUM_EX_SPIKES_PER_100MS_RANGE,
        num_ex_inh_spike_diff_per_100ms_range=NUM_EX_INH_SPIKE_DIFF_PER_100MS_RANGE,
        inst_rate_interval_options_ms=INST_RATE_INTERVAL_OPTIONS_MS,
        temporal_smoothing_sigma_options_ms=TEMPORAL_SMOOTHING_SIGMA_OPTIONS_MS,
        inst_rate_interval_jitter=INST_RATE_INTERVAL_JITTER,
        temporal_smoothing_jitter=TEMPORAL_SMOOTHING_SIGMA_JITTER,
        random_seed=random_seed,
    )
    print(f"    ex_spikes_bin: shape={ex_spikes_bin.shape}, total spikes={ex_spikes_bin.sum()}")
    print(f"    inh_spikes_bin: shape={inh_spikes_bin.shape}, total spikes={inh_spikes_bin.sum()}")
    check_per_synapse_rates(ex_spikes_bin, inh_spikes_bin, SIM_DURATION_MS)

    print("\n[4] Placing synapses (AMPA+NMDA_Mg + GABA-A per segment) ...")
    synapses, spike_trains, netcons = add_synapses_multicompartment(
        all_segs, ex_spikes_bin, inh_spikes_bin, weight_scale=1.0)
    print(f"    {len(synapses)} total synapse objects "
          f"({len(all_segs)} AMPA + {len(all_segs)} NMDA + {len(all_segs)} GABA-A)")

    print("\n[5] Setting up recording (soma + apical + basal) ...")
    t_vec, soma_v, apical_v, basal_v, spike_vec, apc = setup_recording(cell)

    print(f"\n[6] Running simulation ({SIM_DURATION_MS} ms, dt={DT_MS} ms, T={CELSIUS}C) ...")
    print("    (this is a ~2500-compartment model with full active dendrites -- "
          "expect roughly 1-2 minutes of wall-clock time)")
    run_simulation(SIM_DURATION_MS, DT_MS)
    n_aps = int(spike_vec.size())
    print(f"    Done. Somatic APs: {n_aps} (output rate: {1000.0 * n_aps / SIM_DURATION_MS:.2f} Hz)")

    t_ms_arr = np.array(t_vec)
    soma_v_arr = np.array(soma_v)
    api_v_arr = np.array(apical_v)
    bas_v_arr = np.array(basal_v)
    spk_t_arr = np.array(spike_vec)

    print("\n[7] Plotting results ...")
    plot_results(
        t_ms=t_ms_arr, soma_v=soma_v_arr, apical_v=api_v_arr, basal_v=bas_v_arr,
        spike_times=spk_t_arr, ex_spikes_bin=ex_spikes_bin, inh_spikes_bin=inh_spikes_bin,
        n_sections=n_sections, dt_ms=DT_MS,
    )

    print("\n[8] Saving simulation data ...")
    np.savez_compressed(
        'l5pc_real_simulation_data.npz',
        t_ms=t_ms_arr, soma_v_mv=soma_v_arr, apical_v_mv=api_v_arr, basal_v_mv=bas_v_arr,
        somatic_spike_times=spk_t_arr, ex_spikes_bin=ex_spikes_bin, inh_spikes_bin=inh_spikes_bin,
    )
    print("    Saved: l5pc_real_simulation_data.npz")
    print("\n" + "=" * 70 + "\nDone.\n" + "=" * 70)

    return ex_spikes_bin, inh_spikes_bin, t_ms_arr, soma_v_arr, api_v_arr, bas_v_arr, spk_t_arr


if __name__ == '__main__':
    results = main(random_seed=7)
