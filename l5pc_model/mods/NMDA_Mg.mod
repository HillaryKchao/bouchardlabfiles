COMMENT
Jahr-Stevens Mg2+ block NMDA receptor.
Adapted from Beniaguev et al. (2021) neuron_as_deep_net repo.

BUGFIX (vs. earlier version of this file):
The original version had a separate `gmax` PARAMETER (default 0.001 uS)
that was multiplied into BREAKPOINT *on top of* the NetCon `weight`, which
is already scaled by `factor` in NET_RECEIVE. Since gmax was never set from
Python, it silently stayed at its tiny default, so the measured peak NMDA
conductance was ~1000x smaller than the weight implied (0.0005 nS instead
of the intended ~8 nS). Fix: drop gmax entirely. `weight` (set via
nc.weight[0] in Python) now directly sets peak conductance, exactly like
NEURON's built-in Exp2Syn convention for AMPA/GABA_A in this script.
ENDCOMMENT

NEURON {
    POINT_PROCESS NMDA_Mg
    NONSPECIFIC_CURRENT i
    RANGE tau1, tau2, e, mg, i, g
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER {
    tau1 = 1.0    (ms)   : rise time constant
    tau2 = 50.0   (ms)   : decay time constant
    e    = 0.0    (mV)   : reversal potential
    mg   = 1.0    (mM)   : extracellular Mg2+ concentration
}

ASSIGNED {
    v    (mV)
    i    (nA)
    g    (uS)
    factor
}

STATE {
    A  (uS)
    B  (uS)
}

INITIAL {
    : Normalization factor so peak of (B-A) = 1
    LOCAL tp
    A = 0
    B = 0
    tp = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
    factor = 1 / (-exp(-tp/tau1) + exp(-tp/tau2))
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = factor * (B - A) * mgblock(v)
    i = g * (v - e)
}

DERIVATIVE state {
    A' = -A / tau1
    B' = -B / tau2
}

NET_RECEIVE(weight (uS)) {
    A = A + weight * factor
    B = B + weight * factor
}

FUNCTION mgblock(v(mV)) {
    : Jahr & Stevens (1990), using [Mg2+]_o = mg mM
    mgblock = 1 / (1 + (mg / 3.57) * exp(-0.062 * v))
}
