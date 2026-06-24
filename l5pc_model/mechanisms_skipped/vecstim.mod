NEURON {
    POINT_PROCESS VecStim
}

ASSIGNED {
    index
}

INITIAL {
    index = 0
}

NET_RECEIVE (w) {
    : events are delivered externally via NetCon.event()
}