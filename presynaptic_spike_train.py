from neuron import h

class PresynapticSpikeTrain:
    def __init__(self, spike_times_ms, netcon):
        self.spike_times_ms = spike_times_ms
        self.nc = netcon
        self._fih = h.FInitializeHandler(self._schedule_events)

    def _schedule_events(self):
        for t in self.spike_times_ms:
            self.nc.event(t)