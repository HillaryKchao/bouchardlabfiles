"""
plot_nmda_comparison.py
=========================
Plots a comparison of the original NMDA-on run (l5pc_real_simulation_data.npz) 
and the NMDA-off control run (l5pc_nmda_off_data.npz). 
"""

import numpy as np
import matplotlib.pyplot as plt

d_on = np.load('l5pc_real_simulation_data.npz')
d_off = np.load('l5pc_nmda_off_data.npz')

t = d_on['t_ms']
api_on, api_off = d_on['apical_v_mv'], d_off['apical_v_mv']
soma_on, soma_off = d_on['soma_v_mv'], d_off['soma_v_mv']

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

ax = axes[0]
ax.plot(t, api_on, color='#FF9800', lw=0.8, label='NMDA ON (original)')
ax.plot(t, api_off, color='#555555', lw=0.8, alpha=0.8, label='NMDA OFF (weight=0)')
ax.axhline(-40, color='k', lw=0.7, linestyle=':', label='-40 mV reference line')
ax.set_ylabel('Apical Vm (mV)')
ax.set_title('Apical Dendrite (391 um from soma): NMDA on vs NMDA off, same input, same seed', fontsize=11)
ax.set_ylim(-80, 50)
ax.legend(fontsize=9, loc='upper right')

ax = axes[1]
ax.plot(t, soma_on, color='k', lw=0.8, label='Soma, NMDA ON (54 APs)')
ax.plot(t, soma_off, color='#2196F3', lw=0.8, alpha=0.8, label='Soma, NMDA OFF (1 AP)')
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Soma Vm (mV)')
ax.set_title('Somatic output: NMDA on vs NMDA off', fontsize=11)
ax.set_ylim(-80, 50)
ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('nmda_on_off_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: nmda_on_off_comparison.png")
plt.show()