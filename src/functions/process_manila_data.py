import numpy as np
import matplotlib.pyplot as plt
"""file = np.load('manila_data.npz', allow_pickle=True)
T1 = file['T1'].item()
T2 = file['T2'].item()
cx_error = file['cx_errors'].item()
#print(T2[2])
T1avg = []
T2avg = []
cx_error_avg = []
for qubitIndex in [0,1,2,3,4]:
    T1_std = np.std(T1[qubitIndex])
    T2_std = np.std(T2[qubitIndex])
    T1avg.append((np.average(T1[qubitIndex]), T1_std))
    T2avg.append((np.average(T2[qubitIndex]), T2_std))
print(T1avg)
print(T2avg)
coupling_map = [(0,1),(1,2),(2,3),(3,4)]

for qubitpair in coupling_map:
    CX_avg = np.average(cx_error[qubitpair])
    CX_std = np.std(cx_error[qubitpair])
    cx_error_avg.append((CX_avg, CX_std))
print(cx_error_avg)"""

"""fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
ax1.bar(range(len(T1avg)), [x[0] for x in T1avg], yerr=[x[1] for x in T1avg], align='center')
ax2.bar(range(len(T2avg)), [x[0] for x in T2avg], yerr=[x[1] for x in T2avg], align='center')
ax1.set_ylabel(r'Coherence Time ($\mu$s)')
ax1.set_title('T1')
ax2.set_title('T2')
plt.show()"""


file = np.load('manila_data_timing.npz', allow_pickle=True)
cx_error = file['gate_times'].item()
cx_error_avg = []
coupling_map = [(0,1),(1,2),(2,3),(3,4)]

for qubitpair in coupling_map:
    CX_avg = np.average(cx_error[qubitpair])
    CX_std = np.std(cx_error[qubitpair])
    cx_error_avg.append((CX_avg, CX_std))
print(cx_error_avg)