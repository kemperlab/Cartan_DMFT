from datetime import datetime
from qiskit import IBMQ
import numpy as np
from numpy import savez
IBMQ.load_account()
backend='ibmq_manila'
ncsu_provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')#
backend = ncsu_provider.get_backend(backend)
coupling_map = [(0,1),(1,2),(2,3),(3,4)]
cx_errors = {(0,1):[],(1,2):[],(2,3):[],(3,4):[]}
T1 = {0:[],1:[],2:[],3:[],4:[]}
T2 = {0:[],1:[],2:[],3:[],4:[]}
timing = {(0,1):[],(1,2):[],(2,3):[],(3,4):[]}

for hour in [6, 12, 18, 0]:
    for month in [8,9]:
        if month == 8:
            for day in range(10, 32):
                t = datetime(day=day, month=month, year=2021, hour=hour)
                props = backend.properties(datetime=t)
                for item in props.gates:
                    if item.gate == 'cx':
                        for set in coupling_map:
                            if set[0] in item.qubits and set[1] in item.qubits:
                                cx_errors[set].append(item.parameters[0].value)
                                timing[set].append(item.parameters[1].value)
                for (qubit, qubitIndex) in zip(props.qubits, range(len(props.qubits))):
                    T1[qubitIndex].append(qubit[0].value)
                    T2[qubitIndex].append(qubit[1].value)

        else:
            for day in range(1, 10):
                t = datetime(day=day, month=month, year=2021, hour=hour)
                props = backend.properties(datetime=t)
                for item in props.gates:
                    if item.gate == 'cx':
                        for set in coupling_map:
                            if set[0] in item.qubits and set[1] in item.qubits:
                                cx_errors[set].append(item.parameters[0].value)
                for (qubit, qubitIndex) in zip(props.qubits, range(len(props.qubits))):
                    T1[qubitIndex].append(qubit[0].value)
                    T2[qubitIndex].append(qubit[1].value)

savez('manila_data_timing.npz',gate_times=timing, allow_pickle=True)