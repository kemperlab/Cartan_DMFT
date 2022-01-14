import numpy as np
from functions.gate_funcs import GBAt, getGroundStateShortCircuitFunc, GBAt_manual, GBAt_manual_smallK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, transpile, IBMQ
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter


def build_circuit(AIMp, ham, tNpt, times, greenslist, evolution = 'KHK', stabilize=None, barriers=False):
    """                     GROUND STATE                 """
    #GSfunc, groundEnergy = getGroundStateShortCircuitFunc(ham)
    GSfunc, dump = getGroundStateShortCircuitFunc(ham)

    circuitList = []

    """      GENERATE THE CIRCUITS FOR GREEN'S FUNCTIONS """
    iGt_imp = np.zeros(tNpt,dtype=complex)

    #GreensCoeffientList = [0.5, -0.5j, 0.5j, 0.5]
    circuitList = []
    for tt  in range(tNpt):
        currentSum = 0
        time = times[tt]
        for i in range(len(greenslist)): #Changed to just measure XX, also update processing_func
            #Create two Qubit Registers:
            full_qc = QuantumCircuit(5,5, name=str(time))
            subSystem_qc = QuantumCircuit(4)
            #Create the Ground State on the subsystem:s
            subSystem_qc = GSfunc(subSystem_qc)
            #Append the ground state to the full circuit
            full_qc.append(subSystem_qc,range(1,5))
            #Add the Green's function to the circuit:
            full_qc = GBAt(full_qc, AIMp, time, greenslist[i][0:2], evolution=evolution, stabilize = stabilize, barriers=barriers)
            #Get the expectations <Z> = pr(0) - pr(1)
            full_qc.measure(range(0,5), range(0,5))
            circuitList.append(full_qc)
    return circuitList


def build_circuit_manual_linear(AIMp, ham, tNpt, times, Greenslist):

    circuitList = []

    GSfunc, dump = getGroundStateShortCircuitFunc(ham)
    iGt_imp = np.zeros(tNpt,dtype=complex)
    
    #Create Indexed Lists to iterate over the full computation
    #GreensCoeffientList = [0.5, -0.5j, 0.5j, 0.5]
    for tt  in range(tNpt):
        currentSum = 0
        time = times[tt]
        for i in range(len(Greenslist)): #Changed to just measure XX, also update processing_func):
            #Create two Qubit Registers:
            full_qc = QuantumCircuit(5,5, name=str(time))
            subSystem_qc = QuantumCircuit(4)
            
            #Create the Ground State on the subsystem:s
            subSystem_qc = GSfunc(subSystem_qc)
            #Append the ground state to the full circuit
            full_qc.append(subSystem_qc,range(1,5))
            #Add the Green's function to the circuit:
            full_qc = GBAt_manual(full_qc, AIMp, time, Greenslist[i][0:2])#, evolution=evolution, stabilize = stabilize, barriers=barriers)
            #Get the expectations <Z> = pr(0) - pr(1)
            full_qc.measure(range(0,5), range(0,5))
            circuitList.append(full_qc)
    return circuitList

def build_circuit_manual_reduced_k_linear(AIMp, ham, tNpt, times, Greenslist, iterations = 1):

    circuitList = []
    #Measurement Correction Shots = 2^n circuits


    GSfunc, dump = getGroundStateShortCircuitFunc(ham)
    iGt_imp = np.zeros(tNpt,dtype=complex)
    
    #Create Indexed Lists to iterate over the full computation
    #GreensCoeffientList = [0.5, -0.5j, 0.5j, 0.5]
    
    for tt  in range(tNpt):
        currentSum = 0
        time = times[tt]
        for i in range(len(Greenslist)): #Changed to just measure XX, also update processing_func):
            #Create two Qubit Registers:
            full_qc = QuantumCircuit(5,5, name=str(time))
            subSystem_qc = QuantumCircuit(4)
            
            #Create the Ground State on the subsystem:s
            subSystem_qc = GSfunc(subSystem_qc)
            #Append the ground state to the full circuit
            full_qc.append(subSystem_qc,range(1,5))
            #Add the Green's function to the circuit:
            full_qc = GBAt_manual_smallK(full_qc, AIMp, time, Greenslist[i][0:2])#, evolution=evolution, stabilize = stabilize, barriers=barriers)
            #Get the expectations <Z> = pr(0) - pr(1)
            full_qc.measure(range(0,5), range(0,5))
            circuitList.append(full_qc)
    return circuitList
