from functions.DMFT_funcs import *
from functions.anderson import AIM
from logging import raiseExceptions
import qiskit
from qiskit import QuantumCircuit, IBMQ
import numpy as np
from qiskit import circuit
import scipy as sci
import matplotlib.pyplot as plt
import sys
import os
from datetime import date
sys.path.append("..") # Adds higher directory to python modules path.
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, transpile, IBMQ
from qiskit.providers.aer import AerSimulator
from CQS.methods import *
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter)
#from qiskit.providers.honeywell import Honeywell
from qiskit.providers import JobStatus
from functions.job_circ_funcs import getJobResults, sendJobs, loadMostRecentJob, clearCurrentJob, transpileJobs, loadFullInfo, loadCircuitInfo, saveFullInfo, saveCircuitInfo, getManagedJobResults, sendManagedJobs
from functions.gate_funcs import getGroundStateCircuitFunc, getGroundStateShortCircuitFunc, PauliExpGate, circKHK, GBAt
from functions.circuit_builder_funcs import build_circuit, build_circuit_manual_linear, build_circuit_manual_reduced_k_linear
from functions.processing_funcs import GfromCircuit, getFit



def CreateExperiment(Uval, Vstart, iterations, attemptNumber, fileName='', fs=1.5, sampleTime=20,  tNpt = 150, provider='open', backend='ibmq_manila', randomCartan=True, shots=8000, qubitAssignments=[0,1,2,3,4]):
    """
    Submits the job to IBMQ - 
    
    """
    #qubitAssignments = [0,1,2,3,4] #Santiago
    qubits = 5
    IBMQ.load_account()
    HoneywellBack = False
    IBMback = True
    method = 'KHK' #Or add Trotter Steps
    optimizerLevel = 3
    if provider == 'open':
        ncsu_provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')#
        backend = ncsu_provider.get_backend(backend)
    elif provider == 'ncsu':
        ncsu_provider = IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='physics-of-spin-')
        backend = ncsu_provider.get_backend(backend)

    elif provider == 'ornl':
        ncsu_provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy141') 
        backend = ncsu_provider.get_backend(backend)

    elif provider == 'honeywell':
        #Honeywell.save_account('tmsteckm@ncsu.edu')
        #Honeywell.load_account()
        #backend = Honeywell.get_backend('HQS-LT-S1-SIM')
        #ncsu_provider = None
        #HoneywellBack = True
        pass

    if randomCartan:
        randomGuesses = iterations
    else:
        randomGuesses = None

    sites = 2
    mu = Uval/2
    ei = [0, mu]

    # Criteria for the least_squares_fit
    maxFits= 10000
    Convergence = 1e-4

    #Get the Green's functions values from Job results:
    Greenslist = [['cx','cx', 0.5 ]]


    """                      HAMILTONIAN                 """
    #Builds out the Hamiltonian and the Cartan Decomposition of the function
    #Open Fermion Building the Hamiltonian
    OFAIMham = AIM(sites, Uval, Vstart, mu, ei=ei) #sites, U, t, mu #3969 for AIM
    OFAIMham.setHamiltonian() #Creates the Hamiltonian using OpenFermion
    #Assemble Hamiltonain Object
    ham = Hamiltonian(sites*2) #Create empty object
    AIMhamOps = OFAIMham.opsToCartanFormat() #Converts from OpenFermion to CQS format
    ham.addTerms(AIMhamOps) #Add the Converted openFermion Hamiltonian
    ham.removeTerm((0,0,0,0)) #Remove idenity term
    AIMc = Cartan(ham, involution='countY')
    if not randomGuesses == False:
        FindParameters_list = []
        if not randomGuesses == iterations:
            raise Exception('Iterations must equal number of Parameter Guesses')
        while len(FindParameters_list) < randomGuesses:
            try:
                currentGuess = np.random.rand(8)
                AIMp = FindParameters(AIMc, initialGuess=currentGuess)
                AIMp.printResult()
                FindParameters_list.append(AIMp)
            except Exception as e:
                print(e)
                
                
    else:
        AIMp = FindParameters(AIMc)

    freqFileName = fileName + 'F_{}'.format(fs) + '_v_' + str(attemptNumber)
    os.mkdir(freqFileName)
    os.chdir(freqFileName)

    #Create Lists to store Data 
    #tNpt = #int(fs*sampleTime)
    times = np.linspace(0.0001, sampleTime, tNpt)
    iGt_imp = np.zeros(tNpt,dtype=complex)
    if not randomGuesses == False:
        circuit_list = []
        for i in range(randomGuesses):
            circuit_list += build_circuit_manual_reduced_k_linear(FindParameters_list[i], ham, tNpt, times, Greenslist)
        print('Transpiling')
        transpiled = transpileJobs(circuit_list, backend, level = optimizerLevel, layout = qubitAssignments)
        print('Transpiled')
    print('Job Created with Parameters: V = {}, U = {}, fs = {}'.format(Vstart, Uval, fs))
    if randomGuesses == False:
        circuit_list = build_circuit_manual_reduced_k_linear(AIMp, ham, tNpt, times, Greenslist)
        transpiled = transpileJobs(circuit_list, backend, level = optimizerLevel, layout = qubitAssignments)
        newtranspiled = transpiled.copy()
        for i in range(iterations-1):
             newtranspiled += transpiled
        transpiled = newtranspiled

    qr = qiskit.QuantumRegister(5)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubitAssignments, circlabel='mcal')

    cxCounter = [i.count_ops()['cx'] for i in transpiled]
    if HoneywellBack:
        #measurementJob = sendJobs(meas_calibs, backend, fileName, shots, IBMback=IBMback,Measurement=True)
        job = sendJobs(transpiled, backend, freqFileName, shots, IBMback=IBMback,Measurement=False)
        id = job.job_id()
        #measureid = measurementJob.job_id()
        #print(measureid)
        print(id)
        measureid = False
        saveCircuitInfo(freqFileName, times, tNpt, transpiled, Uval, Vstart, fs, sampleTime, id, backend, measureid,shots,state_labels,Greenslist,iterations)
    else:
        measurementJob = sendManagedJobs(meas_calibs, backend, freqFileName, shots, IBMback=IBMback, Measurement=True)
        print('Measurement Job Sent')
        while not ((measurementJob.statuses()[0] is JobStatus.QUEUED) or (measurementJob.statuses()[0] is JobStatus.DONE)):
            time.sleep(60)
        job = sendManagedJobs(transpiled, backend, freqFileName, shots, IBMback=IBMback)
        while not (np.equal(job.statuses(),JobStatus.DONE).all()):
            time.sleep(60)
        id = job.job_set_id()
        measureid = measurementJob.job_set_id()
        print(measureid)
        print(id)
        saveCircuitInfo(freqFileName, times, tNpt, transpiled, Uval, Vstart, fs, sampleTime, id, backend, measureid,shots,state_labels,Greenslist,iterations)
        print('Circuit Information Saved')
        
    os.chdir('../')
    
