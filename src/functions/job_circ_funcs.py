"""
Contains:

 - Packaging Circuits from a list
 - Write Jobs to files + save job IDs to file
 - Read Job IDs from file
 - Wait for Jobs to finish from IBM
"""
from qiskit.compiler.transpiler import transpile
from qiskit.ignis.mitigation.measurement.filters import MeasurementFilter
from qiskit.providers.ibmq.managed import IBMQJobManager
#from pytket.extensions.qiskit import tk_to_qiskit, qiskit_to_tk
#from pytket.passes import DecomposeBoxes, FullPeepholeOptimise
#from pytket.transform import PauliSynthStrat, CXConfigType
import pickle
import shutil
import os
import csv
#from tkinter import *
import sys
sys.path.append(".") # Adds higher directory to python modules path.
from qiskit.providers import JobStatus
import time
# import filedialog module
#from tkinter import filedialog

from qiskit.execute_function import execute


CURRENTJOB = 'currentJob/'
OLDJOBS = 'oldJobs/'

def sendManagedJobs(transpiled, backend, fileName, shots, tket=False, infoDict=None, IBMback = True, Measurement=False):
    """
    Sends jobs to the backend and saves the job IDs to a file. Returns the Job ID(s)

    Args:
        circ_list (list): list of Circuit objects
        backend (IBMBackend): IBM backend object
        fileName (str): name of the file to save the job IDs to
        shots (int): number of shots to run the circuit
    """
    path = os.path.abspath(os.getcwd())

    if IBMback:
        job_manager = IBMQJobManager()
        job = job_manager.run(transpiled, backend=backend, shots=shots)
        while not ((job.statuses()[0] is JobStatus.QUEUED) or (job.statuses()[0] is JobStatus.DONE)):
            time.sleep(10)
        #job = backend.run(transpiled, shots=shots)
        # Save Job ID to the end of the file
        print('Job Sent, Saving ID:' + str(job.job_set_id()))
        try:
            with open(os.path.join(path, fileName + '_index.csv'), 'a') as f:
                #Saves the Job ID
                f.write(str(job.job_set_id()) + ',')
                #Saves the backend Name
                f.write(str(backend.name()) + ',')
                #Saves the file location of the results object
                if not Measurement:
                    resultFileName = 'results' + '.pkl'
                else:
                    resultFileName = 'measurementMatrix' + '.pkl'
                f.write(resultFileName + '\n')
        except Exception as e:
            print(e)
            print('\n\n\nJob ID saving Failed.\n\n\n')
        return job
    else:
        return backend.run(transpiled, shots=shots)

def getManagedJobResults(job, fileName, backend, jobisJob=True, measurement=False):
    """
    Returns the results from a job object or a job ID

    Args:
        job (Job or Str): Job object or Job ID
        fileName (str): name of the file to save the results to
        backend (IBMBackend): IBM backend object (needed to get results from Job ID
        jobisJob (bool): True if job is a Job object, False if job is a Job ID

    """
    
    if jobisJob: #Case when the job Object is 
        try:
            job_result = job.results().combine_results()  # It will block until the job finishes.
        
            print("The job finished")
            #Saves the job_result by appending it to the file
            try:
                if measurement:
                    save_obj(job_result, 'measurementMatrix')
                else:
                    save_obj(job_result,  'results')
            except:
                print('\n\n\nJob Results saving Failed.\n\n\n')
            return job_result
        except Exception as ex:
            print("Something happened! The Job object failed to return a Result. Attemping get the job from the ID: {}".format(ex))
            return getJobResults(job.job_set_id(), fileName, jobisJob=False)
    else:
        job_manager = IBMQJobManager()
        job = job_manager.retrieve_job_set(job, backend)
        job_result = job.results().combine_results()  # It will block until the job finishes.

        #Saves the job_result by appending it to the file
        try:
            if measurement:
                save_obj(job_result, 'measurementMatrix')
                print('Measurement Matrix Saved ID = {}'.format(job))
            else:
                save_obj(job_result, 'results')
                print('Results Saved ID = {}'.format(job))
        except:
            print('\n\n\nJob Results saving Failed.\n\n\n')
        return job_result

def loadMostRecentJob(provider=None, managedJob = False, loadResults = False,measurement=True):
    """
    Loads in the most recent Job from the folder currentJob

    Returns:
        The most recent Results object
    """
    file_names = os.listdir()
    resultsToReturn = []
    for file in file_names:
        if file[-3:] == 'csv':
            counter = 0
            with open(file, newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                #Really bad way to do this, but it works
                for row in reader:
                    returnLine = row
                    resultsFileName = row[-1]
                    if counter == 0:
                        if measurement:
                            MeasurementFilter = True
                        else:
                            resultsToReturn.append(None)
                            counter += 1
                            continue
                            #MeasurementFilter = False
                    else:
                        MeasurementFilter = False
                    counter += 1
                    try:
                        if loadResults:
                            results = load_obj(resultsFileName[:-4])
                            print('Loaded Results From File')
                        else:
                            raise Exception('Load From IBM Backend')
                    except:
                        Id = returnLine[0]
                        print('Loading From ID:' + str(Id))
                        if managedJob:
                            results = getManagedJobResults(Id, file[:-3], provider, jobisJob=False, measurement=MeasurementFilter)
                            print('Loaded Results from JobIDs')
                            #job_manager = IBMQJobManager()
                            #job = job_manager.retrieve_job_set(Id, provider=provider)
                            #results = job.results().combine_results()
                        else:
                            print('Awaiting Results')
                            results = getJobResults(Id, file[:-3], provider, jobisJob=False, measurement=MeasurementFilter)
                            print('Retrived Results')
                            #backend = provider.get_backend(returnLine[1])
                            #job = backend.retrieve_job(Id)
                            #results = job.result()
                    resultsToReturn.append(results)
            return resultsToReturn
    


def sendJobs(transpiled, backend, fileName, shots, tket=False, infoDict=None, IBMback = True, Measurement=False):
    """
    Sends jobs to the backend and saves the job IDs to a file. Returns the Job ID(s)

    Args:
        circ_list (list): list of Circuit objects
        backend (IBMBackend): IBM backend object
        fileName (str): name of the file to save the job IDs to
        shots (int): number of shots to run the circuit
    """
    if IBMback:
        #job_manager = IBMQJobManager()
        #job = job.run(transpiled, backend=backend, shots=shots)
        job = execute(transpiled, backend, shots=shots)#job = backend.run(transpiled, shots=shots)
        # Save Job ID to the end of the file
        print('Job Sent, Saving ID:' + str(job.job_id()))
        try:
            with open('index.csv', 'a') as f:
                #Saves the Job ID
                f.write(str(job.job_id()) + ',')
                #Saves the backend Name
                f.write(str(backend.name()) + ',')
                #Saves the file location of the results object
                if not Measurement:
                    resultFileName = 'results' + '.pkl'
                else:
                    resultFileName = 'measurementMatrix' + '.pkl'
                f.write(resultFileName + '\n')
        except:
            print('\n\n\nJob ID saving Failed.\n\n\n')
        return job
    else:
        return execute(transpiled, backend, shots=shots)#) backend.run(transpiled, shots=shots)
def getJobResults(job, fileName, backend, jobisJob=True, measurement=False):
    """
    Returns the results from a job object or a job ID

    Args:
        job (Job or Str): Job object or Job ID
        fileName (str): name of the file to save the results to
        backend (IBMBackend): IBM backend object (needed to get results from Job ID
        jobisJob (bool): True if job is a Job object, False if job is a Job ID

    """
    if jobisJob: #Case when the job Object is 
        try:
            job_result = job.result()  # It will block until the job finishes.
        
            print("The job finished")
            #Saves the job_result by appending it to the file
            try:
                if measurement:
                    save_obj(job_result, 'measurementMatrix' )
                else:
                    save_obj(job_result, 'results' )
            except Exception as ex:
                print('\n\n\nJob Results saving Failed.\n\n\n')
                print(ex)
            return job_result
        except Exception as ex:
            print("Something happened! The Job object failed to return a Result. Attemping get the job from the ID: {}".format(ex))
            return getJobResults(job.job_id(), fileName, jobisJob=False)
    else:
        print('Loading Job from ID')
        job = backend.retrieve_job(job)
        print('Status')
        #print(job.status())
        print('That was the Status')
        job_result = job.result()  # It will block until the job finishes.
        print("The job finished with result {}".format(job_result))
        #Saves the job_result by appending it to the file
        try:
            if measurement:
                save_obj(job_result, 'measurementMatrix' )
            else:
                save_obj(job_result, 'results' )
        except:
            print('\n\n\nJob Results saving Failed.\n\n\n')
        return job_result

"""def loadMostRecentJob(provider=None):

    Loads in the most recent Job from the folder currentJob

    Returns:
        The most recent Results object
    
    file_names = os.listdir(CURRENTJOB)

    for file in file_names:
        if file[-3:] == 'csv':
            with open(CURRENTJOB + file, newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                #Really bad way to do this, but it works
                for row in reader:
                    returnLine = row
                    resultsFileName = row[-1]
            break
    print(returnLine)
    print(resultsFileName)
    try:
        results = load_obj(CURRENTJOB + resultsFileName)
        return results
    except:
        Id = returnLine[0]
        backend = provider.get_backend(returnLine[1])
        job = backend.retrieve_job(Id)
        results = job.result()
        return results, job
    """

        

def clearCurrentJob():
    """
    Moves the contents of currentJob to oldJob
    """
    file_names = os.listdir()
    for file_name in file_names:
        shutil.move(os.path.join(CURRENTJOB, file_name), OLDJOBS)
        



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def transpileJobs(circuit_list, backend, usetket = False, level=1, layout = None):
    if usetket:
        tket_transpiled = []
        for circuit in circuit_list:
            circUt_tket = qiskit_to_tk(circuit)
            FullPeepholeOptimise().apply(circUt_tket)
            tk_circ = tk_to_qiskit(circUt_tket)
            tket_transpiled.append(tk_circ)

        transpiled = transpile(tket_transpiled, backend, optimization_level=level)
    else:
        if layout is None:
            transpiled = transpile(circuit_list, backend, optimization_level=level)
        else:
            transpiled = transpile(circuit_list, backend, optimization_level=level, initial_layout=layout)

    return transpiled


def saveFullInfo(fileName, timeList, tNpt, Gimp, results, U, V, sampleFrequency, finalTime, circuitList, measureFilter, params, iterations,):
    with open('fullData.pkl','wb') as f:
        pickle.dump([timeList, tNpt, Gimp, results, U, V, sampleFrequency, finalTime, fileName, circuitList, measureFilter, params,iterations], f, pickle.HIGHEST_PROTOCOL)

def saveCircuitInfo(fileName, timeList, tNpt, circuitList, U, V, sampleFrequency, finalTime, jobID, backend, filterJobID,shots,state_labels,GreensList,iterations):
    with open('circuitData.pkl','wb') as f:
        pickle.dump([ timeList, tNpt, circuitList, U, V, sampleFrequency, finalTime, jobID, backend, filterJobID,shots,state_labels, GreensList, iterations], f, pickle.HIGHEST_PROTOCOL)
        
def loadFullInfo(fileName):
    """
    Returns  timeList, tNpt, Gimp, results, U, V, sampleFrequency, finalTime, fileName, circuitList, measureFilter, params
    """
    with open('fullData.pkl','rb') as f:
        return pickle.load(f)

def loadCircuitInfo(fileName):
    """
    Returns times, tNpt, circuitList, Uval, Vstart, sampleFrequency, finalTime, jobID, backendName, filterJobID
    """
    with open('circuitData.pkl','rb') as f:
        return pickle.load(f)

"""
def chooseFile(initialDir):
    # Python program to create
    # a file explorer in Tkinter
    
    # import all components
    # from the tkinter library

    
    # Function for opening the
    # file explorer window
    filename = filedialog.askdirectory(initialdir = initialDir,
                                        title = "Select a File",
                                        )
"""  
"""
    # Change label contents
    #label_file_explorer.configure(text="File Opened: "+filename)
    
    
                                                                                                    
    # Create the root window
    window = Tk()
    
    # Set window title
    window.title('File Explorer')
    
    # Set window size
    window.geometry("500x500")
    
    #Set window background color
    window.config(background = "white")
    
    # Create a File Explorer label
    label_file_explorer = Label(window,
                                text = "File Explorer using Tkinter",
                                width = 100, height = 4,
                                fg = "blue")
    
        
    button_explore = Button(window,
                            text = "Browse Files")
    
    button_exit = Button(window,
                        text = "Exit",
                        command = exit)
    
    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_file_explorer.grid(column = 1, row = 1)
    
    button_explore.grid(column = 1, row = 2)
    
    button_exit.grid(column = 1,row = 3)
    
    # Let the window wait for any events
    window.mainloop()
"""
"""    splitFileName = filename[filename.rindex('/')+1:]
    return splitFileName"""