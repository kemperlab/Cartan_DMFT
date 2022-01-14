
#Master Script to manage DMFT iterations
from numpy.lib.npyio import savez
from Step1 import CreateExperiment
from time import time
from datetime import date
import numpy as np
from Step2 import extractFrequencyHigh, extractFrequencyLow, toHerz, toRad
import os
from numpy import savez

#Essential Parameters
expected_max_w1 = 0.5
expected_max_w2 = 1.5
Uval = 1
Vstart = 0.5
shots = 8000

w1Delta = expected_max_w1  # expected_max_w1*0.7 #So it doesn't pick up zero frequencies
w2Delta = 1.5  # 1.5  #Expected seperation. Might be too big for some low U. Can be adjusted
#IBM systems - Modify "Provider" as needed to change how you send jobs
system = 'ibmq_qasm_simulator'#ibmq_manila'#'ibmq_qasm_simulator'
provider = 'open'

"""
Known valid parameters:
U = 1; w1 = 0.5, w2 = 1.5; Vstart = 0.5, 
U = 2; w1 = 0.75; w2 = 2; Vstart = 0.5; w1Delta = 0.9*w1; w2Delta = 1.5
0.9 is too small to get from step 2 to step 3
Same for U = 3 and U = 2
U = 4; w1 = 0.75; w2 = 3; Vstart = 0.5; w1Delta = 0.9*w1; w2Delta = 1.5
U = 5; same as U = 4
U = 5.5 the same
U = 6.5: Need to lower the lowpass filter range. Peak from Aliased Filter bleeds in
Parameters: w1 = 0.3, w2 = 4, expected_w1 = 0.5
"""


#Sampling Rates (Automatic)
fs2 = toHerz(expected_max_w2) * 4 #2 times the nyquist frequency
fs1 = toHerz(expected_max_w1) * 6 #6 times the nyquist frequency - import because we need to push the aliased frequency to above 3 * the expected frequency

#Number of times for each evaluation of frequency
circuits2 = 150
circuits1 = 150
#The stop times, used in linspace
ft1 = circuits1/fs1
ft2 = circuits2/fs2
#Number of cartan variants to use. Increasing it should decrease noise
iterations = 2

#Initialization
attempt_list = [0]
V_list = [Vstart]
iteration = 0
tolerance = 0.02

#Computing th exact value for the final convergence. Used for comparison and for termination
temp = max(36 - Uval**2, 0)
V_actual = np.sqrt(temp)/6
#There must be 3 evaluations (consecutive) within tol of eachother or of the actual value
Converged = 20

#Creates the generalized File Name
today = date.today()
currentDate = today.strftime("%m%d%y")
randID = str(np.random.rand(1)[0])[2:5]

#File system management (put everything in the jobs folder. Could be better, especially for looking back at results)
os.chdir('jobs')

#Creates a file to store outputs
data_file_name = 'Current_Data_U{}_{}_{}_.txt'.format(Uval, currentDate,randID)
data_file = open(data_file_name, 'a')
data_file.write('V_new, attemptNumber, iteration, w1_new, w2_new, Converged, FFTpassed \n')
data_file.close()


#Ends the loop if "converged," see previous documentation
while not Converged == 0:
    FFTpassed = False #Verify that the detected frequencies pass the tests
    #local attempt, within one iteration
    attemptNumber = 0
    #Step # in the loop
    iteration += 1
    #If FFT passes, end the loop and store the results + move on in the iterations
    while not FFTpassed:
        #Advance local attempt counter
        attemptNumber += 1
        folderName = 'U_{}_V_{}_it_{}_{}_{}_'.format(Uval, str(Vstart), iteration, currentDate,randID)
        attempt_list.append(attemptNumber)
        print('Sending Experiment (High Frequency): Iteration Number {}, Attempt Number {} \n Sample Frequency {}'.format(iteration, attemptNumber, fs2))
        print('Other Important Parameters: U = {}, V = {}, sampleTime = {}, Previous w2 = {}'.format(Uval, Vstart, ft2, expected_max_w2))
        CreateExperiment(Uval, Vstart, iterations, attemptNumber, fileName=folderName, fs=fs2, sampleTime=ft2,  tNpt = circuits2, provider=provider, backend=system, randomCartan=True, shots=shots, qubitAssignments=[0,1,2,3,4])
        try:
            w2_co_w_list, sigma_2, avg_2, iGreal_2, dw = extractFrequencyHigh(folderName, fs2, circuits2, attemptNumber, expected_max_w1)
        except Exception as e:
            print(e)
            print(os.getcwd())
            print(folderName)
            print(attemptNumber)
        data_file = open(data_file_name, 'a')
        data_file.write("co_w_list_2:, {}\n".format(w2_co_w_list))
        data_file.close()
        #First Check: How many peaks appeared in the data? If more than 2 (2 positive, 2 negative) #ibmq_manila
        if len(w2_co_w_list) > 4 or len(w2_co_w_list)<1:
            data_file = open(data_file_name, 'a')
            data_file.write("Did Not Pass the length requirements: {}\n".format(len(w2_co_w_list)))
            data_file.close()
            continue #Break the loop before computing the low frequency
        print('w2 Frequency List = {}'.format(w2_co_w_list))
        #Confirm from Raw Peaks:
        #Checks if either of the peak frequencies are within the expected range
        w2_early = 10
        for item in w2_co_w_list:
            w_test = abs(item[1])
            print(abs(w_test - expected_max_w2))
            if abs(w_test - expected_max_w2) < w2Delta:
                w2_early = w_test
                break
        if w2_early == 10:
            print('No Valid values of w2 found: w2 Min = {}, w2 Max = {}'.format(expected_max_w2 - w2Delta, expected_max_w2 + w2Delta))
            
            continue #If no valid w2, continue the w2 loop 
        print('w2 Frequency = {}'.format(w2_early))
        omega1 = 0.25*(np.sqrt(Uval**2 + 64*Vstart**2) - np.sqrt(Uval**2 + 16*Vstart**2))
        omega2 = 0.25*(np.sqrt(Uval**2 + 64*Vstart**2) + np.sqrt(Uval**2 + 16*Vstart**2))
        Z_expected = (omega1**2)*(omega2**2)/(Vstart**2*(omega1**2 + omega2**2 - Vstart**2))
        print('Expected Values: w1 = {}, w2 = {}, V_new = {} '.format(omega1, omega2, np.sqrt(Z_expected)))
        
        
        freq2 = toHerz(w2_early)
        #Compute the region in which fs2 will place the omega_1 alias inside the region expected omega_1:
        
        highPerceived = abs(freq2 - fs1*np.rint(freq2/fs1))
        print('Perceived Alias Frequency (Initial): {} Radians/s'.format(toRad(highPerceived)))
        threshold = toHerz(expected_max_w1)*5 # We want the aliased frequency to be above 3*w
        maxAlias = (highPerceived, fs1)
        print('Alias Location Threshold: {} Radians/s'.format(toRad(threshold)))
        if (abs(highPerceived) > threshold):
            pass
        else:
            fs1 = toHerz(expected_max_w1)*3.1 #Starts with the guess of fs1 = expected * 6 / 2 = expected * 3
            
            while (abs(highPerceived) < threshold): #If the percieved frequency is below the threshold or the sample rate is outside the bounds 0.5 - 1:
                #Compute the new aliased frequency
                highPerceived = abs(freq2 - fs1*np.rint(freq2/fs1))
                if highPerceived > maxAlias[0]:
                    maxAlias = (highPerceived, fs1)
                a = highPerceived > threshold #Condition 1, it must be over the threshold
                b = (fs1 < toHerz(expected_max_w1) * 10) #Condition 1: Must be inside the bounds
                c = (fs1 > toHerz(expected_max_w1) * 3.1)
                if (a):
                    print("Above Threshold with sampling rate {}*w1_Guess".format(fs1/toHerz(expected_max_w1)))
                    print(fs1)
                    if (b and c):
                        maxpair = (highPerceived, fs1)
                        #Ensures the fs1 selected is optimal locally
                        for test_fs in np.linspace(fs1*.8, fs1*1.2, 10000):
                            test_alias = abs(freq2 - fs1*np.rint(freq2/fs1))
                            if test_alias > maxpair[0]:
                                maxpair = (test_alias, test_fs)
                        fs1 = maxpair[1]
                        break
                if fs1 >  toHerz(expected_max_w1) * 10:
                    print('initial_divisor too large')
                    highPerceived = maxAlias[0]
                    fs1 = maxAlias[1]
                    break
                fs1 *= 1.01
        #Sometimes this method fails due to numerical stability, so we need to additionally mask out 
        lowAttempt = 0
        constant_freq2 = freq2
        while lowAttempt < 3:
            lowAttempt += 1
            freq2 = constant_freq2 + 0.5*(1-1/lowAttempt)*((-1)**lowAttempt)*dw #Shifts the frequency a bit for each attempt, might allow for moving the alias frequency
            highPerceived = abs(freq2 - fs1*np.rint(freq2/fs1))
            print('Perceived Alias Frequency (Final, Radians/s): {}'.format(toRad(highPerceived)))
            ft1 = circuits1/fs1
            t1 = np.linspace(0,ft1,circuits1)
            print('Sending Experiment (Low Frequency): Iteration Number {}, Attempt Number {} \n Sample Frequency {}'.format(iteration, attemptNumber, fs1))
            print('Other Important Parameters: U = {}, V = {}, sampleTime = {}, Previous w1 = {}'.format(Uval, Vstart, ft1, expected_max_w1))
            CreateExperiment(Uval, Vstart, iterations, attemptNumber, fileName=folderName, fs=fs1, sampleTime=ft1,  tNpt = circuits1, provider=provider, backend=system, randomCartan=True, shots=shots, qubitAssignments=[0,1,2,3,4])
            try:
                w1_co_w_list, sigma_1, avg_1, iGreal_1= extractFrequencyLow(folderName, fs1, circuits1, w2_early, dw, attemptNumber, expected_max_w1)
            except Exception as e:
                print(e)
                print(os.getcwd())
                print(folderName)
                print(attemptNumber)
            attemptNumber += 0.1
            print('w1 Frequency List = {}'.format(w1_co_w_list))
            print('Expected Values: w1 = {}, w2 = {}, V_new = {} '.format(omega1, omega2, np.sqrt(Z_expected)))
            data_file = open(data_file_name, 'a')
            data_file.write("co_w_list_1, {}, co_w_list_2, {}\n".format(w1_co_w_list, w2_co_w_list))
            data_file.close()
            
            if len(w1_co_w_list) > 4 or len(w1_co_w_list) < 1:
                data_file = open(data_file_name, 'a')
                data_file.write("w1 did not pass the length requirements: {}\n".format(len(w1_co_w_list)))
                data_file.close()
                continue
            else:
                #Break the loop because there are too many or two few frequencies
                lowAttempt = 4
            
            #Pick two frequencies which satisfy:
            # 1) They are not within the delta of the frequency evaluation: If both samples pick the frequency, take the one with lower error
            # 2) They are within expected*.9 of the old low frequency and within 1 of the old high frequency
            # 3) both w1 and w2 are sorted in terms of prominence, so we stop at the first good match of one from each
            print('Checking List against w1 old = {} and w2 old = {}'.format(expected_max_w1, expected_max_w2))
            for (a1, w1) in w1_co_w_list:
                for (a2, w2) in w2_co_w_list:
                    #If we already found the best w2 and w1, don't continue
                    if not FFTpassed:
                        #Verifies they are not in the same region (within the domain spacing of the w2 FFT)
                        if abs(abs(w1) - abs(w2)) > dw:
                            #Sorts the frequencies so w1 is min. This should already be done though, but just in case
                            w1_new = min(abs(w1), abs(w2))
                            w2_new = max(abs(w1), abs(w2))
                            #Verifies the new value is near the previous iteration
                            if abs(w1_new - expected_max_w1) > w1Delta:
                                print('Bad Low Freq: Old:{}, new:{}'.format(expected_max_w1, w1_new))
                                continue
                            if abs(w2_new - expected_max_w2) > w2Delta:
                                print('Bad High Freq: old:{}, new{}'.format(expected_max_w2, w2_new))
                                continue
                            print('FFT Tests Passed')
                            FFTpassed = True
                            break
                        else:
                            print('Bad Freq Spacing: {}, {}, max(dw) = {}'.format(w1, w2,dw))
                            continue
                    else:
                        print('Already Accepted w1 and w2: {}, {}, Passed = {}'.format(w1_new, w2_new, FFTpassed))
                        pass
            try:
                print('w1 = {}, w2 = {}, Passed = {}'.format(w1_new, w2_new, FFTpassed))
            except:
                print('Failed to extract w1 and w2')
                continue
            if not FFTpassed:
                print('Failed to Pass FFT Tests')
                continue
            else:
                print('Approved Low Frequency Tests')
                lowAttempt = 5
        print('Continuing Analysis')
        if lowAttempt > 2 and not FFTpassed :
            print('Low Frequency Failed')
            print('Reverting DMFT')
            continue
        print('Computing Z')
        #Now the the loop is terminated, we can compute the new V
        Z = (w1_new**2)*(w2_new**2)/(Vstart**2*(w1_new**2 + w2_new**2 - Vstart**2))
        V_new = np.sqrt(Z)
        
        #Ending the Loop requires two solutions within the desired tolerance of the actual correct answer
        
        if abs(V_new - V_actual) < tolerance:
            Converged += -1
        elif abs(V_new - Vstart) < tolerance:
            Converged += -1
        else:
            Converged = 20
            
        expected_max_w1 = w1_new
        expected_max_w2 = w2_new
        fs2 = toHerz(expected_max_w2) * 4
        fs1 = toHerz(expected_max_w1) * 6
        w1Delta = expected_max_w1
        ft1 = circuits1/fs1
        ft2 = circuits2/fs2
        
        V_list.append(V_new)
        Vstart = V_new
        print('Computed new V: {}'.format(Vstart))
        data_file = open(data_file_name, 'a')
        data_file.write("V new, {}, V expected, {}, Attempt Number, {}, Iteration Number, {}, w1, {}, w2, {},w1_actual, {}, w2_actual, {}, Converged, {}, FFTpassed, {}  \n".format(V_new, np.sqrt(Z_expected) ,attemptNumber, iteration, w1_new, w2_new, omega1,omega2, Converged, FFTpassed))
        data_file.close()
        savez('Final_Data_U_{}_ID_{}'.format(Uval, randID), V_list=V_list, attempt_list=attempt_list, Uval=Uval, frequencies=(fs1, fs2), circuitLengths=(circuits1, circuits2))
print('Ended')
savez('Final_Data_U_{}_ID_{}'.format(Uval, randID), V_list=V_list, attempt_list=attempt_list, Uval=Uval, frequencies=(fs1, fs2), circuitLengths=(circuits1, circuits2))
print('Saved')


    


