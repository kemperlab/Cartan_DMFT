from functions.DMFT_funcs import getFitFunc, func
import numpy as np
from scipy.optimize import curve_fit
import scipy as sci
from numpy.fft import fft, fftfreq
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 12}
matplotlib.rc('font', **font)
axFontSize = 16
exactColor = '#1E88E5'
dataColor = '#EC921B'
dottedLineStyle = (0, (5, 5))

def GfromCircuit(tNpt, results, shots, Greenslist, iGt_imp,readoutCheck = True, meas_filter=None, iterations=1, iterationsStart = 0, ensemble=False):
    if not (meas_filter is None):
        results = meas_filter.apply(results)
    for iterationCounter in range(iterationsStart , iterations + iterationsStart):
        badshotsList = np.zeros(tNpt)
        for tt in range(tNpt):
            currentSum = 0
            for i in range(len(Greenslist)): #Only Use XX
                expectation = 0
                #print('Shot from Experiment Number {}'.format(tNpt*iterationCounter + tt + i))
                count = results.get_counts(tNpt*iterationCounter + tt + i)
                #Allows for removing shots from the data set
                badshots = 0
                for key in count.keys():
                    #Particle Count Check:
                    particleCheck = sum([int(i) for i in key[:-1]]) == 2
                    #Spin Parity Check:
                    firstSector = sum([int(i) for i in key[:-3]])
                    secondSector = sum([int(i) for i in key[2:-1]])
                    spinCheck = firstSector == secondSector
                    if not readoutCheck:
                        spinCheck = True
                        particleCheck = True
                    if particleCheck and spinCheck: #Particle Count Check
                        if key[-1] == '0':               
                            expectation += count[key]
                        else:                
                            expectation -= count[key]
                    else:
                        badshots += count[key]
                badshotsList[tt] += badshots
                currentSum += Greenslist[i][2]*expectation*2/(shots - badshots) #Updated to only use XX
                #print(currentSum,badshots, 'Circuit: ' + str(tt) + '/' + str(tNpt)) if (tt % 5 == 0) or (tt == tNpt-1) else None
            iGt_imp[tt] += currentSum
        print('Iteration = {}, iterationsStart = {}'.format((iterations, iterationCounter), iterationsStart))
        #print(iGt_imp)
        #print('Badshots:')
        #print(badshotsList)
        print('Average Badshots') 
        print(sum(badshotsList)/tNpt)
    iGt_imp = iGt_imp/iterations
    return [i.real for i in iGt_imp]

from scipy.signal import find_peaks
def peak_prom(f, times, w):
    y = abs(fft(f))
    avg = np.sum(y)/len(y)
    sigma = np.std(y)
    peaks = find_peaks(y, height=(avg + 2*sigma))[0]
    print('Peaks = {}'.format(peaks))
    indices = peaks
    
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(times,f)
    ax2.plot(w,y)
    ax2.plot(w, np.ones(len(w))*(avg + 2*sigma))
    ax1.set_xlabel('Time')
    ax1.set_ylabel("Green's Function")
    ax2.set_title('FFT')
    ax2.set_xlabel(r'Angular Frequency ($\omega$)')
    ax2.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.set_size_inches(12, 10)

    plt.tight_layout()
    plt.savefig('plot.svg')
    plt.savefig('plot.png')
    
    return indices

def getSingleFreq(iGrealNewNormalized, sampleFrequency):
    yf = abs(fft(iGrealNewNormalized))
    yf = yf/max(yf)/8
    xw = fftfreq(n=len(iGrealNewNormalized), d=1/sampleFrequency)* 2*np.pi
    dw = abs(xw[0]-xw[1])
    yf = yf[:int(len(yf)/2)]
    xw = xw[:int(len(xw)/2)]

    fftPeaks = sci.signal.find_peaks(yf)
    initialfreq = [i for i in xw[fftPeaks[0]]]
    initialCo = [i.real for i in yf[fftPeaks[0]]]

    zipped = [(c,w) for c,w in zip(initialCo,initialfreq) if w < 20]
    zipped.sort(key = lambda zipped: zipped[0])
    print(zipped)
    zipped = zipped[-1:]
    #pairs = itertools.permutations(zipped,2)
    #pairs = [a for a in pairs]
    return(zipped[1])


def getDualFreq(iGrealNewNormalized, sampleFrequency):
    yf = abs(fft(iGrealNewNormalized))
    yf = yf/max(yf)/8
    N = len(iGrealNewNormalized)
    xw = fftfreq(n=N, d=1/sampleFrequency)* 2*np.pi
    dw = abs(xw[0]-xw[1])
    yf = yf[:int(len(yf)/2)]
    xw = xw[:int(len(xw)/2)]
    dw = sampleFrequency/N
    fftPeaks = sci.signal.find_peaks(yf)
    initialfreq = [i for i in xw[fftPeaks[0]]]
    initialCo = [i.real for i in yf[fftPeaks[0]]]

    zipped = [(c,w) for c,w in zip(initialCo,initialfreq) if w < 20]
    zipped.sort(key = lambda zipped: zipped[0])
    print(zipped)
    zipped = zipped[-2:]
    return ([i[1] for i in zipped]), dw





def func (t, a1, a2,d, delta1, delta2, *args):
    w1,w2 = args
    #print('test')
    cos1 = np.cos(w1 * t + delta1)
    cos2 = np.cos(w2 * t + delta2)
    val = a1 * cos1 + a2 * cos2
    return 2*val + d

def getFitFunc(params):
    a1 = params[0]
    a2 = params[1]
    w1 = params[2]
    w2 = params[3]
    d = params[4]
    delta1 = params[5]
    delta2 = params[6]
    def Gt(t):
        cos1 = np.cos(w1 * t + delta1)
        cos2 = np.cos(w2 * t + delta2)
        val = a1 * cos1 + a2 * cos2
        return 2*val + d
    return Gt

def getFunctFreq(w1,w2):
    def func (t, a1, a2,d, delta1, delta2):
        cos1 = np.cos(w1 * t + delta1)
        cos2 = np.cos(w2 * t + delta2)
        val = a1 * cos1 + a2 * cos2
        return 2*val + d
    return func
import itertools
def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def getFit(times, iGreal, sampleFrequency, maxFits = 10000):
    yf = abs(fft(iGreal))
    yf = yf/max(yf)/8
    xw = fftfreq(n=len(iGreal), d=1/sampleFrequency)* 2*np.pi
    dw = abs(xw[0]-xw[1])
    yf = yf[:int(len(yf)/2)]
    xw = xw[:int(len(xw)/2)]
    
    """    fftPeaks = sci.signal.find_peaks(yf)
    print(fftPeaks)
    fftPeaks = [(index, yf[index]) for index in fftPeaks[0]]
    print(fftPeaks)
    fftPeaks.sort(key = lambda fftPeaks: fftPeaks[1])
    print(fftPeaks)
    fftPeaks = fftPeaks[-2:]
    print(fftPeaks)
    initialfreq = [parabolic(yf, pair[0])[0]/sampleFrequency/2 for pair in fftPeaks]
    #initialfreq = [parabolic(yf,index)[0]/sampleFrequency/2 for index in fftPeaks[0]]
    #initialfreq.append(xw[1])
    #Add the smallest frequency to the list b/c it is not included in fftPeaks
    initialParams = initialfreq.copy()
    initialCofreq = [parabolic(yf, pair[0])[1].real for pair in fftPeaks]
    #initialCofreq.append(abs(yf[1].real))
    peakIndex = [i for i in range(len(initialfreq))]
    initialCo = initialCofreq.copy()
    print('FFT analysis of iG(t).real. Sorted Peaks')
    print(initialParams)
    zipped = [(c,w) for c,w in zip(initialCo,initialParams) if w < 20]
    #zipped.sort(key = lambda zipped: zipped[0])
    print(zipped)
    #zipped = zipped[-2:]
    pairs = itertools.permutations(zipped,2)
    pairs = [a for a in pairs]
    #print(pairs)"""
    
    """# FFT to extract initial guess frequencies from Circuit Values
    yf = abs(fft(iGreal))
    yf = yf/max(yf)/8
    xw = fftfreq(n=len(iGreal), d=1/sampleFrequency)* 2*np.pi
    dw = abs(xw[0]-xw[1])
    yf = yf[:int(len(yf)/2)]
    xw = xw[:int(len(xw)/2)]
    """
    fftPeaks = sci.signal.find_peaks(yf)
    initialfreq = [i for i in xw[fftPeaks[0]]]
    #initialfreq.append(xw[1])
    #Add the smallest frequency to the list b/c it is not included in fftPeaks
    initialParams = initialfreq.copy()
    

    #initialCofreq = [(np.sqrt(i.real**2 + i.imag**2)) for i in yf[fftPeaks[0]]]
    #initialCofreq.append(np.sqrt(yf[1].real**2 + yf[1].imag**2))
    
    initialCofreq = [i.real for i in yf[fftPeaks[0]]]
    initialCofreq.append(abs(yf[1].real))
    peakIndex = [i for i in range(len(initialfreq))]
    initialCo = initialCofreq.copy()
    print('FFT analysis of iG(t).real. Sorted Peaks')
    print(initialParams)
    zipped = [(c,w) for c,w in zip(initialCo,initialParams) if w < 20]
    zipped.sort(key = lambda zipped: zipped[0])
    print(zipped)
    zipped = zipped[-2:]
    pairs = itertools.permutations(zipped,2)
    pairs = [a for a in pairs]
    print(pairs)

    #Least Squares Fit, Maximize the overlap by minimizing over the largest difference between two points
    MaxDiff = 1
    fitAttemptNumber = 0
    randomList = np.linspace(0.001, 20, 1000)
    #Least Squares Convergence Loop
    bestParams = [(1,1,1,1,0),200,0, (1,1,1,1,0)] #If there is nothing within the tolerance, gives the one that had the closest
    maxFits = len(pairs)
    while fitAttemptNumber < maxFits:
        
        #print(initialParams)
        if fitAttemptNumber < len(pairs):
            initialParams = [pairs[fitAttemptNumber][0][1], pairs[fitAttemptNumber][1][1]]
            maxCo = 0.125/(max(pairs[fitAttemptNumber][0][0], pairs[fitAttemptNumber][1][0]))
            initialCo = [pairs[fitAttemptNumber][0][0]*maxCo, pairs[fitAttemptNumber][1][0]*maxCo]
            #print(initialfreq)
        else:
            initialParams = []
            initialCo = []
            initialParams.append(np.random.choice(randomList))
            initialCo.append(0.125)
            initialParams.append(np.random.choice(randomList))
            initialCo.append(0.125)
        
        try:
            #print([initialCo[0], initialCo[1], initialParams[0], initialParams[1], 0])
            d = sum(iGreal)/len(iGreal)
            funcFreqs = getFunctFreq(initialParams[0], initialParams[1])
            print('Guess: {}'.format([initialCo[0], initialCo[1], initialParams[0], initialParams[1], d,0,0]))
            #params, params_covariance = sci.optimize.curve_fit(func, times, iGreal, p0=[initialCo[0], initialCo[1], initialParams[0], initialParams[1], d,0,0], xtol=1.0e-3, ftol=1.0e-3, maxfev=10000, bounds=([0,0,initialParams[0]-dw*2, initialParams[1]-dw*2,-1, 0, 0],[0.5,0.5,initialParams[0]+dw*2, initialParams[1]+dw*2,1,2*np.pi, 2*np.pi]))
            
            params, params_covariance = sci.optimize.curve_fit(funcFreqs, times, iGreal, p0=[initialCo[0], initialCo[1], d,0,0], xtol=1.0e-10, ftol=1.0e-10, maxfev=10000, bounds=([0,0,-1, 0, 0],[0.5,0.5,1,2*np.pi, 2*np.pi]))
            params=[params[0], params[1], initialParams[0], initialParams[1], params[2], params[3], params[4]]
            print('Returned:{}',format(params))
            
            Gt = getFitFunc(params)
            GofT = [Gt(t) for t in times]
            MaxDiff = sum(np.abs(np.array(GofT) - np.array(iGreal))**2)
            print('least-squares: {}\n'.format(MaxDiff))
            #if fitAttemptNumber < len(pairs):
            #    print(MaxDiff)
            #    print(params)
            if MaxDiff < bestParams[1]:
                bestParams = [params, MaxDiff, fitAttemptNumber, (initialCo[0], initialCo[1], initialParams[0], initialParams[1], 0)]
            params = bestParams[0]
            MaxDiffReturn = bestParams[1]
            FinalfitAttemptNumber = bestParams[2]
            initialGuess = bestParams[3]

        except:
            print('Error')
        fitAttemptNumber += 1
    print('Returned parameters of fit. Least Squares = '+str(MaxDiffReturn))
    print(params)
    Gt = getFitFunc(params)

    print('Initial Guess:')
    print(initialGuess)
    return Gt, params

def getFitFuncDual(times_Low, times_High, iGreal_High, iGreal_Low, sampleFrequency_Low, sampleFrequency_High, maxfits = 10000):
    # FFT to extract initial guess frequencies from Circuit Values
    yf_Low = abs(fft(iGreal_Low))
    yf_Low = yf_Low/max(yf_Low)/4
    xw_Low = fftfreq(n=len(iGreal_Low), d=1/sampleFrequency_Low)* 2*np.pi
    yf_Low = yf_Low[:int(len(yf_Low)/2)]
    xw_Low = xw_Low[:int(len(xw_Low)/2)]
    fftPeaks_Low = sci.signal.find_peaks(yf_Low)
    initialfreq_Low = [i for i in xw_Low[fftPeaks_Low[0]]]
    initialfreq_Low.append(xw_Low[1])
    #Add the smallest frequency to the list b/c it is not included in fftPeaks
    initialParams_Low = initialfreq_Low.copy()
    initialCofreq_Low = [i.real for i in yf_Low[fftPeaks_Low[0]]]
    initialCofreq_Low.append(abs(yf_Low[1].real))
    #peakIndex_Low = [i for i in range(len(initialfreq_Low))]
    initialCo_Low = initialCofreq_Low.copy()

    yf_High = abs(fft(iGreal_High))
    yf_High = yf_High/max(yf_High)/4
    xw_High = fftfreq(n=len(iGreal_High), d=1/sampleFrequency_High)* 2*np.pi
    yf_High = yf_High[:int(len(yf_High)/2)]
    xw_High = xw_High[:int(len(xw_High)/2)]
    fftPeaks_High = sci.signal.find_peaks(yf_High)
    initialfreq_High = [i for i in xw_High[fftPeaks_High[0]]]
    initialfreq_High.append(xw_High[1])
    #Add the smallest frequency to the list b/c it is not included in fftPeaks
    initialParams_High = initialfreq_High.copy()
    initialCofreq_High = [i.real for i in yf_High[fftPeaks_High[0]]]
    initialCofreq_High.append(abs(yf_High[1].real))
    #peakIndex_High = [i for i in range(len(initialfreq_High))]
    initialCo_High = initialCofreq_High.copy()
    HighNorm = max(initialCo_High)
    LowNorm = max(initialCo_Low)
    initialCo_High = [i / HighNorm for i in initialCo_High]
    initialCo_Low = [i / LowNorm for i in initialCo_Low]
    initialParams = [*initialParams_High , *initialParams_Low]
    initialCo = [*initialCo_High , *initialCo_Low]

    print('FFT analysis of iG(t).real. Sorted Peaks')
    zipped = [(c,w) for c,w in zip(initialCo,initialParams) if w < 20]
    zipped.sort(key = lambda zipped: zipped[0])
    print(zipped)
    pairs = itertools.permutations(zipped,2)
    pairs = [a for a in pairs]

    #Least Squares Fit, Maximize the overlap by minimizing over the largest difference between two points
    MaxDiff = 1
    fitAttemptNumber = 0
    randomList = np.linspace(0.001, 20, 1000)
    #Least Squares Convergence Loop
    bestParams = [(1,1,1,1,0),200,0, (1,1,1,1,0)] #If there is nothing within the tolerance, gives the one that had the closest
    times = [*times_Low , *times_High]
    iGreal = [*iGreal_Low , *iGreal_High]
    while fitAttemptNumber < maxfits:
        
        #print(initialParams)
        if fitAttemptNumber < len(pairs):
            initialParams = [pairs[fitAttemptNumber][0][1], pairs[fitAttemptNumber][1][1]]
            initialfreq = [pairs[fitAttemptNumber][0][0], pairs[fitAttemptNumber][1][0]]
        else:
            initialParams = []
            initialCo = []
            initialParams.append(np.random.choice(randomList))
            initialCo.append(0.125)
            initialParams.append(np.random.choice(randomList))
            initialCo.append(0.125)
        
        try:
            #print([initialCo[0], initialCo[1], initialParams[0], initialParams[1], 0])
            params, params_covariance = sci.optimize.curve_fit(func, times, iGreal, p0=[initialCo[0], initialCo[1], initialParams[0], initialParams[1], 0,0,0], xtol=1.0e-3, ftol=1.0e-3, maxfev=10000, bounds=([0,0,0,0,-1, 0, 0],[0.5,0.5,10,20,1,2*np.pi, 2*np.pi]))
            Gt = getFitFunc(params)
            GofT = [Gt(t) for t in times]
            MaxDiff = sum(np.abs(np.array(GofT) - np.array(iGreal))**2)
            if MaxDiff < bestParams[1]:
                bestParams = [params, MaxDiff, fitAttemptNumber, (initialCo[0], initialCo[1], initialParams[0], initialParams[1], 0)]
            params = bestParams[0]
            MaxDiffReturn = bestParams[1]
            FinalfitAttemptNumber = bestParams[2]
            initialGuess = bestParams[3]

        except:
            pass#print('Error')
        fitAttemptNumber += 1
    print('Returned parameters of fit. Least Squares = '+str(MaxDiffReturn))
    print(params)
    Gt = getFitFunc(params)

    print('Initial Guess:')
    print(initialGuess)
    return Gt, params