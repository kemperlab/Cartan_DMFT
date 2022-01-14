from CQS.methods import *
import matplotlib.pyplot as plt
from CQS.util.IO import tuplesToMatrix
from CQS.util.PauliOps import X, Y, I
from CQS.util.verification import KHK, exactU
from numpy.fft import fft, fftfreq
from numpy import kron
import numpy as np
import scipy as sci
from scipy.signal import find_peaks


#Green's Function Computation - Directly
#Ground State:

def GimpC(t, Parameters, psi0, sites):
    #Ut = exactU(Parameters.hamiltonian.HCoefs, Parameters.hamiltonian.HTuples, -1*t)
    Ut = KHK(Parameters.kCoefs, np.multiply(-1*t,Parameters.hCoefs), Parameters.cartan.k, Parameters.cartan.h)

    def Ggreater(Ut):
        """
        Computes G>(t) = -<GS| U†(t) c U(t) c†  |GS>
        Note, I've neglected the i term in the front
        """
        XI = kron(X,I)
        YI = kron(Y,I)
        for i in range(2, sites*2):
            XI = kron(XI, I)
            YI = kron(YI, I)
        step = (XI - 1j* YI) @ psi0
        step =  Ut @ step 
        step = (XI + 1j* YI) @ step
        step =  np.conjugate(Ut).T @ step
        step = np.conjugate(psi0).T @ step
        return  -1*step

    def Glesser(Ut):
        """
        Computes G<(t) =  <GS|  c† U†(t) c U(t) |GS>
        Note the i term is neglected
        """
        XI = kron(X,I)
        YI = kron(Y,I)
        for i in range(2, sites*2):
            XI = kron(XI, I)
            YI = kron(YI, I)

        return np.conjugate(psi0).T @ (XI - 1j* YI) @ np.conjugate(Ut).T @ (XI + 1j* YI) @ Ut @ psi0

    return 0.25j*(Ggreater(Ut) - Glesser(Ut))
    
def GimpE(t, Parameters, psi0):
    Ut = exactU(Parameters.hamiltonian.HCoefs, Parameters.hamiltonian.HTuples, -1*t)
    #Ut = KHK(Parameters.kCoefs, np.multiply(-1*t,Parameters.hCoefs), Parameters.cartan.k, Parameters.cartan.h)

    def Ggreater(Ut):
        """
        Computes G>(t) = -<GS| U†(t) c U(t) c†  |GS>
        Note, I've neglected the i term in the front
        """
        XI = kron(X,I)
        YI = kron(Y,I)
        for i in range(2, sites*2):
            XI = kron(XI, I)
            YI = kron(YI, I)
        step = (XI - 1j* YI) @ psi0
        step =  Ut @ step 
        step = (XI + 1j* YI) @ step
        step =  np.conjugate(Ut).T @ step
        step = np.conjugate(psi0).T @ step
        return  -1*step

    def Glesser(Ut):
        """
        Computes G<(t) =  <GS|  c† U†(t) c U(t) |GS>
        Note the i term is neglected
        """
        XI = kron(X,I)
        YI = kron(Y,I)
        for i in range(2, sites*2):
            XI = kron(XI, I)
            YI = kron(YI, I)

        return np.conjugate(psi0).T @ (XI - 1j* YI) @ np.conjugate(Ut).T @ (XI + 1j* YI) @ Ut @ psi0

    return 0.25j*(Ggreater(Ut) - Glesser(Ut))


#Fit iG_imp = a1cos(w1 t) + a2 cos(w2 t)
def func (t, a1, a2, w1, w2):
    cos1 = np.cos(w1 * t)
    cos2 = np.cos(w2 * t)
    val = a1 * cos1 + a2 * cos2
    return 2*val

def getFitFunc(params):
    a1 = params[0]
    a2 = params[1]
    w1 = params[2]
    w2 = params[3]
    def Gt(t):
        cos1 = np.cos(w1 * t)
        cos2 = np.cos(w2 * t)
        val = a1 * cos1 + a2 * cos2
        return 2*val
    return Gt

def GwFunc (params, delta=0.01):
    a1 = params[0]
    a2 = params[1]
    w1 = params[2]
    w2 = params[3]
    def Gw(w):
        return a1 * (1/(w + 1j*delta + w1) + 1/(w + 1j*delta - w1)) + a2 * (1/(w + 1j*delta + w2) + 1/(w + 1j*delta - w2))
    return Gw

def DeltaFunc(V, e1, mu):
    def Delta(w):
        return (V**2) / (w - (e1 - mu))
    return Delta

def SigmaFunc(Gw, Delta, mu):
    def Sigma(w):
        return (w + mu - Delta(w)) - 1/Gw(w)
    return Sigma

def QuasiZ(Sigma):
    region = np.linspace(-0.5, 0.5, 1000)
    minInRegion = np.argmin(Sigma(region))
    maxInRegion = np.argmax(Sigma(region))

    if minInRegion > maxInRegion:
        fitRegion = [x for x in np.linspace(maxInRegion, minInRegion, 10) if x!=0]
    else:
        fitRegion = [x for x in np.linspace(minInRegion, maxInRegion, 10) if x!=0]
    def linear(x, a):
        return a*x
    fit = sci.optimize.curve_fit(linear, fitRegion, Sigma(fitRegion))
    a = fit[0][0]
    return 1/(1 - a), a

def generateSpectralFunction(Gw):
    def spectralFunction(w):
        return -1/np.pi*(Gw(w).imag)
    return spectralFunction




