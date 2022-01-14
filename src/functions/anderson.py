"""
# Functionality:
Using OpenFermion and the Cartan Quantum Synethsizer, does stuff with the hubbard model

@author: Thomas Steckmann
"""
__docformat__ = 'google'

from openfermion import FermionOperator, jordan_wigner, normal_ordered
import warnings as warn
import itertools
import numpy as np
import openfermion
from CQS.methods import *
from CQS.util.PauliOps import I, X,Y,Z
from CQS.util.verification import Nident
from CQS.util.IO import tuplesToMatrix

stringToTuple  = {   
                        'X': 1,
                        'Y': 2,
                        'Z': 3
                    }
IZindex = [0,3]
#TODO: Add indexing to V
class AIM:


    def __init__(self, sites, U, V, mu, ei=None, S=0, filling='half'):
        """
        'up-up' = |↑<sub>impurity Site</sub>↑↑↑↓<sub>impurity Site</sub>↓↓↓>,


        Args:
            sites (int): The number of lattice points
            U (float): The coulumb repulsion coefficient
            V (float): tunneling coefficient
            mu (float): Chemical potential
            S (float, default = 0): Total Spin in the Z axis
            filling (String, int, default = 'half'): How many electrons?
        """
        self.sites = sites
        self.U = U
        self.V = V
        self.mu = mu
        if ei is None:
            self.ei = [1 for i in range(sites)]
        else:
            self.ei = ei
        self.S = 0
        self.Ne = filling

    def setHamiltonian(self,compress=False):
        """Uses Open Fermion to generate the Hubbard Model Hamiltonian. This lets us use CAR and automatically convert to Pauli Basis or do diagonalization

        Creation: 'i^'
        Annihilation: 'i'

        Returns:
            An OpenFermion CAR operator representing the Hamiltonian
        """
        if self.Ne =='half':
            self.Ne = self.sites
        #Creates a blank operator
        H = FermionOperator()  # zero operator
        for siteIndex in range(self.sites):
            for spinIndex in range(2):
                if siteIndex != 0: #Hopping 
                    #print('Adding Hopping Terms')
                    if abs(self.V) > 0.000001: #
                        H = H + self.V*FermionOperator('{0}^ {1}'.format( #Impurity to bath
                            spinIndex * self.sites, siteIndex + spinIndex * self.sites))

                        H = H + self.V*FermionOperator('{1}^ {0}'.format( #Bath to impurity
                            spinIndex * self.sites, siteIndex + spinIndex * self.sites))
                    #print(H)
                    #print(jordan_wigner(H))
                #print('Adding site energy for siteIndex = ' + str(siteIndex))
                #Chemical Potenial and site energy for site index
                H = H + (self.ei[siteIndex] - self.mu) * (FermionOperator('{latticePoint}^ {latticePoint}'.format(latticePoint = siteIndex + (self.sites) * spinIndex)))
                #print(H)
                #print(jordan_wigner(H))
        #print('Adding Coulomb interaction')
        #Coulomb Interacton between 0th up and 0th down
        H = H + self.U * (FermionOperator('{latticePointSpinDown}^ {latticePointSpinDown} {latticePointSpinUp}^ {latticePointSpinUp}'.format(latticePointSpinDown = 0, latticePointSpinUp = self.sites)))
        #print(H)
        #print(jordan_wigner(H))
        self.H = H
        self.HPauli = jordan_wigner(H)
        #remove small imaginary and real parts (OpenFermion default .compress(abs_tol=1e-08))
        if compress:
            self.HPauli.compress()

    def getH(self):
        return self.H
    
    def getHPauli(self):
        return self.HPauli
    
    def generateStateList(self):
        #Choose the basis states with the correct spins
        if self.S == 0:
            if int(self.Ne) % 2 != 0:
                warn('Invalid Filling and Total Spin: S = 0 requires even particles')
            else:
                sectorParticles = int(self.Ne / 2) #Number of particles in each of the ↑ and ↓ sectors
                if self.ordering == 'up-up':
                    sectorEmpty = self.sites - sectorParticles
                    halfPermutationsNotUnique = itertools.permutations(''.zfill(sectorEmpty) + ''.ljust(sectorParticles,'1'))
                    halfStatesTuple = [i for i in set(halfPermutationsNotUnique)] #The allowed computational basis elements for the sector
                    halfStatesString = [''.join(i) for i in halfStatesTuple]
                    fullStateList = []
                    for i in halfStatesString:
                        for j in halfStatesString:
                            fullStateList = fullStateList + [i + j]
                    self.allowedStates = fullStateList
        elif self.S == 0.5:
            if self.Ne % 2 == 0:
                warn('Invalid filling: Total spins must be odd for 0.5 spin')
            else:
                upSectorParticles = int((self.Ne + 1) / 2)
                downSectorParticles = int((self.Ne - 1) / 2)
                if self.ordering =='staggered':
                    upSectorEmpty = self.sites - upSectorParticles
                    upPermutationsNotUnique = itertools.permutations(''.zfill(upSectorEmpty) + ''.ljust(upSectorParticles,'1'))
                    downSectorEmpty = self.sites - downSectorParticles
                    downPermutationsNotUnique = itertools.permutations(''.zfill(downSectorEmpty) + ''.ljust(downSectorParticles,'1'))
                    upStatesTuple = [i for i in set(upPermutationsNotUnique)] #The allowed computational basis elements for the sector
                    downStatesTuple = [i for i in set(downPermutationsNotUnique)] #The allowed computational basis elements for the sector
                    upStatesString = [''.join(i) for i in upStatesTuple]
                    downStatesString = [''.join(i) for i in downStatesTuple]
                    fullStateList = []
                    for i in upStatesString:
                        for j in downStatesString:
                            fullStateList = fullStateList + [i + j]
                    self.allowedStates = fullStateList

    
    def opsToCartanFormat(self):
        """
        Converts the Operators to a list of (PauliStrings)
        """
        latticePoints = self.sites * 2
        opList = []
        coList = []
        for op in self.HPauli.terms.keys(): #Pulls the operator out of the QubitOperator format
            #tempTuple = (0,) * latticePoints
            coList.append(self.HPauli.terms[op])
            opIndexList = []
            opTypeDict = {}
            tempTuple = ()
            for (opIndex, opType) in op:
                opIndexList.append(opIndex)
                opTypeDict[opIndex] = opType
            for index in range(latticePoints):
                if index in opIndexList:
                    tempTuple += (stringToTuple[opTypeDict[index]],)
                else:
                    tempTuple += (0,)
            opList.append(tempTuple)
        
        return (coList, opList)
            
    def getAllowedStates(self):
        """
        Returns a list of allowed states
        """
        return self.allowedStates

