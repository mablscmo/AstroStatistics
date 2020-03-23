# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:45:17 2019

@author: monte
"""
import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy import integrate
import scipy.stats as stats
from scipy import optimize
from scipy.stats import genextreme as gev
from scipy.stats import percentileofscore as invperc
from scipy.stats import scoreatpercentile as scoreperc
import numpy.random
from timeit import default_timer as timed
from astropy.timeseries import LombScargle


font = {'weight' : 'bold'}
plt.rcParams.update({'font.size': 30})

#Load in the stars
P1 = 'hd000142.txt'
P2 = 'hd027442.txt'
P3 = 'hd102117.txt'


class LSP:
    def __init__(self, data, n0=5,method=4,FAP=None):
        #Checks datatype to enable the class to run from both the datafiles and from the 
        #bootstrap data for the FAP
        if type(data) == str:
            self.data = np.loadtxt(data)
        else:
            self.data = data
        #Loads data and calculates uncertainties
        self.t = self.data[:,0]
        self.rv = self.data[:,1]
        self.Munc = self.data[:,2]**2
        self.unc = 1/np.sum((1/self.data[:,2])**2)                   
        self.time = self.t - np.amin(self.t)
        self.RV = self.rv - np.mean(self.rv)
        self.sig2 = np.var(self.RV)
        
        #Sets limits for the frequencies
        self.N = len(self.RV)
        self.n0 = n0
        self.fmin = 1/np.amax(self.time) 
        self.fmax = np.pi*np.median(1/np.diff(self.t))
        self.Neval = int(self.n0 * np.amax(self.time) * self.fmax)
        self.flist = np.linspace(self.fmin,self.fmax,self.Neval)
        
        #For the FAP, selects a random frequency with an interval of K frequencies around it
        if FAP != None:
            frand = np.random.uniform(FAP/2,self.Neval-FAP/2)
            self.flist = self.flist[int(frand-FAP/2):int(frand+FAP/2)]
            self.Neval = FAP
            
        #Switch between different normalisations for the LS periodogram
        self.method = method
        
        #Sets the names of the stars for the plots
        if data == P1:
            self.name = 'hd000142'
            self.cheat = 349.7
        elif data == P2:
            self.name = 'hd027442'
            self.cheat = 415.2
        elif data == P3:
            self.name = 'hd102117'
            self.cheat = 20.67
            
    def test(self):
        #Not yet implemented z-level method
        P = self.search()
        Ptop = np.amax(P)
        
        Psing = 1-np.exp(-Ptop)
        W = self.fmax*np.sqrt(4*np.pi*np.var(self.time))
        tau = W * np.sqrt(Ptop) * (1-Ptop)**((self.N-4)/2)
        
        FAP = 1 - Psing*np.exp(-tau)
        
        return(FAP)
    
    def APSearch(self,pq=None):
        #Implementation of astropys LombScargle, used only for testing purposes
        AstroLS = LombScargle(self.time,self.RV,self.Munc)        
        Search = AstroLS.power(self.flist)
        return(Search)
    
    def APFAP(self,P):
        #FAP from LombScargle, not entierly accurate as I'd want because my LombScargle 
        #returns worse estimations than theirs
        
        AstroLS = LombScargle(self.time,self.RV,self.Munc)
        FAP = AstroLS.false_alarm_level(P)
        return(FAP)
    
    def LombS(self, f):
        tau = np.arctan(np.sum(np.sin(4*np.pi*f*self.time))/np.sum(np.cos(4*np.pi*f*self.time)))/(4*np.pi*f)
        PLS = (((np.sum(self.RV*np.cos(2*np.pi*f*(self.time-tau))))**2/(np.sum(np.cos(2*np.pi*f*(self.time-tau))**2))) + ((np.sum(self.RV*np.sin(2*np.pi*f*(self.time-tau))))**2/(np.sum(np.sin(2*np.pi*f*(self.time-tau))**2))))/2
        return(PLS) 

    def LombS2(self, f):
        tau = np.arctan(np.sum(np.sin(4*np.pi*f*self.time))/np.sum(np.cos(4*np.pi*f*self.time)))/(4*np.pi*f)
        PLS = (((np.sum(self.RV*np.cos(2*np.pi*f*(self.time-tau))))**2/(np.sum(np.cos(2*np.pi*f*(self.time-tau))**2))) + ((np.sum(self.RV*np.sin(2*np.pi*f*(self.time-tau))))**2/(np.sum(np.sin(2*np.pi*f*(self.time-tau))**2))))/(2*self.sig2)
        return(PLS) 
        
    def LombS3(self, f):
        tau = np.arctan(np.sum(np.sin(4*np.pi*f*self.time))/np.sum(np.cos(4*np.pi*f*self.time)))/(4*np.pi*f)
        PLS = (((np.sum(self.RV*np.cos(2*np.pi*f*(self.time-tau))))**2/(np.sum(np.cos(2*np.pi*f*(self.time-tau))**2))) + ((np.sum(self.RV*np.sin(2*np.pi*f*(self.time-tau))))**2/(np.sum(np.sin(2*np.pi*f*(self.time-tau))**2))))/(2*self.unc)
        return(PLS) 
        
    def LombS4(self, f):
        #Currently used normalisation, of the shape P = 1-X_1/X_0, 
        #should be changed as it produces inferior results compared to 
        #Astropy's implementation.
        tau = np.arctan(np.sum(np.sin(4*np.pi*f*self.time))/np.sum(np.cos(4*np.pi*f*self.time)))/(4*np.pi*f)
        w = f*2*np.pi
        c = np.cos(w*(self.time-tau))
        s = np.sin(w*(self.time-tau))
        A = np.sum(self.RV*c/self.Munc)/np.sum(c**2/self.Munc)
        B = np.sum(self.RV*s/self.Munc)/np.sum(s**2/self.Munc)
        H0 = np.sum(self.RV**2/self.Munc)
        H1 = np.sum((A*c+B*s-self.RV)**2/self.Munc)
        PLS = 1-H1/H0
        return(PLS)
        
    
    def search(self):
        #Takes in data and calculates the Lomb-Scargle power
        Plist = []
        if self.method == 'A':
            Plist = self.APSearch()
            
        elif self.method == 2:
            for i in range(self.Neval):
                Plist.append(self.LombS2(self.flist[i]))        
        elif self.method == 3:
            for i in range(self.Neval):
                Plist.append(self.LombS3(self.flist[i]))
        elif self.method == 4:
            for i in range(self.Neval):
                Plist.append(self.LombS4(self.flist[i]))            
        else:
            for i in range(self.Neval):
                Plist.append(self.LombS(self.flist[i]))            
        return(Plist)
    
    def FAPz(self, z, Neff):
        #Not completed FAP implementation
        Psing = 1 - np.exp(-z)
        FAP = 1 - Psing**Neff
        return(FAP)
    
    
    def plot(self,FAPname,Nlevels,cheat=True):
        '''Plots the periodogram with significance levels provided by 
        the bootstrap. Number of displayed significance levels adjusted
        with Nlevels. cheat shows the FAP calculated by astrop, as well
        as a marker showing a tabulated value for the frequency of the planets
        oscillation. Not finished code for calcualting the FAP based on
        the z-levels is still present.'''
        P = self.search()
        Ptop = np.amax(P)
        ftop = self.flist[np.where(P == Ptop)[0][0]]
        Levels = np.array([50,90,95,99,99.9])[:Nlevels]
        # Neff = 0
        # for i in range(self.Neval-2):
        #     if (P[i]<P[i+1]) & (P[i+1]>P[i+2]):
        #         Neff = Neff + 1
        # # print(Neff)
        # Neff = self.fmax * 1/(self.flist[1]-self.flist[0])
        # # print(Neff)
        
        FAPfile = np.loadtxt(FAPname + 'FAPNormTest.txt')
        #PLevels = scoreperc(FAPfile,Levels)
        
        fit = gev.fit(FAPfile)
        PLevels = gev.ppf(Levels/100,*fit)
        
        
        plt.figure(figsize=(20,14))
        plt.hlines(PLevels,self.fmin,self.fmax,'g')
        plt.plot(self.flist,P)
        plt.text(self.fmax-(self.fmax-self.fmin)/2.25,plt.ylim()[1],'False alarm probability')
        plt.ylim(0,Ptop+0.1)
        for i in range(Nlevels):
            plt.text(self.fmax-(self.fmax-self.fmin)/3,PLevels[i]+0.003,str(np.round(1-Levels[i]/100,3)))        
        plt.plot(ftop,Ptop,'r',marker='o',linestyle='none',markerfacecolor='none',markersize=35)
        if cheat == True:
            plt.vlines(1/self.cheat,0,Ptop,'g')
            
            CheatLevels = self.APFAP(1-Levels/100)
            plt.hlines(CheatLevels,self.fmin,self.fmax,'r')
        plt.xlabel('Frequency [1/day]')
        plt.ylabel('Lomb-Scargle Power')
        plt.title('Lomb - Scargle Periodogram for {planet}'.format(planet = self.name))
        print('Highest probability of period = {p} days'.format(p=round(1/ftop,3)))
    
    def foldplot(self,guess=[30,0.03,8,5],shift=10):
        '''Takes data and folds it over the most likely period found by search.
        Uses exterior function sinefit to fit a sinecurve to the folded data,
        needs to be rewritten as it takes too many arguments, the b parameter
        should be replaced by a fixed number f_top and the d parameter should 
        probably not be necessary since mean(RV) should be 0.
        Also needs to have guesses for different stars loaded.'''
        P = self.search()
        period = 1/self.flist[np.where(P == np.amax(P))[0][0]]
        t_fold = self.time - np.floor(self.time /period)*period
        # time = self.time
        # while np.amax(time)>period:
        #     for i in range(len(time)):
        #         if time[i]>period:
        #             time[i] = time[i]-period
        
        par, par_covariance = curve_fit(sinefit, t_fold, self.RV, p0=guess )
        print (*par)
        foldtime = np.linspace(np.min(t_fold), np.max(t_fold), 1000 )
        RVmax = np.amax(np.abs(self.RV))
        plt.figure(figsize=(14,14))
        plt.plot(t_fold, self.RV,'.',markersize=25,label='Data')
        plt.plot(foldtime, sinefit(np.array(foldtime), *par), color='r', label=r'Sine fit = ${a}\cdot \sin({b}\cdot t+{c})+({d})$'.format(a=round(par[0]),b=round(par[1],3),c=round(par[2]),d=round(par[3])))
        plt.xlabel('Time [days]')
        plt.ylabel('Radial velocity [m/s]')
        plt.title('Folded data with fitted sine curve for {name}'.format(name = self.name))
        plt.ylim((-(RVmax+shift),RVmax+shift))
        plt.legend(loc='best',fancybox=True)

def sinefit(x, a, b, c, d):
    '''Gives a sinewave'''
    return a*np.sin(b*x+c)+d

def FAP(Data,n0,R,L,K):
    '''Takes a dataset, seperates the columns, randomly chooses velocities
    (can choose same point more than once) and creates a new random dataset
    which should replicate the Gaussian noise. The LSP class recognices
    bootstrapped data and selects a random frequency interval with the correct 
    parameters. The highest power for each bootstrap is saved.'''
    data = np.loadtxt(Data)
    t = np.vstack(data[:,0])
    rv = data[:,1]
    unc = np.vstack(data[:,2])
    RV = rv - np.mean(rv)   

    N = len(data)

    Plist=[]
    for i in range(R):
        Rand = np.random.randint(0,N,(1,N))
        NewRV = np.vstack(RV[Rand][0])
        New = np.append(t,NewRV,axis=1)
        BootData = np.append(New,unc,axis=1)
        
        Ptemp = []
        for j in range(L):
            PBoot = np.amax(LSP(BootData,n0,4,FAP=K).search())            
            Ptemp.append(PBoot)
        Plist.append(np.amax(Ptemp))
        print('{nummer}% done'.format(nummer=round(100*(1+i)/R,1)))
    return(Plist)
    
''' Creates the datafiles for the FAP from bootstrapping. Everything is
saved to files because of an earlier implementation which took up to an 
hour to run for one star. The specific L & K parameters are chosen based
on the amount of frequencies and the number of datapoints.'''   
#P1 FAP(P1,5,1000,6,11)
    
tic = timed()

TestSet = FAP(P1,5,1000,6,11)
np.savetxt('P1FAPNormTest.txt',TestSet)

print(timed()-tic)

#P2 FAP(P2,5,1000,6,15)
    
tic = timed()

TestSet = FAP(P2,5,1000,6,15)
np.savetxt('P2FAPNormTest.txt',TestSet)

print(timed()-tic)

#P3 FAP(P3,5,1000,6,20)
    
tic = timed()

TestSet = FAP(P3,5,1000,12,20)
np.savetxt('P3FAPNormTest.txt',TestSet)

print(timed()-tic)

'''Plotting the periodograms with the significance levels. Change cheat to 
True to see astropy's FAP, change method to 'A' to see astropy's periodogram.
n0 can be changed if resolution appears to be too low, but should be OK.
Nlevels gives the number of significance levels shown, up to 5.'''
LSP(P1,n0=5,method=4).plot('P1',Nlevels=5,cheat=False)
LSP(P2,n0=5,method=4).plot('P2',Nlevels=4,cheat=False)
LSP(P3,n0=5,method=4).plot('P3',Nlevels=5,cheat=False)

'''Folded data plots, change shift if the there is a problem with the window. '''
LSP(P1,5,4).foldplot(shift=10)
LSP(P2,5,4).foldplot(shift=10)
LSP(P3,5,4).foldplot(shift=10)

'''Example of Generalised extreme-value distribution  '''
fit = gev.fit(np.loadtxt('P3FAPNormTest.txt'))
plt.figure(figsize=(12,12))
plt.plot(np.linspace(0,1,200), gev.pdf(np.linspace(0,1,200), *fit))
plt.title('GEV for HD 102117')
plt.xlabel('Periodogram power')
plt.ylabel('Frequency in bootstrap data')
plt.tight_layout()


