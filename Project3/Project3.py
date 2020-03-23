# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:59:46 2019

@author: monte
"""

import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import minimize
from scipy import integrate
import numpy.random
font = {'weight' : 'bold'}
plt.rcParams.update({'font.size': 25})

#Load data, first 3 rows were removed manually
dataset = 6
data = np.loadtxt('P3data0{nr}.txt'.format(nr = dataset))
#Set limits for where the peak should be measured. Whole dataset displayed by adjusting limits.
if dataset == 4:
    imin = 0
    imax = 1000
elif dataset == 6:
    imin = 0
    imax = 24630
    
#Slice datasets according to the limits
flux = data[:,0][imin:imax]
lam = 10**(data[:,1][imin:imax])
ivar = data[:,2][imin:imax]
sig = np.sqrt(1/ivar)

#Set initial guesses
if dataset == 4:
    t1, t2, t3, t4, t5, lamref = 42, -0.0178, 65, 4254, 57, 4000
elif dataset == 6:
    t1, t2, t3, t4, t5, lamref = 9.28, -1.800e-03, 14.8, 6172, 60, 6000
    
t0=[t1,t2,t3,t4,t5]

#Lineprofiles
def Gauss(x):
    return(np.exp(-x**2/2)/np.sqrt(2*np.pi))
def Lorenz(x):
    return(1/(np.pi*(1+x**2)))


#Gives continuum + emission line as linear funcion + line profile, eq 2 in report
def f(t0):
    t1, t2, t3, t4, t5 = t0
    if func == 'G':
        return(t1+t2*(lam-lamref)+t3*Gauss((lam-t4)/t5))
    if func == 'L':
        return(t1+t2*(lam-lamref)+t3*Lorenz((lam-t4)/t5))
        
#Calculates chi squared for some parameters t0 provided by either the initial guess or minimization
def model(t0):
    chi2 = np.sum((flux-f(t0))**2 *ivar)
    return(chi2)

#Calculates the redshift based on some calculated \lambda_obs, eq 1 in report
def z(lobs):
    return((lobs-2800.3)/2800.3)


#Minimizes chi-square and calculate residuals for Gaussian and Lorentzian fit.
func = 'G'
res1 = minimize(model,t0)
residual1 = flux-f(res1.x)
flux1 = f(res1.x)

func = 'L'
res2 = minimize(model,t0)
residual2 = flux-f(res2.x)
flux2 = f(res2.x)


#Plots the data and the two fits
plt.figure(figsize=(20,10))
plt.plot(lam,flux,label='Data')
#plt.plot(lam,flux1,'r-',linewidth=3,label='Gaussian')
#plt.plot(lam,flux2,'k-',linewidth=3,label='Lorentzian')
plt.xlabel('Wavelength [Ångström]')
plt.ylabel(r'Flux [10$^{-17}$erg/cm$^2$/s/Å]')
plt.title('Line profile fits to the data')
#plt.legend(fancybox=True)
plt.minorticks_on()
plt.grid() 
plt.box(True) 


##Plots residuals for both profiles
#plt.figure(figsize=(15,9))
#plt.plot(lam,np.zeros(len(lam)),'g--')
#plt.plot(lam,residual1,'r.',label='Gaussian')
#plt.xlabel('Wavelength [Ångström]')
#plt.ylabel('Residuals')
#plt.box(True) 
#plt.plot(lam,residual2,'k.',label='Lorentzian')
#plt.title('Residuals for the Lorentzian and Gaussian fits')
#plt.ylim([-13,13])
#plt.minorticks_on()
#plt.grid() 
#plt.legend(fancybox=True)


#Calcualtes reduced chi-squared to compare the two fits
rchi2G = model(res1.x)/(len(lam)-5)
rchi2L = model(res2.x)/(len(lam)-5)


#Uncertainty Calculations
#Uncertainty alternative function, takes synthetic dataset   
def synthchi2(t0,*args):
    return(np.sum((args-f(t0))**2 *ivar))
    
#Applying alternate function to get values for z from the synthetic datasets
zG = []
zL = []
##for i in range(200):
#    syntG = flux1+sig*np.random.normal(0,1,len(lam))
#    syntL = flux2+sig*np.random.normal(0,1,len(lam)) 
#    func = 'G'
#    tG = minimize(synthchi2,res1.x,args=syntG,tol=1e-10,method='Nelder-Mead').x
#    func = 'L'
#    tL = minimize(synthchi2,res2.x,args=syntL,tol=1e-10,method='Nelder-Mead').x
#    zG.append(z(tG[3]))
#    zL.append(z(tL[3]))    

##Standard deviation in the synthetic z
#stdG = np.std(zG)
#stdL = np.std(zL)

##Looking at how the synthetic dataset looks
#plt.figure(figsize=(15,7))
#plt.plot(lam,flux,'k.',label='Real data')
#plt.plot(lam,syntG,'r.',label='Synthetic Gaussian data')
#plt.plot(lam,syntL,'b.',label='Synthetic Lorentzian data')
#plt.legend(fancybox=True)
#plt.xlabel('Wavelength [Ångströms]')
#plt.ylabel(r'Flux [10$^{-17}$erg/cm$^2$/Å]')
#plt.title('Comparison of real and synthetic datasets')
#plt.tight_layout()

