# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:53:25 2019

@author: monte
"""

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import beta, chi2
import numpy.random
from scipy.special import gamma
import itertools


font = {'weight' : 'bold'}
plt.rcParams.update({'font.size': 35})

class bfit:
    def __init__(self,P=(0,10),inc=500,AB=5,count=False,file='data.txt'):
        '''Class treats one subset of the data on exoplanet eccentricities and periods from exoplanets.org.
        The purpose is to calculate the log-likelihood for different parameters of the beta distribution,
        and to plot it in a useful way.'''
        
        #Sets grid that defines alpha & Beta parameterspace
        self.a=np.linspace(0.0001,AB,inc)
        self.b=np.linspace(0.0001,AB,inc)
        
        #Period-ranges and stepsize when probing parameterspace.
        self.Pmin=P[0]
        self.Pmax=P[1]
        self.inc=inc
        
        #Load and filter data
        self.data=np.genfromtxt(file,delimiter=',')
        self.data[self.data==0]=0.001
        self.e=self.data[(self.data[:,0]>self.Pmin)&(self.data[:,0]<self.Pmax)][:,1]
        self.P=self.data[(self.data[:,0]>self.Pmin)&(self.data[:,0]<self.Pmax)][:,0]
        
        #Check if enough/too many planets are selected, set as variable to be included in plots
        self.N=len(self.e)
        if count == True:
            print('Chosen sample contains {elen} exoplanets.'.format(elen=self.N))

    def logL(self):
        '''Calculates a matrix with the log-likelihoods of different parameter combinations.
        The use of a for-loop is kind of slow, but I couldn't get the np.arrays working like I wanted 
        and this wasn't too bad, time wise.'''
        
        results=np.zeros((self.inc,self.inc))
        for i, j in itertools.product(range(self.inc),range(self.inc)):
            results[j,i]=sum(np.log(self.e**(self.a[i]-1) * (1-self.e)**(self.b[j]-1) * gamma(self.a[i]+self.b[j]) / (gamma(self.a[i]) * gamma(self.b[j]))))        
        return(results)

    def test(self,a,b,length=396):
        '''Creates a log-likelihood matrix for values drawn from a beta distribution,
        its purpose is to test later functions, to see if they give reasonable reults.'''
        r = beta.rvs(a, b, size=length)
        results=np.zeros((self.inc,self.inc))
        for i, j in itertools.product(range(self.inc),range(self.inc)):
            results[j,i]=sum(np.log(r**(self.a[i]-1) * (1-r)**(self.b[j]-1) * gamma(self.a[i]+self.b[j]) / (gamma(self.a[i]) * gamma(self.b[j]))))        
        return(results)        
        
    def cont(self,test=False,a=2,b=3,conf=0.9):
        '''Creates a contour plot with discrete colours to show the different log-likelihoods. 
        The 90% confidence region and the most likely estimate are marked.
        The alpha and beta values for the MLE are also printed.'''
        
        if test==True:
            l=self.test(a,b)
        else:
            l=self.logL()
        index=np.unravel_index(l.argmax(), l.shape)
        amax, bmax = self.a[index[1]], self.b[index[0]]
        lmax=np.amax(l)
        
        plt.figure(figsize=(14,14))
        plt.contourf(self.a,self.b,l,cmap='inferno',levels=10)
        cbar = plt.colorbar()
        cbar.set_label(r'$l(\alpha, \beta | e)$')
        plt.contour(self.a,self.b,2*(lmax-l)-chi2.ppf(conf,2),levels=(1,100000),cmap='PiYG')
        plt.plot(amax,bmax,'kx')
        plt.xlabel(r'$\alpha$')
        plt.xticks(ticks=[0,1,2,3,4,5])
        plt.ylabel(r'$\beta$')
        plt.yticks(ticks=[0,1,2,3,4,5])
        plt.minorticks_on() 
        plt.title(r'$P_{min}, P_{max}$'+'={mini},{maxi}  '.format(mini=self.Pmin,maxi=self.Pmax) + r' $N_{planets}$' + '={N}'.format(N=self.N),fontsize=33)

        
        
        print(r'A={a} B={b}'.format(a=round(amax,3),b=round(bmax,3)))
        return(2*(lmax-l)-chi2.ppf(conf,2))
    def fitplot(self,test=False,a=2,b=3):
        '''Displays a histogram of the eccentricity values alongside a visualisation of the most likely beta functions pdf.'''
        
        if test==True:
            l=self.test(a,b)
        else:
            l=self.logL()
        index=np.unravel_index(l.argmax(), l.shape)
        amax, bmax = self.a[index[1]], self.b[index[0]] 
        
        x = np.linspace(beta.ppf(0.01, amax, bmax), beta.ppf(0.99, amax, bmax), 10)
        plt.figure(figsize=(18,14))
        plt.plot(x, beta.pdf(x, amax, bmax),'r-', lw=5, alpha=0.7, label='beta pdf A={am} B={bm}'.format(am=round(amax,2),bm=round(bmax,2)))
        plt.hist(self.e,bins=10,density=True,histtype='stepfilled', label='eccentricity histogram') 
        plt.ylim([0,10])
        plt.xlim([0,1])
        plt.xlabel('Eccentricity')
        plt.ylabel('Relative frequency')
        plt.legend()
        plt.title(r'$P_{min}, P_{max}$'+'={mini},{maxi}  '.format(mini=self.Pmin,maxi=self.Pmax) + r' $N_{planets}$' + '={N}'.format(N=self.N),fontsize=33)
        
        print(r'MLE with A={a} B={b}'.format(a=round(amax,3),b=round(bmax,3)))
        
    def periodplot(self):
        '''Check distribution of eccentricities. Didn't include in report because it felt like 
        there was enough figures. Used it to look for interesting regions..'''
        plt.figure()
        plt.plot(self.P,self.e,'.',markersize=5)
        plt.xlabel('Period [days]')
        plt.ylabel('Eccentricity')
        plt.title('Distribution of eccentricities by period')
