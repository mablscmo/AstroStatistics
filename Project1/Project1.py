"""
Created on 2019-11-18
@author: Martin Montelius
"""
import matplotlib.pyplot as plt 
import numpy as np
from scipy.spatial.distance import cdist
import numpy.random
from timeit import default_timer as timed

font = {'weight' : 'bold'}
plt.rcParams.update({'font.size': 35})

class PCorr:
    def __init__(self,file,n=9404,nth=100,thMax=1000,thMin=0,randit=10):
        '''Reads in the data. Sets a number of varibles:
        number of particles, number of bins, range of theta, number of iterations for averaging random data.'''

        #Datasets
        self.name=file 
        self.file=np.loadtxt(file)        
     
        #Variables
        self.n=n
        self.nth=nth
        self.thMax=thMax
        self.thMin=thMin
        self.dth=int(self.thMax/self.nth)
        self.theta=np.linspace(self.dth/2,self.thMax-self.dth/2,self.nth)
        self.randit=randit
        
    def distDD(self):
        '''Calculates distances between the points in the given dataset.
        np.histogram sorts the data into bins.
        Each distance is counted twice, theta_ij == theta_ji, so the count is divided by 2.
        self.n/2 is removed from the first bin to compensate for cdist counting the theta_ij i==j points.
        Returns the bins with the counts for the number of pairs in each interval.'''

        dist = cdist(self.file,self.file, 'euclidean')
        DD=np.histogram(dist,self.nth,(self.thMin,self.thMax))[0]/2
        DD[0]=DD[0]-self.n/2           
        print('distDD done')
        return(DD)  
        
    def distRR(self):
        '''Creates random, uniform data and a matrix with all the distances in the random dataset.
        Otherwise the same as self.distDD'''  
        
        RandData = np.random.uniform(0,1000,(self.n,2))
        
        distR = cdist(RandData,RandData, 'euclidean')
        RR=np.histogram(distR,self.nth,(self.thMin,self.thMax))[0]/2
        RR[0]=RR[0]-self.n/2
        return(RR)  
        
    def RRavg(self):
        '''Loops self.distRR to get average values for the RR(theta) bins.'''
        
        RR10=np.zeros(self.nth)
        print('distRR averaging')
        for i in range(self.randit):
            RR10 = RR10+self.distRR()
            print('{index}/{tot} done'.format(index = i+1, tot = self.randit))
        RR10 = RR10/self.randit
        return(RR10)
        
    def distDR(self):    
        '''Sets up a random dataset and calculates the distances from points in the given dataset to the random data.
        Since theta_ij =/= theta_ji and theta_ij i==j isn't a problem they do not need to be corrected for.'''
        
        RandData = np.random.uniform(0,1000,(self.n,2))
        DRdist=cdist(self.file,RandData, 'euclidean')
        DR=np.histogram(DRdist,self.nth,(self.thMin,self.thMax))[0]
        return(DR)
        
    def DRavg(self):
        '''Loops self.distDR to get average values for the DR(theta) bins.'''
        
        DR10=np.zeros(self.nth)
        print('distDR averaging')
        for i in range(self.randit):
            DR10 = DR10+self.distDR()
            print('{index}/{tot} done'.format(index = i+1, tot = self.randit))            
        DR10 = DR10/self.randit
        return(DR10)

    def w1(self,mult=False,log=False):
        '''The natural estimator for the 2-point correlation function, n==r is assumed throughout the code.
        The mult tag can be set if multiple datasets are to be compared in the same figure.'''

        DD=self.distDD()
        RR=self.RRavg()
        
        w1=DD/RR -1
        
        if mult==False:
            plt.figure()  
        
        if log==True:
            plt.semilogx(self.theta,w1,label=self.name,linewidth=10)
        else:
            plt.plot(self.theta,w1,label=self.name, linewidth=10)
        plt.xlabel(r'$\theta$',fontsize=40)
        plt.ylabel(r'$w_1(\theta)$',fontsize=40)
        plt.xlim([0,1000])
        plt.ylim([-0.1,0.1])
        plt.tight_layout()
        plt.title(r'$w_1$ for dataset {num}'.format(num=self.name[7]))
        if mult==False:        
            plt.savefig(self.name[7] + '_w1' + '.png',dpi=500)
        print('Dataset {num} done'.format(num = self.name[7]))
        
    def w3(self,mult=False,log=False):
        '''The Landy & Szalay estimator for the 2-point correlation function.'''
        
        DD=self.distDD()
        RR=self.RRavg()
        DR=self.DRavg()
        
        w3 = DD/RR - ((self.n-1)/self.n)*(DR/RR) + 1
        
        if mult==False:
            plt.figure()
        if log==True:
            plt.semilogx(self.theta,w3,label=self.name,linewidth=10)
        else:
            plt.plot(self.theta,w3,label=self.name, linewidth=10)
        plt.xlim([0,1000])
        plt.ylim([-0.1,0.1])
        plt.xlabel(r'$\theta$',fontsize=40)
        plt.ylabel(r'$w_3(\theta)$',fontsize=40)
        plt.tight_layout()
        plt.title(r'$w_3$ for dataset {num}'.format(num=self.name[7]))
        if mult==False:
            plt.savefig(self.name[7] + '_w3' + '.png',dpi=500)        
        print('Dataset {num} done'.format(num = self.name[7]))
        
    def w4w3(self):
        '''Extra: Compares the results from w3 and w4.'''
        DD=self.distDD()
        RR=self.RRavg()
        DR=self.DRavg()
        w3 = DD/RR - ((self.n-1)/self.n)*(DR/RR) + 1
        w4 = 4*self.n**2/((self.n-1)**2) * DD*RR/(DR**2) - 1
        return(w4-w3)
        
    def PolyCompare(self):
        '''Extra: Compares linear fits of w1, w3 and w4 to eachother.'''
        DD=self.distDD()
        RR=self.RRavg()
        DR=self.DRavg()
        
        w1=DD/RR -1
        w3 = DD/RR - ((self.n-1)/self.n)*(DR/RR) + 1
        w4 = 4*self.n**2/((self.n-1)**2) * DD*RR/(DR**2) - 1
        
        w1fit=np.polyfit(self.theta,w1,1)[0]
        w3fit=np.polyfit(self.theta,w3,1)[0]
        w4fit=np.polyfit(self.theta,w4,1)[0]
        
        return([w1fit,w3fit,abs(w3fit/w1fit)])
        
'''---------------- Code for getting the plots with all datasets in them -------------
   --------------------------------- change w_1 and w_3 ------------------------------'''

plt.figure(figsize=(14,18))
for i in range(6):
    PCorr('P1data0{num}.txt'.format(num=i+1),nth=100).w3(mult=True,log=True)
plt.xlim([0,1000])
plt.ylim([-0.1,0.1])
plt.legend(loc='lower left',frameon=False,labelspacing=0.2)
plt.title(r'$w_3$ plot for all datasets')
plt.savefig('w3' + '.png',dpi=500)


