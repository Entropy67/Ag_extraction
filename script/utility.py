## -----------------------------------
## -----------------------------------

"""
Utility functions for data visulization and calculation
    File name: utility.py
    Author: Hongda Jiang
    Date created: 10/19/2019
    Date last modified: 10/19/2019
    Python Version: 3.6
    Requirst package: Numpy, Matplotlib, Random
    
    Log:
    10192019: created the file
"""

__author__ = "Hongda Jiang"
__copyright__ = "Copyright 2018, UCLA PnA"
__license__ = "UCLA"
__email__ = "hongda@physics.ucla.edu"
__status__ = "Building"



import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats


### calculate fisher infor
def get_fisher_info_tau(dE, f, n=100, dx=0):
    kT = 4.012
    xa, ka =1.5,  1
    xb = xa + dx
    info = 0
    dem, nom1, nom2 = 0, 0, 0
    
    for i in range(1, n+1):
        if f*xb/(i*kT) > 500:
            continue
        kb = np.exp(-dE+f*xb/(i*kT))
        lmd = i*(ka*np.exp(f*xa/(i*kT)) + kb)
        dem += 1/(lmd**2)
        nom1 += i*kb/lmd**2
        nom2 += i*kb/lmd**3
    info = (nom1**2 + 0*2*nom2**2 / dem) / dem
    return info



def get_fisher_info_nlist(dE, f, n=100, dx=0):
    kT = 4.012
    
    info = 0
    
    for i in range(1, n+1):
        info += np.exp(dE - f*dx/(i*kT))/(1+np.exp(dE - f*dx/(i*kT)))**2
    return info



class Fig_gen:
    
    def __init__(self):
        self.labelsize=15
        self.titlesize=15
        self.figsize=(5,4)
        self.frame_width = 1.3

    def get_fig(self, x="x", y="y"):

        fig, ax = plt.subplots(figsize=self.figsize, dpi=100)
        plt.xlabel(x, fontsize=self.labelsize)
        plt.ylabel(y, fontsize=self.labelsize)
        self.set_ax_width(ax)
        return ax
    
    def set_ax_width(self, ax):
        frame_width = self.frame_width
        ax.spines['top'].set_linewidth(frame_width)
        ax.spines['right'].set_linewidth(frame_width)
        ax.spines['bottom'].set_linewidth(frame_width)
        ax.spines['left'].set_linewidth(frame_width)
        return
        
### estimator estimating Fisher information
def estimator(y1, y2, dE, bins, method="plug-in", a=0.5):
    """
    y1, y2: samped data from two parameters with difference dE
    bins: bins to get the histogram
    """
    
    
    info =None
    
    if method=="plug-in":
        ### get the histogram
        h1, b1 = np.histogram(y1, bins=bins, density=True)
        h2, b2 = np.histogram(y2, bins=bins, density=True)
        ### get the log likelihood
        l1 = np.asarray([np.log(h1i) if h1i>0 else 0 for h1i in h1 ])
        l2 = np.asarray([np.log(h2i) if h2i>0 else 0 for h2i in h2 ])

        ### estimate Fisher info
        info = np.sum(h1* ((l2-l1)/dE)**2)*(bins[1]-bins[0])
    elif method == "div":
        info = 4*Divergence(y1, y2, bins, a)/dE**2
        
    elif method == "FR_test":
        info = 4*Divergence_FR_test(y1, y2)/dE**2
        
    elif method == "cdf":
        info = 2*Divergence_cdf(y1, y2)/dE**2
        
    elif method == "kernel":
        P1 = stats.gaussian_kde(y1)
        P2 = stats.gaussian_kde(y2)
        ymin, ymax = min(min(y1), min(y2)), max(max(y1), max(y2))
        xlist= np.linspace(ymin, ymax, 51)
        info = 0
        for xi in xlist:
            info += P1(xi)[0]*((np.log(P1(xi)[0]) - np.log(P2(xi)[0]))/dE)**2*(xlist[1]-xlist[0])
            
    elif method == "gaussian":
        info = ((np.mean(y1) - np.mean(y2) )/(dE*np.std(y1)) )**2 
        
    else:
        print("No such method!")
    return info

### approximate the divergence of two distributions
def Divergence(y1, y2, bins, a=0.5):
    """
    calculate the f-divergence
    """
    ### get the histogram
    h1, b1 = np.histogram(y1, bins=bins, density=True)
    h2, b2 = np.histogram(y2, bins=bins, density=True)
    
    prefactor = 1/(4*a*(1-a))
    const = (2*a-1)**2
    
    nom = (a*h1 - (1-a)*h2)**2
    denom = a*h1 + (1-a)*h2
    
    div = prefactor*(np.sum(nom/denom)*(bins[1]-bins[0]) - const)
    return div

def Divergence_FR_test(y10, y20, test=False):
    
    y1=np.asarray(y10)
    y2 = np.asarray(y20)
    n1, n2 = y1.shape[0], y2.shape[0]
    y= np.concatenate([y1, y2])
    y.sort()
    y1set = set(y1)
    y2set = set(y2)
    if test:
        print("y1=", y1)
        print("y2=", y2)
        print("n1=", n1, ", n2=", n2)
        print("sorted y=", y)
    count = 0
    for i in range(y.shape[0]-1):
        if y[i] in y1set and y[i+1] in y2set:
            count +=1
        elif y[i] in y2set and y[i+1] in y1set:
            count +=1
        if y[i] in y2set and y[i] in y1set:
            raise Exception("error")
    if test: print("count=", count)
    return 1-count*(n1+n2)/(2*n1*n2)

def Divergence_cdf(y10, y20):
    ### get the histogram
    y1=np.asarray(y10)
    y2 = np.asarray(y20)
    
    n = len(y1)
    y1.sort()
    y2.sort()

    P = interpolate.interp1d(y1, np.arange(n)/n)
    Q = interpolate.interp1d(y2, np.arange(n)/n)
    
    dk = 0
    e= min(y1[1:]-y1[:-1])/4
    for i in range(0, n):
        if y1[i]- e<y1[0] or y1[i]>y2[-1] or y1[i]-e < y2[0]:
            continue
        dk += np.log((P(y1[i])-P(y1[i]-e))/ (Q(y1[i])-Q(y1[i]-e)))
    return dk/(n)-1


def get_most_prob(array):
    hist, bins = np.histogram(array, bins=30)
    index = np.argmax(hist)
    return (bins[index] +bins[index+1])/2


def convert_dict_to_string(my_dict):
    info = "{\n"
    for key, value in my_dict.items():
        info += "\t"
        info += key
        info += ": " + str(value)
        info += "\n"
    info += "\n}"
    return info

def gen_pi(n):
    if n==0:
        return '_3'
    elif n==1:
        return '.1'
    else:
        return str(mp.pi)[n+1]
    
def printProgress(n, N):
    percent = int(100.0*n/N)
    toPrint = "progress: "
    for i in range(percent//2):
        toPrint += '|'
    toPrint += "{:d}%    ".format(percent)
    print(toPrint, end='\r')
    return

fig_gen = Fig_gen()
