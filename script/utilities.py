
'''

useful functions


'''

import numpy as np
import json
from mpmath import mp
import os
from tabulate import tabulate
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.colors as cm

def load_prm(filename):
    ### load the json
    with open(filename, 'r') as fp:
        prm = json.load(fp)
    return prm


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
    for i in range(percent//5):
        toPrint += '|'
    toPrint += "{:d}%    ".format(percent)
    print(toPrint, end='\r')
    return


def dump(dataset, filename, unique=True, mod='w', multiple_dict=False, sep=False):
    """
    save data to file
    """

    if unique:
        ### create uniq filename
        v = 0
        while os.path.exists(filename+'.txt'):
            print("file exists! ")
            filename += gen_pi(v)
            v += 1
            
    if not multiple_dict: ### only one dict
        with open(filename+'.txt', mod) as fp:
            fp.write(json.dumps(dataset, indent=4))
            if sep:
                fp.write("\n\n")
                fp.write("*"*20)
                fp.write("\n\n")
    else:
        data_saved = []
        if mod=='a' and os.path.exists(filename+'.txt'):
            with open(filename +'.txt', 'r') as fp:
                data_saved = json.load(fp)
        data_saved.append(dataset)
        with open(filename+'.txt', mod) as fp:
            fp.write(json.dumps(data_saved, indent=4))
    return filename

def convert_dict_to_table(prm_dict, ncols=3):
    ret = []
    row = []
    c = 0
    for k, v in prm_dict.items():
        c += 1
        row.append(k)
        if isinstance(v, str) and len(v) > 10:
            row.append(colored(v[:10]+"...", "magenta"))
        else:
            row.append(colored(v, "magenta"))
        if c % (ncols) == 0:
            ret.append(row.copy())
            row = []
    if c % (ncols) != 0:
        while c % (ncols) != 0:
            row.append("")
            row.append("")
            c += 1
        ret.append(row.copy())
    return ret

def print_dict(prm_dict, ncols=3):
    print(tabulate(convert_dict_to_table(prm_dict, ncols), 
               headers=["name", "value"]*ncols, 
               tablefmt="fancy_grid",
              colalign=("right", "left")*ncols
           ))
    return 


def get_ax(xlabel="", ylabel="", xlim=None, ylim=None, ncols=1, figsize=None):
    if ncols==1:
        fig, ax = plt.subplots(figsize=(3, 2.5) if figsize is None else figsize, dpi=150)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if xlim is not None:
            ax.set(xlim=xlim)
        if ylim is not None:
            ax.set(ylim=ylim)
        return fig, ax
    else:
        fig, axes = plt.subplots(figsize=(3*ncols, 2.7) if figsize is None else figsize, ncols=ncols, dpi=150)
        plt.subplots_adjust(wspace=0.2)
        if isinstance(xlabel, list):
            for xl, ax in zip(xlabel,  axes):
                ax.set(xlabel=xl)
        else:
            for ax in axes:
                ax.set(xlabel=xlabel)
        if isinstance(ylabel, list):
            for yl, ax in zip(ylabel,  axes):
                ax.set(ylabel=yl)
        else:
            for ax in axes:
                ax.set(ylabel=ylabel)
        if isinstance(xlim, list):
            for xl, ax in zip(xlim,axes):
                ax.set(xlim=xl)
        elif xlim is not None:
            for ax in axes:
                ax.set(xlim=xlim)
        if isinstance(ylim, list):
            for yl, ax in zip(ylim,axes):
                ax.set(ylim=yl)
        elif ylim is not None:
            for ax in axes:
                ax.set(ylim=ylim)
                
        return fig, axes

################ plot figures ####################
def hash_color(c, cmin=-0.5, cmax=2, n=20, cmap = plt.cm.coolwarm):
    colors = [cm.to_hex(cmap(i/n)) for i in range(n)]
    if np.isnan(c):
        return "white"
    ci = min(c, cmax)
    ci = max(c, cmin)
    index = int(  n * (c-cmin) / (cmax - cmin) )
    index = min(index, n-1)
    index = max(index, 0)
    return colors[index]

