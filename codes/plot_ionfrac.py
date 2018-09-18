#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import MultipleLocator

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

t_list = ['3.7', '5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['4', '6', '9', '12', '16', '19']
color = ['c', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

Z2el = {6: 'C', 8: 'O', 14: 'Si', 26: 'Fe'}
ion2symbol = {0: 'I', 1: 'II', 2: 'III'}
par2label = {None: '', 'CII1012': '6580', 'CI1119': '10693'}

fs = 20

def retrieve_number_dens(_syn, _Z):
    numdens = pd.read_hdf(_syn + '.hdf', '/simulation/plasma/number_density')
    iondens = pd.read_hdf(_syn + '.hdf', '/simulation/plasma/ion_number_density')
    v_inner = pd.read_hdf(_syn + '.hdf', '/simulation/model/v_inner').values / 1.e5
    elecdens = pd.read_hdf(_syn + '.hdf', '/simulation/plasma/electron_densities')
    
    #print zip(elecdens.values,v_inner)
    
    n_shells = len(v_inner)
    n_el = [numdens.loc[_Z][j] for j in range(n_shells)]
    f_ionI = [iondens.loc[_Z,0][j] / n_el[j] for j in range(n_shells)]
    f_ionII = [iondens.loc[_Z,1][j] / n_el[j] for j in range(n_shells)]
    f_ionIII = [iondens.loc[_Z,2][j] / n_el[j] for j in range(n_shells)]
    return v_inner, np.array(f_ionI), np.array(f_ionII), np.array(f_ionIII)
    
class Plot_Ionfractions(object):
    '''
    Description:
    ------------
    This routine plots the fraction of neutral and singly and doubly ionized
    species of a given element. These quantities are retrieved from the TARDIS
    simulations of a default (best) model -- see paper for details.
    
    Parameters:
    -----------
    syn_list : ~list
        List containing the full path to the .hdf5 files of the simulations.
    Z : ~int
        Atomic number of the element to be ploted.    
      
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_ionfrac-ELEMENT.pdf' 
    '''
    
    def __init__(self, syn_list, Z, show_fig=True, save_fig=False):
        
        self.syn_list = syn_list
        self.Z = Z
        self.show_fig = show_fig
        self.save_fig = save_fig      
        
        self.F = {}
        self.D = {}
        self.fig = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111) 

        self.make_plot()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""

        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = (r'$\frac{n(\mathrm{' + Z2el[self.Z] + '_{X}})}'\
                   + '{n(\mathrm{' + Z2el[self.Z] + '})}$')
        self.ax.set_yscale('log')
        
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xlim(7500.,24000.)
        self.ax.set_ylim(1.e-12,1.5)
        
        #Adding log ticks by hand. This is necessary when the log scale
        #spans more than 10 orders of magnitude, as is the case here.
        locmaj = mpl.ticker.LogLocator(base=10, numticks=13) 
        self.ax.yaxis.set_major_locator(locmaj)
        
        locmin = mpl.ticker.LogLocator(
          base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), numticks=13)
        self.ax.yaxis.set_minor_locator(locmin)
        self.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(5000.))

    def plot_quantities(self):
                
        for i, syn in enumerate((self.syn_list)):
            v, fI, fII, fIII = retrieve_number_dens(syn, self.Z)
            self.ax.plot(v, fI, ls='-', lw=3., marker='None', color=color[i])
            self.ax.plot(v, fII, ls='--', lw=3., marker='None', color=color[i])
            self.ax.plot(v, fIII, ls=':', lw=3., marker='None', color=color[i])

    def make_legend(self):
        self.ax.plot([np.nan], [np.nan], ls='-', lw=3., color='k',
                     label=r'$\mathrm{X=I}$')
        self.ax.plot([np.nan], [np.nan], ls='--', lw=3., color='k',
                     label=r'$\mathrm{X=II}$')
        self.ax.plot([np.nan], [np.nan], ls=':', lw=3., color='k',
                     label=r'$\mathrm{X=III}$')
        for i in range(len(self.syn_list)):
            self.ax.plot([np.nan], [np.nan], ls='-', lw=10., color=color[i],
                         label= r'$\rm{t_{exp}\ =\ ' + t_list[i] + '\ \mathrm{d}}$')
        
        self.ax.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1,
                       labelspacing=0.1, loc='best')        
            
    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/FIGURES/Fig_ionfrac-' + Z2el[self.Z] + '.pdf'
            plt.savefig(fpath , format='pdf')
        if self.show_fig:
            plt.show()
        plt.close()
        
    def make_plot(self):
        self.set_fig_frame()
        self.plot_quantities()
        self.make_legend()
        self.manage_output()
        
if __name__ == '__main__': 
    fname = 'line_interaction-downbranch_' + 'C-F2-1.00_C-F1-0.2'
    syn_list = [path_tardis_output + '11fe_' + t + 'd_C-best/' + fname
                     + '/' + fname for t in t_label]  

    Plot_Ionfractions(syn_list, Z=6, show_fig=True, save_fig=True)

