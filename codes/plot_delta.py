#!/usr/bin/env python

import os                                                               

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import MultipleLocator

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

L_list = ['8.505', '9.041', '9.362', '9.505', '9.544']
t_list = ['5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['6', '9', '12', '16', '19']
v_list = ['12400', '11300', '10700', '9000', '7850']

color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

Z2el = {6: 'C', 8: 'O', 14: 'Si', 26: 'Fe'}
ion2symbol = {0: 'I', 1: 'II', 2: 'III'}
par2label = {None: '', 'CII1012': '6580', 'CI1119': '10693'}

fs = 20

def retrieve_delta(_syn, _Z, _ion):
    v_inner = pd.read_hdf(_syn + '.hdf', '/simulation/model/v_inner').values / 1.e5
    delta = pd.read_hdf(_syn + '.hdf', '/simulation/plasma/delta') 
    return v_inner, delta.loc[_Z,_ion].values
    
class Plot_Delta(object):
    """
    Description:
    ------------
    Plots the values of delta for the default simulations of 11fe. delta is
    a parameter defined in the nebular approximation of TARDIS which aims to
    account for the high opacities in the UV.       
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_delta.pdf
    
    References:
    -----------
    Mazzali & Lucy 1993: http://adsabs.harvard.edu/abs/1993A%26A...279..447M
    Marion+ 2006: http://adsabs.harvard.edu/abs/2006ApJ...645.1392M
    TARDIS: http://adsabs.harvard.edu/abs/2014MNRAS.440..387K     
    """
    
    def __init__(self, syn_list, Z, ion, show_fig=True, save_fig=False):
        
        self.syn_list = syn_list
        self.Z = Z
        self.ion = ion
        self.show_fig = show_fig
        self.save_fig = save_fig      
        
        self.F = {}
        self.D = {}
        self.fig = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111) 

        self.make_plot()

    def set_fig_frame(self):
        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = (r'$\delta$')
        self.ax.set_xlabel(x_label, fontsize=20.)
        self.ax.set_ylabel(y_label, fontsize=20.)
        self.ax.set_xlim(7500.,24000.)
        #self.ax.set_ylim(1.e-12,1.e0)
        self.ax.tick_params(axis='y', which='major', labelsize=20., pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=20., pad=8)
        self.ax.tick_params('both', length=8, width=1., which='major')
        self.ax.tick_params('both', length=4, width=1., which='minor')    
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(5000.))

        plt.title(Z2el[self.Z] + ion2symbol[self.ion], fontsize=20.)

    def plot_quantities(self):
        for i, syn in enumerate((self.syn_list)):
            v, delta = retrieve_delta(syn, self.Z, self.ion)
            self.ax.plot(v, delta, ls='-', lw=2., marker='None', color=color[i])

    def make_legend(self):
        for i in range(len(self.syn_list)):
            self.ax.plot([np.nan], [np.nan], ls='-', lw=10., color=color[i],
                         label= r'$\rm{t_{exp}\ =\ ' + t_list[i] + '\ \mathrm{d}}$')
        self.ax.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1,
                       labelspacing=0.1, loc='best')        
            
    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/FIGURES/Fig_delta.pdf'
            plt.savefig(fpath , format='pdf')
        if self.show_fig:
            plt.show()
        plt.close()
        
    def make_plot(self):
        self.set_fig_frame()
        self.plot_quantities()
        #self.make_legend()
        self.manage_output()
        
if __name__ == '__main__': 
    
    #Mazzali profiles.
    case_folder = path_tardis_output + '11fe_default_L-scaled/'
    f1 = 'velocity_start-'
    f2 = '_loglum-'
    f3 = '_line_interaction-downbranch_time_explosion-'
    syn_list_orig = [case_folder + (f1 + v + f2 + L + f3 + t) + '/'
                     + (f1 + v + f2 + L + f3 + t)
                     for (v, L, t) in zip(v_list, L_list, t_list)]

    #Accepted model profile
    X_i = '0.2'
    X_o = '1.00'

    fname = 'line_interaction-downbranch_excitation-dilute-lte_'\
            + 'C-F2-' + X_o + '_C-F1-' + X_i
    syn_list = [path_tardis_output + '11fe_' + t + 'd_C-best/' + fname
                     + '/' + fname for t in t_label]  
    
    #fname = 'velocity_start-10700_loglum-9.362_line_interaction-downbranch_C-F2-1.00_C-F1-0.2_time_explosion-12.1'
    #syn_list = [path_tardis_output + '11fe_best_delta/' + fname + '/' + fname]

    fname = 'seed-23111970'
    syn_list = [path_tardis_output + 'fast_single_blank/' + fname + '/' + fname]

    Plot_Delta(syn_list, Z=6, ion=1, show_fig=True, save_fig=False)

 
