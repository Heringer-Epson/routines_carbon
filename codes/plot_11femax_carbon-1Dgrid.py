#!/usr/bin/env python

""" Code to make a plot of the pEW of the strong silicon feature (f7) vs.
the weak silicon feature (f6). This includes objects from the BSNIP sample
and the simulation of 11fe and 05bl with scaled luminosity.
"""

import os                                                               
import sys
import time

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
import matplotlib.collections as mcoll

import colormaps as cmaps
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class Carbon_L(object):

    def __init__(self, lm='downbranch', show_fig=True, save_fig=False):
                                
        self.lm = lm
        self.show_fig = show_fig
        self.save_fig = save_fig
                
        self.L_array = np.logspace(8.544, 9.72, 20)[::-1]
               
        self.list_label_11fe, self.list_label_05bl = [], []
        self.list_pkl_11fe, self.list_L_05bl = [], []
      
        self.FIG = plt.figure(figsize=(10, 10))
        self.ax = plt.subplot(111)  
        self.color_11fe = None

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
                   
        self.run_parspace()
        
    def set_fig_frame(self):
       

        self.ax.set_xlabel(r'$L / L_{\mathrm{11fe}}  $' ,fontsize=self.fs_label)        
        self.ax.set_ylabel(r'pEW $\mathrm{[\AA]}$ of '
          + r'$\rm{C}\,\mathrm{II} \ \lambda$6580', fontsize=self.fs_label) 
        #self.ax.set_xlim(40.,180.)
        #self.ax.set_ylim(0.,70.)
        #self.ax.xaxis.set_minor_locator(MultipleLocator(5.))
        #self.ax.xaxis.set_major_locator(MultipleLocator(20.))
        #self.ax.yaxis.set_minor_locator(MultipleLocator(2.))
        #self.ax.yaxis.set_major_locator(MultipleLocator(10.))
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
    
    def plotting(self):

        case_folder = path_tardis_output + '11fe_L-grid/'                        

        x, y, y_unc = [], [], []

        for i, L in enumerate(self.L_array):
            if  L / 3.5e9 <= 1.5 and L / 3.5e9 > 0.28:

                L_str = str(format(np.log10(L), '.3f'))       
                fname = ('loglum-' + L_str +  '_line_interaction-' + self.lm) 
                with open(case_folder + fname + '/' + fname + '.pkl', 'r') as inp:
                    pkl = cPickle.load(inp)
                    x.append(L / 3.5e9)
                    y.append(pkl['pEW_fC'])
                    y_unc.append(pkl['pEW_unc_fC'])
    
        self.ax.errorbar(
          x, y, yerr=y_unc, ls='-', marker='s', markersize=10., capsize=0.,
          color='k', zorder=1)                       

    def add_legend(self):
        
        self.ax.errorbar([np.nan], [np.nan], xerr=[np.nan], yerr=[np.nan],
                         ls='-', marker='D', markersize=10., capsize=0.,
                         color=self.color_11fe, label=r'SN 2011fe grid')                       
        
        self.ax.errorbar([np.nan], [np.nan], xerr=[np.nan], yerr=[np.nan],
                         ls=':', marker='p', markersize=14., capsize=0.,
                         color='g',label=r'SN 2005bl grid')                       
        
        subtypes = ['Ia-norm', 'Ia-91bg', 'Ia-91T', 'Ia-99aa', 'other']      
        markers = subtype2marker(subtypes)
               
        #Plot nan to get legend entries.
        for (subtype,marker) in zip(subtypes,markers):
            self.ax.errorbar([np.nan], [np.nan], xerr=[np.nan], yerr=[np.nan],
                             ls='None', marker=marker, color='gray', alpha=0.5,
                             markersize=9., capsize=0., label=subtype)        
        
        self.ax.legend(frameon=True, fontsize=20., numpoints=1, ncol=1,
                       handletextpad=0.2, labelspacing=0.05, loc=2)

    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_carbon_L-grid_' + self.lm + '.'
                        + extension, format=extension, dpi=dpi)
        
    def run_parspace(self):
        self.set_fig_frame()
        self.plotting()
        #self.add_legend()
        self.save_figure(extension='pdf')
        if self.show_fig:
            plt.show()             

parspace_object = Carbon_L(lm='downbranch', show_fig=True, save_fig=True)


