#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import itertools
from matplotlib.ticker import MultipleLocator
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class Neutral_C(object):
    
    def __init__(self, case='hot', show_fig=True, save_fig=False):

        self.case = case
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
        self.pkl_0, self.pkl_10 = None, None  

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14    
        
        if self.case == 'hot':
            self.L = '9.362'
        elif self.case == 'cold':
            self.L = '8.5'

        self.run_comparison()

        
    def set_fig_frame(self):
        
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Absolute \ f}_{\lambda}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs_label)
        self.ax.set_ylabel(y_label,fontsize=self.fs_label)
        self.ax.set_xlim(1500.,12000.)
        self.ax.set_ylim(-1.5,3.5)      
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(2.))        
        self.ax.tick_params(labelleft='off')
        
        self.ax.set_title(r'11fe @ 12d with log $L= ' + self.L + '$',
                          fontsize=self.fs_label)                
    
    def load_spectra(self):

        def get_path(C): 
            case_folder = path_tardis_output + '11fe_12d_C-neutral_test/'
            #case_folder = path_tardis_output + '11fe_12d_C-neutral_kromer/'
            filename = ('loglum-' + self.L + '_C-F1-' + C)
            path_sufix = filename + '/' + filename + '.pkl'
            return case_folder + path_sufix
            
        """11fe"""

        fname = get_path('0.00')
        with open(fname, 'r') as inp:
            self.pkl_0 = cPickle.load(inp)        
        
        fname = get_path('10.00')
        with open(fname, 'r') as inp:
            self.pkl_10 = cPickle.load(inp)
                

    def plotting(self):

        """11fe"""
        self.ax.plot(
          self.pkl_0['wavelength_corr'], self.pkl_0['flux_normalized'],
          color='r', linewidth=1., label='No C')

        self.ax.plot(
          self.pkl_10['wavelength_corr'], self.pkl_10['flux_normalized'],
          color='b', linewidth=1., label='With C')

        self.ax.legend(frameon=False, fontsize=20, numpoints=1, ncol=1, loc=1)  

    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_test_neutral_C_' + self.case +'.'
                         + extension, format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def run_comparison(self):
        self.set_fig_frame()
        self.load_spectra()
        plt.tight_layout()
        self.plotting()
        self.save_figure()
        self.show_figure()  

compare_spectra_object = Neutral_C(case='hot', show_fig=True, save_fig=False)
compare_spectra_object = Neutral_C(case='cold', show_fig=True, save_fig=False)

#compare_spectra_object = Neutral_C(case='hot', show_fig=False, save_fig=True)
#compare_spectra_object = Neutral_C(case='cold', show_fig=False, save_fig=True)
