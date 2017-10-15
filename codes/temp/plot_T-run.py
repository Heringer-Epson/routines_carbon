#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import pandas as pd
import cPickle   
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


class Compare_11fe(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, show_fig=True, save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig

        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14    
        self.run_comparison()

        self.pkl_T = None
        self.pkl_L = None
        
    def set_fig_frame(self):
        
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Absolute \ f}_{\lambda}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs_label)
        self.ax.set_ylabel(y_label,fontsize=self.fs_label)
        self.ax.set_xlim(1500.,10000.)
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

    def load_data(self):

        #Load simplified (homogeneous) synthetic spectrum.
        case_folder = path_tardis_output + 'quick_test/'
        
        fname = 'loglum-8.538'
        fullpath = case_folder + fname + '/' + fname + '.pkl'                  
        with open(fullpath, 'r') as inp:
            self.pkl_L = cPickle.load(inp)

        fname = 'loglum-8.538_temperature_requested-9734.410'
        fullpath = case_folder + fname + '/' + fname + '.pkl'                  
        with open(fullpath, 'r') as inp:
            self.pkl_T = cPickle.load(inp)        

    def plot_spectra(self):
        
        w_L= self.pkl_L['wavelength_corr']
        f_L = self.pkl_L['flux_smoothed']

        w_T= self.pkl_T['wavelength_corr']
        f_T = self.pkl_T['flux_smoothed']       
  
        self.ax.plot(w_L, f_L, ls='-', lw=2., color='b', label='L set')
        self.ax.plot(w_T, f_T, ls='-', lw=2., color='r', label='T set')
    
        self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=1) 
                       
                       
        print 'L:', self.pkl_L['t_inner'], self.pkl_L['w']                       
        print 'T:', self.pkl_T['t_inner'], self.pkl_T['w']                             

    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_11fe_homogeneous_' + self.lm + '.'
                        + extension, format=extension, dpi=dpi)
    
    def run_comparison(self):
        self.set_fig_frame()
        self.load_data()
        self.plot_spectra()
        self.save_figure()
        if self.show_fig:
            plt.show()

        
Compare_11fe(show_fig=True, save_fig=False)




