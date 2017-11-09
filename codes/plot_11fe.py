#!/usr/bin/env python

import os                                                               
import sys
import time

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class Plot_11fe_Sequence(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, show_fig=True, save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig 
        self.fig, self.ax = plt.subplots(figsize=(10.,18))

        self.obs_list = [
          '2011_08_25', '2011_08_28', '2011_08_31', '2011_09_03',
          '2011_09_07', '2011_09_10', '2011_09_13', '2011_09_19']
        
        self.syn_list = [
          ['7.903', '13300', '3.7'], ['8.505', '12400', '5.9'],
          ['9.041', '11300', '9.0'], ['9.362', '10700', '12.1'],
          ['9.505', '9000', '16.1'], ['9.544', '7850', '19.1'],
          ['9.505', '6700', '22.4'], ['9.362', '4550', '28.3']]  

        self.make_plot()     
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'
        
        fs = 26.
        self.ax.set_xlabel(x_label,fontsize=fs)
        self.ax.set_ylabel(y_label,fontsize=fs)
        self.ax.set_xlim(1500., 10000.)
        self.ax.set_ylim(-5.5, 12.)      
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(2.))        
        self.ax.tick_params(labelleft='off')  

    def load_and_plot_observational_spectrum(self):
        
        directory = '/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'\
                    + 'observational_spectra/2011fe/'
        
        offset = 9.
        for date in self.obs_list:
            with open(directory + date + '.pkl', 'r') as inp:
                pkl = cPickle.load(inp)
                self.ax.plot(
                  pkl['wavelength_corr'], pkl['flux_smoothed'] + offset,
                  color='k', ls='-', lw=3., zorder=2.)
            offset -= 2.  
                    
    def load_and_plot_synthetic_spectrum(self):
        
        def make_fpath(L, v, t):
            fname = ('line_interaction-downbranch_loglum-' + L
                    + '_velocity_start-' + v + '_time_explosion-' + t)
            return (path_tardis_output + '11fe_default_L-scaled_UP/' + fname
                    + '/' + fname + '.pkl')
        
        syn_fpath_list = [make_fpath(s[0], s[1], s[2]) for s in self.syn_list]    
                    
        offset = 9.
        for fpath in syn_fpath_list:
            with open(fpath, 'r') as inp:
                pkl = cPickle.load(inp)
                self.ax.plot(
                  pkl['wavelength_corr'], pkl['flux_smoothed'] + offset,
                  color='b', ls='-', lw=3., zorder=2.)
            offset -= 2.                      

    def add_text(self):
        offset = 9.
        for s in self.syn_list:
            self.ax.text(1800., offset + 0.4, s[2] + ' d',
                         fontsize=20., horizontalalignment='left')        
            offset -= 2.                      
                    
    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_11fe_sequence.png',
                        format='png', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_and_plot_observational_spectrum()
        self.load_and_plot_synthetic_spectrum()
        self.add_text()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()    

if __name__ == '__main__':
    Plot_11fe_Sequence(show_fig=True, save_fig=False)
    

          
          
