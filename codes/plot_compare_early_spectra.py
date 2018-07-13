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

class Macroatom_Comparison(object):
    
    def __init__(self, show_fig=True, save_fig=False):
        """Creates a figure where spectra computed using the 'downbranch' and
        'macroatom' modes are compared for both 11fe and 05bl.
        """

        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
        self.pkl_11fe_macroatom, self.pkl_11fe_downbranch = None, None  
        self.pkl_05bl_macroatom, self.pkl_05bl_downbranch = None, None  
        self.pkl_11fe_obs, self.pkl_11fe_obs = None, None  

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14    
        self.run_comparison()
        
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
    
    def load_spectra(self):
            
        inp_dir = path_tardis_output + '11fe_default_L-scaled/'
        fname = 'velocity_start-13300_loglum-7.903_line_interaction-downbranch'\
                + '_time_explosion-3.7'
        with open(inp_dir + fname + '/' + fname + '.pkl', 'r') as inp:
            self.pkl_11fe = cPickle.load(inp)        

        path_data = './../INPUT_FILES/observational_spectra/'
        with open(path_data + '2011fe/2011_08_25.pkl', 'r') as inp:
            self.pkl_11fe_obs = cPickle.load(inp)                                        

    def plotting(self):

        offset = -1.5

        """11fe"""
        #Observational
        wavelength_obs = self.pkl_11fe_obs['wavelength_corr']
        flux_obs = self.pkl_11fe_obs['flux_normalized']
        
        self.ax.plot(
          wavelength_obs, flux_obs, color='k', linewidth=1., label='SN 2011fe')

        #Downbranch
        wavelength_11fe = self.pkl_11fe['wavelength_corr']
        flux_11fe = self.pkl_11fe['flux_normalized']
          
        self.ax.plot(wavelength_11fe, flux_11fe,
          color='b', linewidth=1., label=r'${\rm 2011fe} \, - \, '
          + r't_{\mathrm{exp}=3.9\} d$')              

    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_compare_early_spectra.pdf',
                        format='pdf', dpi=dpi)
        
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

compare_spectra_object = Macroatom_Comparison(show_fig=True, save_fig=False)
