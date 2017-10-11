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
    
    def __init__(self, L, show_fig=True, save_fig=False):

        self.L = L
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.FIG = plt.figure(figsize=(14,8))
        self.ax = plt.subplot(111)  

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14    
        self.run_comparison()

        self.pkl_syn_down_orig = None
        self.pkl_syn_down_new = None
        self.pkl_syn_macr_new = None
        self.pkl_obs = None
        
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

        #Load original downbranch model of 11fe-05bl paper.
        case_folder = path_tardis_output + '11fe_default_L-scaled/'
        fname = ('velocity_start-12400_loglum-' + self.L + '_line_interaction'
                 + '-downbranch_time_explosion-5.9')
        fullpath = case_folder + fname + '/' + fname + '.pkl'                  
        with open(fullpath, 'r') as inp:
            self.pkl_syn_down_orig = cPickle.load(inp)
        
        
        #Load new stratified downbranch model.
        case_folder = path_tardis_output + '11fe_default_L-scaled_new/'
        fname = ('velocity_start-12400_loglum-' + self.L + '_line_interaction'
                 + '-downbranch_time_explosion-5.9')
        fullpath = case_folder + fname + '/' + fname + '.pkl'          
        with open(fullpath, 'r') as inp:
            self.pkl_syn_down_new = cPickle.load(inp)        


        #Load new stratified macroatom model.
        case_folder = path_tardis_output + '11fe_default_L-scaled_new/'
        fname = ('velocity_start-12400_loglum-' + self.L + '_line_interaction'
                 + '-macroatom_time_explosion-5.9')
        fullpath = case_folder + fname + '/' + fname + '.pkl'          
        with open(fullpath, 'r') as inp:
            self.pkl_syn_macr_new = cPickle.load(inp)    
       
       
        #Load observed spectrum. 
        fullpath = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'\
                    + 'observational_spectra/2011fe/2011_08_28.pkl')
        with open(fullpath, 'r') as inp:
            self.pkl_obs = cPickle.load(inp)

    def plot_spectra(self):
        
        w_syn_down_orig = self.pkl_syn_down_orig['wavelength_corr']
        f_syn_down_orig = self.pkl_syn_down_orig['flux_smoothed']

        w_syn_down_new = self.pkl_syn_down_new['wavelength_corr']
        f_syn_down_new = self.pkl_syn_down_new['flux_smoothed']
        
        w_syn_macr_new = self.pkl_syn_macr_new['wavelength_corr']
        f_syn_macr_new = self.pkl_syn_macr_new['flux_smoothed']     

        w_obs = self.pkl_obs['wavelength_corr']
        f_obs = self.pkl_obs['flux_smoothed']        
        
       
        self.ax.plot(w_syn_down_orig, f_syn_down_orig, ls='-', lw=2.,
                     color='b', label='Original downbranch model')
        
        self.ax.plot(w_syn_down_new, f_syn_down_new, ls='-', lw=2.,
                     color='c', label='New downbranch model')
                     
        self.ax.plot(w_syn_macr_new, f_syn_macr_new, ls='-', lw=2.,
                     color='r', label='New macroatom model')
                                
        self.ax.plot(w_obs, f_obs, ls='-', lw=3., color='k',
                     label='Observed spectrum at 5.9 days') 
                     
        self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=1)           

    def save_figure(self, extension='png', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_11fe_early-comparison.'
                        + extension, format=extension, dpi=dpi)
    
    def run_comparison(self):
        self.set_fig_frame()
        self.load_data()
        self.plot_spectra()
        self.save_figure()
        if self.show_fig:
            plt.show()

        
Compare_11fe(L='8.505', show_fig=True, save_fig=True)




