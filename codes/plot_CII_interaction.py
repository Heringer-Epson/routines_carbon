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
from retrieve_interactions import LastLineInteraction as LLI
from astropy import units as u
                             
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class Analyse_Iter(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, y2_mode, mode, lm, L, Fes, show_fig=True, save_fig=False):

        self.y2_mode = y2_mode
        self.mode = mode
        self.lm = lm
        self.L = L
        self.Fes = Fes
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.FIG = plt.figure(figsize=(10,10))
        self.ax1 = plt.subplot(111)
        self.ax2 = self.ax1.twinx()  
        
        self.hdf_fname = None
        self.pkl = None
        self.hdf = None
        
        self.model_t_rad = None
        self.model_dens = None
        self.model_v_inner = None
        self.shell = None
        self.lvl_frac = []
        self.C_fraction = []
        self.Fe_fraction = []
        
        #Variables pertaining y2 mode:
        self.y2 = None
        self.y2_label = None
        self.y2_legend = None
        self.name_suffix = None
        
        self.fsize = 26
        
        self.make_plot()

    def set_y2mode(self):
        if self.y2_mode == 'T':
            self.y2_label = r'$T \ \rm{[K]}$'
            self.y2_legend = 'Temperature'
            self.name_suffix = 'y2-T'
        elif self.y2_mode == 'lvl_frac':
            self.y2_label = r'Fraction of C_II ions in level 10'
            self.y2_legend = 'C_II in level 10'
            self.name_suffix = 'y2-lvl10'        
        elif self.y2_mode == 'dens':
            self.y2_label = r'$\rho \ \rm{[10^{13} \ g\ cm^{-3}]}$'
            self.y2_legend = 'Density'
            self.name_suffix = 'y2-dens'   
            
    def set_fig_frame(self):
        
        self.ax1.set_yscale('log')
        
        x_label = r'$v \ \rm{[km \ s^{-1}]}$'
        y1_label = r'Fraction of packets'
        
        self.ax1.set_xlabel(x_label, fontsize=self.fsize)
        self.ax1.set_ylabel(y1_label, fontsize=self.fsize)
        #self.ax1.set_xlim(1500.,10000.)
        #self.ax1.set_ylim(-1.5,3.5)      
        self.ax1.tick_params(axis='y', which='major', labelsize=self.fsize, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=self.fsize, pad=8)
        self.ax1.minorticks_on()
        self.ax1.tick_params('both', length=8, width=1, which='major')
        self.ax1.tick_params('both', length=4, width=1, which='minor')  

        self.ax2.set_ylabel(self.y2_label, fontsize=self.fsize)  
        self.ax2.tick_params(axis='y', which='major', labelsize=self.fsize, pad=8)      
        self.ax2.tick_params(axis='x', which='major', labelsize=self.fsize, pad=8)
        self.ax2.minorticks_on()
        self.ax2.tick_params('both', length=8, width=1, which='major')
        self.ax2.tick_params('both', length=4, width=1, which='minor')     
        self.ax2.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax2.xaxis.set_major_locator(MultipleLocator(2000.))  
        
    def load_data(self):

        case_folder = path_tardis_output + 'early-carbon-grid/'
        def get_fname(lm, L, Fe): 
            fname = 'loglum-' + L + '_line_interaction-' + lm + '_Fes-' + Fe
            fname = case_folder + fname + '/' + fname
            return fname                  
        
        with open(get_fname(self.lm, self.L, self.Fes) + '.pkl', 'r') as inp:
            self.pkl = cPickle.load(inp)
        
        self.hdf_fname = get_fname(self.lm, self.L, self.Fes) + '.hdf'
        self.hdf = pd.HDFStore(self.hdf_fname)
        
    def analyse_data(self):

        #Initialize variables:
        last_line_interaction_in_id = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/last_line_interaction_in_id')
        last_line_interaction_out_id = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/last_line_interaction_out_id')
        last_line_interaction_shell_id = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/last_line_interaction_shell_id')
        output_nu = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/output_nu')
        lines = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/lines')
        model_lvl_dens = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/level_number_density')
        self.model_t_rad = pd.read_hdf(
          self.hdf_fname, '/simulation15/model/t_radiative').values   
        self.model_dens = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/density').values * u.g / u.cm**3
        self.model_v_inner = (pd.read_hdf(
          self.hdf_fname, '/simulation15/model/v_inner').values * u.cm / u.s).to(u.km / u.s)
        self.shell = np.arange(0, len(self.model_dens), 1)

        for shell in self.shell:

            self.lvl_frac.append((model_lvl_dens[shell][6][1][10]
                              / sum(model_lvl_dens[shell][6][1])))
            
            C_frac, Fe_frac = LLI(shell, last_line_interaction_in_id,
                                  last_line_interaction_out_id,
                                  last_line_interaction_shell_id,
                                  output_nu, lines).run_analysis()

            self.C_fraction.append(C_frac)
            self.Fe_fraction.append(Fe_frac)
                                
        self.hdf.close()

    def plot_data(self):
        
        if self.y2_mode == 'T':
            self.y2 = self.model_t_rad
        elif self.y2_mode == 'lvl_frac':
            self.y2 = self.trough_frac
        elif self.y2_mode == 'dens':
            self.y2 = self.model_dens * 1.e13
                    
        self.ax1.plot(self.model_v_inner, self.C_fraction, ls='None', marker='s',
                      markersize=10., color='b', zorder=2.)

        self.ax1.plot(self.model_v_inner, self.Fe_fraction, ls='None', marker='o',
                      markersize=10., color='g', zorder=2.)

        self.ax2.plot(self.model_v_inner, self.y2, ls='None', marker='^',
                      markersize=10., color='r', zorder=1.)        
    
    def add_title_and_legend(self):
        
        title = r'11fe_early_CII-interaction_' + self.name_suffix
        self.ax1.set_title(title, fontsize=self.fsize)        
        
        self.ax1.plot([np.nan], [np.nan], marker='s', markersize=10.,
                      color='b', ls='None', label='Trough forming')
        self.ax1.plot([np.nan], [np.nan], marker='^', markersize=10.,
                      color='r', ls='None', label=self.y2_legend)

        self.ax1.legend(frameon=False, fontsize=self.fsize, numpoints=1, ncol=1,
                       labelspacing=0.05, loc='best') 

        plt.tight_layout()

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_CII-interaction_' + self.name_suffix
                        + '_' + self.lm + '.pdf', format='pdf', dpi=360)
        
    def make_plot(self):
        self.set_y2mode()
        self.set_fig_frame()
        self.load_data()
        self.analyse_data()
        self.plot_data()
        self.add_title_and_legend()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
Analyse_Iter(y2_mode='T', mode='line_in_nu', lm='macroatom', L='8.283',
             Fes='10.0', show_fig=True, save_fig=False)
#Analyse_Iter(y2_mode='dens', mode='line_in_nu', lm='macroatom', L='8.505',
#             Fes='0.04', show_fig=True, save_fig=True)
#Analyse_Iter(y2_mode='lvl_frac', mode='line_in_nu', lm='macroatom', L='8.505',
#             Fes='0.04', show_fig=True, save_fig=True)

#Questions:
#Where are the virtual pack interactions stored?


