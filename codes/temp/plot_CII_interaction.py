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
from astropy import constants as const
                             
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

atomic_mass = const.u.cgs.value
M_sun = const.M_sun.cgs.value

class Analyse_Iter(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, fpath, show_fig=True, save_fig=False):

        self.fpath = fpath
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.FIG = plt.figure(figsize=(10,10))
        self.ax1 = plt.subplot(111)
        
        self.hdf_fname = None
        self.pkl = None
        self.hdf = None
        
        self.model_t_rad = None
        self.model_dens = None
        self.model_v_inner = None
        self.shell = None
       
        self.Q = {}
        self.C_fraction = []
        self.Fe_fraction = []
        self.var_list = ['C_dens', 'CII_dens', 'CII10_dens']
        
        #Variables pertaining y2 mode:
        self.y2 = None
        self.y2_label = None
        self.y2_legend = None
        self.name_suffix = None
        
        self.fsize = 26

        for var in self.var_list:
            self.Q[var] = []
        
        self.make_plot() 
            
    def set_fig_frame(self):
        
        self.ax1.set_yscale('log')
        x_label = r'$v \ \rm{[km \ s^{-1}]}$'
        y1_label = r'Number of packets'
        self.ax1.set_xlabel(x_label, fontsize=self.fsize)
        self.ax1.set_ylabel(y1_label, fontsize=self.fsize)
        self.ax1.set_xlim(6000.,20000.)
        self.ax1.set_ylim(1.e-4,100)      
        self.ax1.tick_params(axis='y', which='major', labelsize=self.fsize, pad=8)      
        self.ax1.tick_params(axis='x', which='major', labelsize=self.fsize, pad=8)
        self.ax1.minorticks_on()
        self.ax1.tick_params('both', length=8, width=1, which='major')
        self.ax1.tick_params('both', length=4, width=1, which='minor')  
        self.ax1.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax1.xaxis.set_major_locator(MultipleLocator(5000.)) 

    def load_data(self):
        fname = '/' + self.fpath.split('/')[-1]              
        with open(self.fpath + fname + '.pkl', 'r') as inp:
            self.pkl = cPickle.load(inp)
        
        self.hdf_fname = self.fpath + fname + '.hdf'
        self.hdf = pd.HDFStore(self.hdf_fname)
        #print self.hdf
        
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
        self.model_number_dens = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/number_density')        
        self.model_t_rad = pd.read_hdf(
          self.hdf_fname, '/simulation15/model/t_radiative').values   
        self.model_dens = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/density').values * u.g / u.cm**3
        abun = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/abundance')
        self.model_v_inner = (pd.read_hdf(
          self.hdf_fname, '/simulation15/model/v_inner').values * u.cm / u.s).to(u.km / u.s)
        self.shell = np.arange(0, len(self.model_dens), 1)

        self.Q['X_C'] = []
        
        
        for shell in self.shell:
            C_frac, Fe_frac, N_packets = LLI(
              shell, last_line_interaction_in_id, last_line_interaction_out_id,
              last_line_interaction_shell_id, output_nu, lines).run_analysis()

            self.C_fraction.append(C_frac)
            self.Fe_fraction.append(Fe_frac)

            #print shell, sum(model_lvl_dens[shell])
            self.Q['C_dens'].append(sum(model_lvl_dens[shell][6]))
            self.Q['CII_dens'].append(sum(model_lvl_dens[shell][6][1]))
            self.Q['CII10_dens'].append(model_lvl_dens[shell][6][1][10]
                                        + model_lvl_dens[shell][6][1][11])
            self.Q['X_C'].append(abun[shell][6])                            
                                
        #Convert lists to arrays.
        for var in self.var_list:
            self.Q[var] = np.asarray(self.Q[var])
        self.Q['X_C'] = np.asarray(self.Q['X_C'])
        self.C_fraction = np.asarray(self.C_fraction)
        self.Fe_fraction = np.asarray(self.Fe_fraction)
        
        self.hdf.close()

    def compute_column_density(self):
        """Column density is given as number of atoms (satisfying given
        criterium) per cm^2 [# cm^-2]. For instance, 'CII_coldens' is the
        number of singly ionized carbon atoms per cm^2. Numbers are given per
        #shell.
        """

        #C_mass = np.multiply(self.pkl['volume'],self.Q['CII_dens']).value
        #C_mass = C_mass * atomic_mass / M_sun
        #print sum(C_mass)

        r_inner = self.pkl['r_inner']
        r_outer = self.pkl['r_outer']
        delta_r = r_outer - r_inner
        
        new_var_list = ['C_coldens', 'CII_coldens', 'CII10_coldens']
        
        for var, new_var in zip(self.var_list, new_var_list):
            self.Q[new_var] = (self.Q[var] * delta_r).value
        
    def plot_data(self):

        r_inner = self.pkl['r_inner']
        r_outer = self.pkl['r_outer']
        delta_r = r_outer - r_inner
        
        #C_numdens = np.multiply(self.model_dens, self.Q['X_C']) / (12. * const.u.to('g'))
        #print np.divide(C_numdens.value - self.Q[ 'C_dens'], self.Q[ 'C_dens'])
        
        #Plot fraction of C trough forming packets.
        self.ax1.plot(
          self.model_v_inner[1:], self.C_fraction[1:], ls='None', marker='s',
          markersize=10., color='gold', zorder=2.,
          label='C trough forming')
                
        #Plot fraction of Fe trough filling packets.        
        self.ax1.plot(
          self.model_v_inner, self.Fe_fraction, ls='None', marker='s',
          markersize=10., color='m', zorder=2.,
          label='Fe trough filling')        
        
        #Plot columnd density of C_II ions at level 10 or 11. 
        self.ax1.plot(
          self.model_v_inner[1:], self.Q[ 'CII10_coldens'][1:] / 1.e13, 
          ls='-', marker='^', markersize=10., color='gold', zorder=2.,
          label=r'$ \eta(\rm{CII-10-11)\ [10^{-13} \ cm^{-2}]}$')          
             
        self.ax1.legend(frameon=False, fontsize=self.fsize, numpoints=1, ncol=1,
                       labelspacing=0.05, loc='best') 

        plt.tight_layout()

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_trough_diagnosis_001.png',
                        format='png', dpi=360)
        
    def make_plot(self):
        self.set_fig_frame()
        self.load_data()
        self.analyse_data()
        self.compute_column_density()
        self.plot_data()
        self.save_figure()
        if self.show_fig:
            plt.show()

if __name__ == '__main__':

    input_fpath = (
      path_tardis_output + '05bl_default_L-scaled_extra0.01Fe/Fe0-+0.01_loglum'
      + '-8.648_line_interaction-macroatom_velocity_start-7600_time_explosion-14.0')
    
    Analyse_Iter(fpath=input_fpath, show_fig=True, save_fig=False)

        
    #Analyse_Iter(y2_mode='T', mode='line_in_nu', lm='macroatom', L='8.283',
    #             Fes='10.0', show_fig=True, save_fig=False)
    #Analyse_Iter(y2_mode='dens', mode='line_in_nu', lm='macroatom', L='8.505',
    #             Fes='0.04', show_fig=True, save_fig=True)
    #Analyse_Iter(y2_mode='lvl_frac', mode='line_in_nu', lm='macroatom', L='8.505',
    #             Fes='0.04', show_fig=True, save_fig=True)

    #Questions:
    #Where are the virtual pack interactions stored?


