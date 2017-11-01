#!/usr/bin/env python

import os                                                               
import sys
import time

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import new_colormaps as cmaps

from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))
L_scal = np.arange(0.2, 1.61, 0.1)

texp2date = {'3.7': '2011_08_25', '5.9': '2011_08_28', '9.0': '2011_08_31',
             '12.1': '2011_09_03', '16.1': '2011_09_07', '19.1': '2011_09_10',
             '22.4': '2011_09_13', '28.3': '2011_09_19'}

texp2v = {'3.7': '13300', '5.9': '12400', '9.0': '11300',
          '12.1': '10700', '16.1': '9000', '19.1': '7850',
          '22.4': '6700', '28.3': '4550'}

texp2L = {'3.7': 0.08e9, '5.9': 0.32e9, '9.0': 1.1e9,
          '12.1': 2.3e9, '16.1': 3.2e9, '19.1': 3.5e9,
          '22.4': 3.2e9, '28.3': 2.3e9}

class Plot_C_scaled_Spectra(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, t_exp='12.1', L='9.362', show_fig=True, save_fig=False):

        self.t_exp = t_exp
        self.L = L
        self.show_fig = show_fig
        self.save_fig = save_fig 
        self.fig, self.ax = plt.subplots(figsize=(14,8))
        self.fs = 26.
        
        self.C_list = ['0.00', '0.20', '0.50', '1.00', '2.00', '5.00',
                       '10.00', '20.00']
        self.make_plot()     
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'
        
        fs = 26.
        self.ax.set_xlabel(x_label,fontsize=self.fs)
        self.ax.set_ylabel(y_label,fontsize=self.fs)
        self.ax.set_xlim(1500., 10000.)
        self.ax.set_ylim(-0.2, 3.5)      
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(1.))        
        self.ax.tick_params(labelleft='off')  

    def load_and_plot_observational_spectrum(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        with open(directory + texp2date[self.t_exp] + '.pkl', 'r') as inp:
            pkl = cPickle.load(inp)
            self.ax.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                         color='k', ls='-', lw=3., zorder=2.)
                
    def load_and_plot_synthetic_spectrum(self):

        cmap = cmaps.viridis
        Norm = colors.Normalize(vmin=0., vmax=7.)
        
        def make_fpath(C):
            fname = 'C-' + C + '_loglum-' + self.L
            return (path_tardis_output + '11fe_12d_C-scaled_v0/' + fname
                    + '/' + fname + '.pkl')
                    
        for i, C in enumerate(self.C_list):
            with open(make_fpath(C), 'r') as inp:
                pkl = cPickle.load(inp)
                self.ax.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                             color=cmap(Norm(i)), ls='-', lw=3., zorder=2.)

        #Add colorbar
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=Norm)
        aux_mappable.set_array([])
        cbar = plt.colorbar(aux_mappable)
        cbar.set_ticks(np.arange(0, 7.01, 1.))
        cbar.ax.tick_params(width=1, labelsize=self.fs)
        cbar.set_label('C scale factor', fontsize=self.fs)     
        cbar.set_ticklabels(self.C_list)

    def add_L_text(self):
        frac = str(format(10. ** float(self.L) / texp2L[self.t_exp], '.1f'))
        text = r'$L\ =\ ' + frac + '\\times L_{\\mathrm{11fe}}$'
        self.ax.text(1800., 3.2, text, fontsize=20., horizontalalignment='left')  
                          
    def save_figure(self):
        top_dir = './../OUTPUT_FILES/FIGURES/C_scaled_spectra/'
        if not os.path.isdir(top_dir):
            os.mkdir(top_dir)        
        
        outdir = top_dir + 't_exp-' + self.t_exp + '/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
                      
        if self.save_fig:
            fpath = outdir + 'Fig_11fe_C_scaled_spectra_L-' + self.L + '.png'
            plt.savefig(fpath, format='png', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_and_plot_observational_spectrum()
        self.load_and_plot_synthetic_spectrum()
        self.add_L_text()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()


class Plot_C_scaled_Trough(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, t_exp='12.1', L='9.362', show_fig=True, save_fig=False):

        self.t_exp = t_exp
        self.L = L
        self.show_fig = show_fig
        self.save_fig = save_fig 
        self.fig, self.ax = plt.subplots(figsize=(14,8))
        self.fs = 26.
        
        self.C_list = ['0.00', '0.20', '0.50', '1.00', '2.00', '5.00',
                       '10.00', '20.00']
        self.make_plot()     
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$v \ \mathrm{[km\ s^{-1}}]}$'
        y_label = r'n$(\rm{C_{III,n=10-11}})/n(\rm{C})$'
        
        fs = 26.
        self.ax.set_yscale('log')
        self.ax.set_xlabel(x_label,fontsize=self.fs)
        self.ax.set_ylabel(y_label,fontsize=self.fs)
        self.ax.set_xlim(10000., 15000.)
        #self.ax.set_ylim(1.e-12, 1.e-8)      
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax.xaxis.set_major_locator(MultipleLocator(1000.))

    def load_and_plot_synthetic_spectrum(self):

        cmap = cmaps.plasma
        Norm = colors.Normalize(vmin=0., vmax=7.)
        
        def make_fpath(C):
            fname = 'C-' + C + '_loglum-' + self.L
            return (path_tardis_output + '11fe_12d_C-scaled_v0/' + fname
                    + '/' + fname + '.hdf')
                    
        '''
        r i, C in enumerate(self.C_list):
         
            hdf = make_fpath(C)
            
            v_inner = pd.read_hdf(hdf, '/simulation/model/v_inner') / 1.e5
            lvl_10 = pd.read_hdf(
              hdf, '/simulation/plasma/level_number_density').loc[6,1,10]
            lvl_11 = pd.read_hdf(
              hdf, '/simulation/plasma/level_number_density').loc[6,1,11]
            
            total_ions = 0.
            for k in range(5):
                total_ions += sum(pd.read_hdf(
                  hdf, '/simulation/plasma/ion_number_density').loc[6,k])
            
            #C_II atoms at lvl 10 or 11 in each layer, normalized but the total
            #number of C atoms in the ejecta. 
            frac_C = (lvl_10 + lvl_11) / total_ions
            self.ax.plot(v_inner, frac_C, color=cmap(Norm(i)), ls='-', lw=3.)

        '''
        for i, C in enumerate(self.C_list):
         
            hdf = make_fpath(C)
            
            v_inner = pd.read_hdf(hdf, '/simulation/model/v_inner') / 1.e5
            lvl_7 = pd.read_hdf(
              hdf, '/simulation/plasma/level_number_density').loc[14,1,7]
            
            total_ions = 0.
            for k in range(5):
                total_ions += sum(pd.read_hdf(
                  hdf, '/simulation/plasma/ion_number_density').loc[14,k])
            
            #C_II atoms at lvl 10 or 11 in each layer, normalized but the total
            #number of C atoms in the ejecta. 
            frac_Si = lvl_7 / total_ions
            self.ax.plot(v_inner, frac_Si, color=cmap(Norm(i)), ls='-', lw=3.)

        #Add colorbar
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=Norm)
        aux_mappable.set_array([])
        cbar = plt.colorbar(aux_mappable)
        cbar.set_ticks(np.arange(0, 7.01, 1.))
        cbar.ax.tick_params(width=1, labelsize=self.fs)
        cbar.set_label('C scale factor', fontsize=self.fs)     
        cbar.set_ticklabels(self.C_list)

    def add_L_text(self):
        frac = str(format(10. ** float(self.L) / texp2L[self.t_exp], '.1f'))
        text = r'$L\ =\ ' + frac + '\\times L_{\\mathrm{11fe}}$'
        self.ax.text(14000., 4.e-9, text, fontsize=20., horizontalalignment='left')  
                          
    def save_figure(self):
        top_dir = './../OUTPUT_FILES/FIGURES/C_scaled_trough/'
        if not os.path.isdir(top_dir):
            os.mkdir(top_dir)        
        
        outdir = top_dir + 't_exp-' + self.t_exp + '/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
                      
        if self.save_fig:
            fpath = outdir + 'Fig_11fe_C_scaled_trough_L-' + self.L + '.png'
            plt.savefig(fpath, format='png', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_and_plot_synthetic_spectrum()
        self.add_L_text()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()

if __name__ == '__main__':
    for t in ['12.1']:
        L_list = [lum2loglum(float(texp2L[t]) * l) for l in L_scal]
        #L_list = ['9.362']
        for L, l in zip(L_list, L_scal):
            print 'Plotting t_exp = ' + t + ', L = ' + str(l) + 'L_11fe'
            #Plot_C_scaled_Spectra(t_exp=t, L=L, show_fig=False, save_fig=True)
            Plot_C_scaled_Trough(t_exp=t, L=L, show_fig=True, save_fig=False)
        

          
          
