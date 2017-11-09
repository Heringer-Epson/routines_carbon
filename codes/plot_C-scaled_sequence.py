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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
                                                
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
    """
    """
    
    def __init__(self, t_exp='12.1', l=1.0, show_fig=True, save_fig=False):

        self.t_exp = t_exp
        self.l = l
        self.L = lum2loglum(texp2L[self.t_exp] * self.l)
        self.show_fig = show_fig
        self.save_fig = save_fig 
        self.fig, self.ax = plt.subplots(figsize=(14,8))
        self.fs = 26.
        
        self.C_list = ['0.00', '0.01', '0.02', '0.05', '0.10', '0.20',
                       '0.50', '1.00', '2.00', '5.00', '10.00', '20.00']
        self.t = str(int(float(self.t_exp)))
        self.pkl_list = []               

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

        #Add colorbar
        cmap = cmaps.viridis
        Norm = colors.Normalize(vmin=0., vmax=len(self.C_list) - 1.)
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=Norm)
        aux_mappable.set_array([])
        cbar = plt.colorbar(aux_mappable)
        cbar.set_ticks(np.arange(0, len(self.C_list), 1.))
        cbar.ax.tick_params(width=1, labelsize=self.fs)
        cbar.set_label('C scale factor', fontsize=self.fs)     
        cbar.set_ticklabels(self.C_list)

        #Inset axis.
        if l >= 0.7:
            self.axins = self.fig.add_axes([0.50, 0.55, 0.25, 0.35])
            self.axins.set_xlim(6150., 6550.)
            self.axins.set_ylim(0.4, 1.15)
            self.axins.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)      
            self.axins.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
            self.axins.minorticks_on()
            self.axins.tick_params('both', length=8, width=1, which='major')
            self.axins.tick_params('both', length=4, width=1, which='minor')
            self.axins.xaxis.set_minor_locator(MultipleLocator(50.))
            self.axins.xaxis.set_major_locator(MultipleLocator(200.))
            self.axins.yaxis.set_minor_locator(MultipleLocator(0.05))
            self.axins.yaxis.set_major_locator(MultipleLocator(0.2))    

    def load_and_plot_observational_spectrum(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        with open(directory + texp2date[self.t_exp] + '.pkl', 'r') as inp:
            pkl = cPickle.load(inp)
            self.ax.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                         color='k', ls='-', lw=3., zorder=2.)
            self.ax.plot(pkl['wavelength_corr'], pkl['flux_raw'] / pkl['norm_factor'],
                         color='grey', ls='-', lw=3., alpha=0.5, zorder=1.)
            if l >= 0.7:
                self.axins.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                                color='k', ls='-', lw=3., zorder=2.)
                self.axins.plot(pkl['wavelength_corr'], pkl['flux_raw'] / pkl['norm_factor'],
                             color='grey', ls='-', lw=3., alpha=0.5, zorder=1.)
            #print 'pEW obs=', pkl['pEW_fC']
            #print 'v obs=', pkl['velocity_fC']
                
    def load_and_plot_synthetic_spectrum(self):

        cmap = cmaps.viridis
        Norm = colors.Normalize(vmin=0., vmax=len(self.C_list) - 1.)
        
        def make_fpath(C):
            fname = 'loglum-' + self.L + '_C-F1-' + C 
            return (path_tardis_output + '11fe_' + self.t
                    + 'd_C-scaled/' + fname + '/' + fname + '.pkl')
                    
        for i, C in enumerate(self.C_list):
            with open(make_fpath(C), 'r') as inp:
                pkl = cPickle.load(inp)
                self.pkl_list.append(pkl)
                self.ax.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                             color=cmap(Norm(i)), ls='-', lw=2., zorder=2.)
                #Inset plot
                if l >= 0.7:
                    self.axins.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                                    color=cmap(Norm(i)), ls='-', lw=2., zorder=2.)

    def add_L_text(self):
        frac = str(format(10. ** float(self.L) / texp2L[self.t_exp], '.1f'))
        text = r'$L\ =\ ' + frac + '\\times L_{\\mathrm{11fe}}$'
        self.ax.text(1800., 3.2, text, fontsize=20., horizontalalignment='left')        

    def compute_diff_C_pEW(self):

        def get_pEW(wavelength_region, flux_region, pseudo_flux):           
            pEW = sum(np.multiply(
              np.diff(wavelength_region),
              np.divide(pseudo_flux[0:-1] - flux_region[0:-1], pseudo_flux[0:-1])))
            return pEW

        w_min = self.pkl_list[-1]['wavelength_maxima_blue_fC']        
        w_max = self.pkl_list[-1]['wavelength_maxima_red_fC']        
        wav = f_c = self.pkl_list[0]['wavelength_corr']
        window = ((wav >= w_min) & (wav <= w_max))
        
        f_c = self.pkl_list[0]['flux_smoothed'][window]
        w = wav[window]
        
        for k, pkl in enumerate(self.pkl_list):
            f = pkl['flux_smoothed'][window]
            pEW = get_pEW(w, f, f_c)
            print ('C factor=', self.C_list[k], 'pEW_measured=',
                  pkl['pEW_fC'], 'vel_measured=', pkl['velocity_fC'], 'pEW new', pEW)

    def test_fixed_boundary_pEW(self):

        def get_pEW(wavelength_region, flux_region, pseudo_flux):           
            pEW = sum(np.multiply(
              np.diff(wavelength_region),
              np.divide(pseudo_flux[0:-1] - flux_region[0:-1], pseudo_flux[0:-1])))
            return pEW

        w_min = 6100.        
        w_max = 6500.        
        wav = f_c = self.pkl_list[0]['wavelength_corr']
        window = ((wav >= w_min) & (wav <= w_max))
        
        f_c = self.pkl_list[0]['flux_smoothed'][window]
        w = wav[window]
        
        for k, pkl in enumerate(self.pkl_list):
            f = pkl['flux_smoothed'][window]
            pEW = get_pEW(w, f, f_c)
            print ('C factor=', self.C_list[k], 'pEW_measured=', pkl['pEW_fC'], 'pEW new', pEW)
    
    
    def save_figure(self):
        top_dir = './../OUTPUT_FILES/FIGURES/C_scaled_spectra/'
        if not os.path.isdir(top_dir):
            os.mkdir(top_dir)        
        
        outdir = top_dir + 't_exp-' + self.t_exp + '/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
                      
        if self.save_fig:
            fpath = (outdir + 'Fig_11fe_C_scaled_spectra_texp-' + self.t_exp
                     + '_L-' + str(self.l) + '.png')
            plt.savefig(fpath, format='png', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_and_plot_observational_spectrum()
        self.load_and_plot_synthetic_spectrum()
        self.add_L_text()
        self.compute_diff_C_pEW()
        self.test_fixed_boundary_pEW()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()


class Plot_C_scaled_Trough(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, t_exp='12.1', l=1.0, show_fig=True, save_fig=False):

        self.t_exp = t_exp
        self.l = l
        self.L = lum2loglum(texp2L[self.t_exp] * self.l)
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
        self.ax.set_xlim(10000., 20000.)
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
            fpath = (outdir + 'Fig_11fe_C_scaled_trough_texp-' + self.t_exp
                     + '_L-' + str(self.l) + '.png')
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
    #for t in ['12.1', '19.1']:
    for t in ['19.1']:
        #for l in L_scal:
        for l in [1.6]:
            print 'Plotting t_exp = ' + t + ', L = ' + str(l) + 'L_11fe'
            Plot_C_scaled_Spectra(t_exp=t, l=l, show_fig=True, save_fig=False)
            #Plot_C_scaled_Trough(t_exp=t, l=l, show_fig=True, save_fig=False)
        

          
          
