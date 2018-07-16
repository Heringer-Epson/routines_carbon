#!/usr/bin/env python

import os                                                               

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cPickle
from matplotlib.ticker import MultipleLocator
from astropy import units as u

from mpl_toolkits.axes_grid.inset_locator import InsetPosition

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

date_list = np.array(['5.9', '9.0', '12.1', '16.1', '19.1'])
Nd = len(date_list)
fs = 20.

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

texp2date = {'3.7': '2011_08_25', '5.9': '2011_08_28', '9.0': '2011_08_31',
             '12.1': '2011_09_03', '16.1': '2011_09_07', '19.1': '2011_09_10',
             '22.4': '2011_09_13', '28.3': '2011_09_19'}

texp2v = {'3.7': '13300', '5.9': '12400', '9.0': '11300',
          '12.1': '10700', '16.1': '9000', '19.1': '7850',
          '22.4': '6700', '28.3': '4550'}

texp2L = {'3.7': 0.08e9, '5.9': 0.32e9, '9.0': 1.1e9,
          '12.1': 2.3e9, '16.1': 3.2e9, '19.1': 3.5e9,
          '22.4': 3.2e9, '28.3': 2.3e9}

texp2logL = {'3.7': '7.903', '5.9': '8.505', '9.0': '9.041',
          '12.1': '9.362', '16.1': '9.505', '19.1': '9.544',
          '22.4': '9.505', '28.3': '9.362'}

class Plot_Best(object):
    """
    Description:
    ------------
    Makes Fig. 7 in the carbon paper. This code plots the default (best) 11fe
    models at multiple epochs. Plotted spectra are computed under the
    'Downbranch' and 'Macroatom' line interaction modes.
    
    It also has an option to include spectra
    computed for the W7 and double detonation models for comparison.       

    Parameters:
    -----------
    add_models : ~bool
        If True, then include W7 and double detonation spectra. These are
        compute with C abundances that were designed to be an approximation
        of those derived from actual simulation. In other words, the C profiles
        are not interpolated from real models.
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_best.pdf
    ./../OUTPUT_FILES/FIGURES/Fig_best_with-models.pdf
    """
    
    def __init__(self, add_models=False, show_fig=True, save_fig=False):

        self.add_models = add_models
        self.show_fig = show_fig
        self.save_fig = save_fig   
            
        self.F = {}
        self.fig = plt.figure(figsize=(8,14))        
        for j in range(Nd):
            idx = str(j)
            self.F['ax' + idx] = plt.subplot(Nd, 1, j + 1)        
     
        xloc = 0.70
        dx = 0.19
        dy = 0.08
        
        self.F['axi_o0'] = self.fig.add_axes([xloc, 0.794, dx, dy])
        self.F['axi_o1'] = self.fig.add_axes([xloc, 0.640, dx, dy])
        self.F['axi_o2'] = self.fig.add_axes([xloc, 0.485, dx, dy])
        self.F['axi_o3'] = self.fig.add_axes([xloc, 0.330, dx, dy])
        self.F['axi_o4'] = self.fig.add_axes([xloc, 0.175, dx, dy])

        plt.subplots_adjust(hspace=0.03)

        self.run_make_best()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$\lambda \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{log \ f}_{\lambda}\ \mathrm{[arb.\ units]}$'

        x_cbar_label = r'$t_{\mathrm{exp}}\ \mathrm{[days]}$'
        y_cbar_label = (r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                      + r'$ \ \lambda$6580')       

        self.F['ax2'].set_ylabel(y_label, fontsize=fs, labelpad=8)

        for j in range(Nd):
            idx = str(j)        
        
            self.F['ax' + idx].set_yscale('log')
            self.F['ax' + idx].set_xlim(3000., 12000.)
            self.F['ax' + idx].set_ylim(0.05, 10.)      
            self.F['ax' + idx].tick_params(
              axis='y', which='major', labelsize=fs, pad=8)      
            self.F['ax' + idx].tick_params(
              axis='x', which='major', labelsize=fs, pad=8)
            self.F['ax' + idx].minorticks_on()
            self.F['ax' + idx].tick_params(
              'both', length=8, width=1, which='major', direction='in')
            self.F['ax' + idx].tick_params(
              'both', length=4, width=1, which='minor', direction='in')
            self.F['ax' + idx].xaxis.set_minor_locator(MultipleLocator(500.))
            self.F['ax' + idx].xaxis.set_major_locator(MultipleLocator(2000.))         
            self.F['ax' + idx].tick_params(labelleft='off')          
            if idx != '4':
                self.F['ax' + idx].tick_params(labelbottom='off')          
            if idx == '4':
                self.F['ax' + idx].set_xlabel(x_label, fontsize=fs)    

    def loop_dates(self):
        for j, date in enumerate(date_list):
            idx = str(j)        
            Add_Curves(self.F['ax' + idx], self.F['axi_o' + idx],
                       date, j, self.add_models).run_add_curves()  
            
    def save_figure(self):
        fname = 'Fig_best'    
        if self.add_models:
            fname += '_with-models'        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + fname + '.pdf',
                        format='pdf', dpi=360, bbox_inches='tight')
    
    def run_make_best(self):
        self.set_fig_frame()
        self.loop_dates()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()

class Add_Curves(object):
    
    def __init__(self, ax, axi_o, t_exp, idx=1, add_models=False):
        
        self.t_exp = t_exp        
        self.ax = ax
        self.axi_o = axi_o
        self.idx = idx
        self.add_models = add_models
        self.t = str(int(round(float(self.t_exp))))
        self.panel = ['a', 'b', 'c', 'd', 'e']

        self.pkl_list = []
        self.D = {}
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""

        x_label_o = r'$\lambda \ \mathrm{[\AA}]}$'
        x_label_n = r'$\lambda \ \mathrm{[\mu m}]}$'

        self.axi_o.set_xlim(6200., 6550.)
        self.axi_o.set_ylim(0.70, 1.3)
        self.axi_o.tick_params(axis='y', which='major', labelsize=fs - 4., pad=8)      
        self.axi_o.tick_params(axis='x', which='major', labelsize=fs - 4., pad=8)
        self.axi_o.minorticks_on()
        self.axi_o.tick_params(
          'both', length=8, width=1, which='major', pad=2, direction='in')
        self.axi_o.tick_params(
          'both', length=4, width=1, which='minor', pad=2, direction='in')
        self.axi_o.xaxis.set_minor_locator(MultipleLocator(50.))
        self.axi_o.xaxis.set_major_locator(MultipleLocator(150.))
        self.axi_o.yaxis.set_minor_locator(MultipleLocator(0.05))
        self.axi_o.yaxis.set_major_locator(MultipleLocator(0.2)) 
        self.axi_o.tick_params(labelleft='off')          

    def add_texp_text(self):            
        self.ax.text(3300., 0.08, r'$t_{\rm{exp}}=' + self.t_exp + '\\ \\rm{d}$',
                     fontsize=20., horizontalalignment='left', color='k')
        self.ax.text(3300., 0.2, r'$\mathbf{' + self.panel[self.idx] + '}$',
                     fontsize=20., horizontalalignment='left', color='k')

    def load_and_plot_observational_spectrum(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        with open(directory + texp2date[self.t_exp] + '.pkl', 'r') as inp:
            pkl = cPickle.load(inp)
            flux_raw = pkl['flux_raw'] / pkl['norm_factor']
            self.ax.plot(pkl['wavelength_corr'], flux_raw,
                         color='k', ls='-', lw=3., zorder=2.,
                         label=r'$\mathrm{SN\ 2011fe}$')
            self.axi_o.plot(pkl['wavelength_corr'], flux_raw,
                         color='k', ls='-', lw=3., zorder=2.)            
            print self.t_exp, pkl['velocity_f7']

        #Plot Si feature label.
        x = pkl['wavelength_minima_f7']
        y = pkl['flux_minima_f7']
        self.ax.plot([x, x], [y + 0.4, y + 0.8], ls='-', marker='None',
                         color='grey', lw=2.)

        self.ax.text(x, y + 1., r'Si', fontsize=20.,
                         horizontalalignment='center', color='grey')

        #Plot C feature label.
        x = pkl['wavelength_minima_f7'] + 220.
        y = pkl['flux_minima_f7']            
        
        self.ax.plot([x, x], [y + 0.7, y + 2.2], ls='-', marker='None',
                         color='grey', lw=2.)

        self.ax.text(x, y + 2.4, r'C', fontsize=20.,
                     horizontalalignment='center', color='grey')
        
        #Get measured pEW.
        if np.isnan(pkl['pEW_fC']):
            pkl['pEW_fC'], pkl['pEW_unc_fC'] = 0., 0.
        
        self.D['pEW_obs'] = pkl['pEW_fC']
        self.D['unc_obs'] = pkl['pEW_unc_fC']        

    def load_and_plot_synthetic_spectrum(self):
        
        def make_fpath(LI, XCi, XCo):
            fname = ('line_interaction-' + LI + '_excitation-dilute-lte_C-F2-'
            + XCo + '_C-F1-' + XCi)
            return (path_tardis_output + '11fe_' + self.t
                    + 'd_C-best/' + fname + '/' + fname + '.pkl')
            
        #colors = ['#d95f02', '#7570b3', '#1b9e77']
        colors = ['#7570b3']
        ls = ['-', ':', '--']
        labels = [r'${\tt Downbranch}$', r'${\tt Macroatom}$']
        
        for j, LI in enumerate(['downbranch', 'macroatom']):
            for i, (XCi, XCo) in enumerate(zip(['0.2'], ['1.00'])):

                with open(make_fpath(LI, XCi, XCo), 'r') as inp:
                    pkl = cPickle.load(inp)
                    
                    flux_raw = pkl['flux_raw'] / pkl['norm_factor']      
                    self.ax.plot(pkl['wavelength_corr'], flux_raw,
                                 color=colors[i], ls=ls[j], lw=2., zorder=1.,
                                 label=labels[j])

                    self.axi_o.plot(pkl['wavelength_corr'], flux_raw,
                                       color=colors[i], ls=ls[j], lw=2.,
                                       zorder=1.)
                    
    def load_and_plot_proxy_models(self):
        
        def make_fpath(model):
            fname = ('velocity_start-' + texp2v[self.t_exp] +  '_loglum-'
                     + texp2logL[self.t_exp] + '_time_explosion-' + self.t_exp)
            return (path_tardis_output + '11fe_' + model + '_C-prof/'
                    + fname + '/' + fname + '.pkl')
            
        colors = ['#d95f02', '#1b9e77']
        ls = [':', '--']
        labels = ['W7', 'Double Det.']

        for i, mdl in enumerate(['W7', 'ddet']):
            with open(make_fpath(mdl), 'r') as inp:
                pkl = cPickle.load(inp)
                
                flux_raw = pkl['flux_raw'] / pkl['norm_factor']      
                self.ax.plot(pkl['wavelength_corr'], flux_raw,
                             color=colors[i], ls=ls[i], lw=2., zorder=1.,
                             label=labels[i])

                self.axi_o.plot(pkl['wavelength_corr'], flux_raw,
                                   color=colors[i], ls=ls[i], lw=2., zorder=1.)

    def add_legend(self):
        if self.idx == 0:
            if self.add_models:
                self.ax.legend(
                  frameon=False, fontsize=fs - 4, numpoints=1, ncol=1,
                  loc=10, bbox_to_anchor=(0.56,0.77), labelspacing=-0.1,
                  handlelength=1.5, handletextpad=0.2)  
            else:
                self.ax.legend(
                  frameon=False, fontsize=fs, numpoints=1, ncol=1,
                  loc=10, bbox_to_anchor=(0.56,0.77), labelspacing=-0.1,
                  handlelength=1.5, handletextpad=0.2)                 
                 
    def run_add_curves(self):
        self.set_fig_frame()
        self.add_texp_text()
        self.load_and_plot_observational_spectrum()
        self.load_and_plot_synthetic_spectrum()
        if self.add_models:
            self.load_and_plot_proxy_models()
        self.add_legend()
                
if __name__ == '__main__':
    Plot_Best(add_models=True, show_fig=False, save_fig=True)

    
