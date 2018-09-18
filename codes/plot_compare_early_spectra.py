#!/usr/bin/env python

import os                                                               
path_tardis_output = os.environ['path_tardis_output']

import cPickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 26.

class Compare_Early(object):
    """
    Description:
    ------------
    Makes a figure where the observed and synthetic spectra of 11fe are plotted
    for t_exp = 3.7 days. This is a side figure that helps to illustrate
    the differences between the observed spectra and the simulated one, which
    are particularly obvious in the region around 3000 angs.       
    """
    
    def __init__(self, show_fig=True, save_fig=False):
        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
 
        self.run_comparison()
        
    def set_fig_frame(self):
        
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'
        
        self.ax.set_xlabel(x_label,fontsize=fs)
        self.ax.set_ylabel(y_label,fontsize=fs)
        self.ax.set_xlim(1500.,10000.)
        self.ax.set_ylim(-.5,2.5)      
        self.ax.tick_params(
          axis='y', which='major', labelsize=fs, pad=8)      
        self.ax.tick_params(
          axis='x', which='major', labelsize=fs, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1, which='minor', direction='in')
        self.ax.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(2.))
        self.ax.xaxis.set_ticks_position('both')        
        self.ax.yaxis.set_ticks_position('both')        
        self.ax.tick_params(labelleft='off')
        plt.title(r'$t_{\mathrm{exp}=\mathrm{3.7\ d}}$', fontsize=fs)                

    def plot_observed_spectrum(self):          
        path_data = './../INPUT_FILES/observational_spectra/'
        with open(path_data + '2011fe/2011_08_25.pkl', 'r') as inp:
            pkl = cPickle.load(inp)                                        
            self.ax.plot(pkl['wavelength_corr'], pkl['flux_normalized'],
                         color='k', lw=2., label=r'SN 2011fe : obs')
                             
    def plot_synthetic_spectrum(self):
        inp_dir = path_tardis_output + '11fe_4d_C-best/'
        fname = 'line_interaction-downbranch_C-F2-1.00_C-F1-0.2'
        with open(inp_dir + fname + '/' + fname + '.pkl', 'r') as inp:
            pkl = cPickle.load(inp)              
            self.ax.plot(pkl['wavelength_corr'], pkl['flux_normalized'],
                         color='b', lw=2., label=r'SN 2011fe : syn')                    

        self.ax.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1, loc=1) 

    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_compare_early_spectra.pdf',
                        format='pdf', dpi=dpi)
        
    def run_comparison(self):
        self.set_fig_frame()
        self.plot_observed_spectrum()
        self.plot_synthetic_spectrum()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
                    
if __name__ == '__main__':
    Compare_Early(show_fig=True, save_fig=False)
