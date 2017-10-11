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

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

class Spectra_Operation(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, lm, show_fig=True, save_fig=False):

        self.lm = lm
        self.show_fig = show_fig
        self.save_fig = save_fig

        L_scal = np.arange(0.6, 2.01, 0.1)
        self.L = [lum2loglum(0.32e9 * l) for l in L_scal]                
        
        self.Fe = ['0.0', '0.02', '0.04', '0.1', '0.2', '0.4', '1.0', '2.0',
                   '4.0', '10.0']
            
        #For testing
        #L_scal = [2.0]
        #self.L = [lum2loglum(0.32e9 * l) for l in L_scal]                
        #self.Fe = ['4.0']
                   
        self.fig = None
        self.ax_bot = None  
        self.ax_top = None  

        self.fsize = 26
        
        self.C_min = 6200.
        self.C_max = 6500.
   
        self.loop_L_and_Fe()
        
    def set_fig_frame(self):

        plt.close()
        self.fig = plt.figure(figsize=(10,10))
        self.ax_bot = plt.subplot(212)  
        self.ax_top = plt.subplot(211, sharex=self.ax_bot) 
        
        x_label_bot = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label_bot = r'$f_{\rm{C}} \ / \ f_{\rm{no \ C}}$'
        y_label_top = r'$\mathrm{Relative \ f}_{\lambda}$'
        
        self.ax_bot.set_xlabel(x_label_bot,fontsize=self.fsize)
        self.ax_bot.set_ylabel(y_label_bot,fontsize=self.fsize)
        self.ax_bot.set_xlim(1500.,10000.)
        self.ax_bot.set_ylim(0.,2.)      
        self.ax_bot.tick_params(axis='y', which='major', labelsize=self.fsize, pad=8)      
        self.ax_bot.tick_params(axis='x', which='major', labelsize=self.fsize, pad=8)
        self.ax_bot.minorticks_on()
        self.ax_bot.tick_params('both', length=8, width=1, which='major')
        self.ax_bot.tick_params('both', length=4, width=1, which='minor')
        self.ax_bot.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax_bot.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax_bot.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax_bot.yaxis.set_major_locator(MultipleLocator(1.))        

        self.ax_top.set_ylabel(y_label_top,fontsize=self.fsize)
        self.ax_top.set_ylim(-1.,5.)      
        self.ax_top.tick_params(axis='y', which='major', labelsize=self.fsize, pad=8)      
        self.ax_top.minorticks_on()
        self.ax_top.tick_params('both', length=8, width=1, which='major')
        self.ax_top.tick_params('both', length=4, width=1, which='minor')
        self.ax_top.yaxis.set_minor_locator(MultipleLocator(1.))
        self.ax_top.yaxis.set_major_locator(MultipleLocator(2.))        
        self.ax_top.tick_params(labelleft='off')
        plt.setp(self.ax_top.get_xticklabels(), visible=False)
        
        self.fig.subplots_adjust(hspace=0.1)
        
        self.ax_bot.axhline(y=1., color='k', ls='--', lw=2., zorder=1)  
        self.ax_bot.axvline(x=self.C_min, color='k', ls=':', lw=2., zorder=1)  
        self.ax_bot.axvline(x=self.C_max, color='k', ls=':', lw=2., zorder=1)  

    def load_data(self, lm, L, Fe, folder_appendix):
        
        if folder_appendix == 'C':
            case_folder = path_tardis_output + 'early-carbon-grid/'
        elif folder_appendix == 'no-C':
            case_folder = path_tardis_output + 'early-carbon-grid_no-C/'
                
        fname = 'loglum-' + L + '_line_interaction-' + lm + '_Fes-' + Fe
        bot_dir = case_folder + fname + '/'
        fullpath = bot_dir + fname + '.pkl'                  
        
        with open(fullpath, 'r') as inp:
            pkl = cPickle.load(inp)
            
        return pkl, bot_dir, fname

    def make_plot(self, w_C, f_C, w_noC, f_noC, f_div, fig_fullpath, fig_title):
        self.set_fig_frame()
        
        self.ax_top.plot(w_C, f_C, ls='-', lw=3., color='b',
                         label='Ejecta contains carbon') 
        self.ax_top.plot(w_noC, f_noC, ls='-', lw=3., color='g',
                         label='Carbon-free ejecta') 
        self.ax_bot.plot(w_C, f_div, ls='-', lw=3., color='k',
                         label='C / no-C') 
                     
        self.ax_top.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=1)           

        plt.title(fig_title, fontsize=self.fsize)       
    
        if self.save_fig:
            plt.savefig(fig_fullpath, format='png', dpi=360)
        if self.show_fig:
            plt.show()    
                                
    def loop_L_and_Fe(self):

        def get_window(w):
            return ((w >= self.C_min) & (w <= self.C_max))

        def compute_pEW(w, f):
            """This simplified form is because the continuum, by definition is 
            equal 1."""
            return w[-1] - w[0] - sum(np.multiply(np.diff(w), f[0:-1]))
            
        def divide_spectra(f_C, f_noC):
            return f_C / f_noC
        
        def compute_depth(f):
            return 1. - min(f)    
                
        for i, L in enumerate(self.L):
            for Fe in self.Fe:
                print '...' + str(i + 1) + '/' + str(len(self.L))

                pkl_C, bot_dir, fname = self.load_data(self.lm, L, Fe, 'C')
                pkl_noC, trash1, trash2 = self.load_data(self.lm, L, Fe, 'no-C')
                
                w_C, f_C = pkl_C['wavelength_corr'], pkl_C['flux_smoothed']
                w_noC, f_noC = pkl_noC['wavelength_corr'], pkl_noC['flux_smoothed']
                
                f_div = divide_spectra(f_C, f_noC)
        
                C_window = get_window(w_C)
                w_window, f_window = w_C[C_window], f_div[C_window]
                
                #Create update pkl output.
                new_pkl_fullpath = bot_dir + fname + '_up.pkl'
                pkl_C['diff_pEW-fC'] = compute_pEW(w_window, f_window)
                pkl_C['diff_depth-fC'] = compute_depth(f_window)

                with open(new_pkl_fullpath, 'w') as out_pkl:
                    cPickle.dump(pkl_C, out_pkl, protocol=cPickle.HIGHEST_PROTOCOL)                 
                    
                #Make figure.
                fig_fullpath = bot_dir + fname + '_C-div.png'
                fig_title = r'Compare: 1% to 0% carbon at logL=' + L + ', Fe x' + Fe
                self.make_plot(w_C, f_C, w_noC, f_noC, f_div, fig_fullpath, fig_title)
        
#Spectra_Operation(lm='macroatom', L='8.283', Fe='4.0', show_fig=False, save_fig=False)
#Spectra_Operation(lm='macroatom', L='8.651', Fe='4.0', show_fig=False, save_fig=False)
#Spectra_Operation(lm='macroatom', L='8.806', Fe='4.0', show_fig=False, save_fig=False)
#Spectra_Operation(lm='macroatom', L='8.806', Fe='0.0', show_fig=False, save_fig=False)
Spectra_Operation(lm='macroatom', show_fig=False, save_fig=True)




