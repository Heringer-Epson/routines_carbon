#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import matplotlib.pyplot as plt
import matplotlib as mpl
import cPickle
import new_colormaps as cmaps
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from matplotlib import cm
from matplotlib import colors

from mpl_toolkits.axes_grid.inset_locator import InsetPosition

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

C_list = ['0.00', '0.10', '0.20', '0.50', '1.00', '2.00', '5.00', '10.00',
          '20.00']   
cmap = plt.cm.get_cmap('winter')
Norm = colors.Normalize(vmin=0., vmax=len(C_list) - 1.)
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

class Add_Curves(object):
    
    def __init__(self, ax, axins, ax_bot, t_exp='16.1', idx=1):
        
        self.t_exp = t_exp        
        self.ax = ax
        self.axins = axins
        self.ax_bot = ax_bot
        self.idx = idx
        self.t = str(int(float(self.t_exp)))
        self.panel = ['a', 'b', 'c', 'd']

        self.pkl_list = []
        self.D = {}
        self.D['C'], self.D['pEW'], self.D['unc'] = [], [], []
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        self.axins.set_xlim(6200., 6550.)
        self.axins.set_ylim(0.55, 1.05)
        self.axins.tick_params(axis='y', which='major', labelsize=fs - 4., pad=8)      
        self.axins.tick_params(axis='x', which='major', labelsize=fs - 4., pad=8)
        self.axins.minorticks_on()
        self.axins.tick_params('both', length=8, width=1, which='major')
        self.axins.tick_params('both', length=4, width=1, which='minor')
        self.axins.xaxis.set_minor_locator(MultipleLocator(50.))
        self.axins.xaxis.set_major_locator(MultipleLocator(100.))
        self.axins.yaxis.set_minor_locator(MultipleLocator(0.05))
        self.axins.yaxis.set_major_locator(MultipleLocator(0.2)) 
        self.axins.tick_params(labelleft='off')          
        
    def add_texp_text(self):
        self.ax.text(1750., 2.90, r'$t_{\rm{exp}}=' + self.t_exp + '\\ \\rm{d}$',
                     fontsize=20., horizontalalignment='left', color='k')
        self.ax.text(1750., 2.35, r'$\mathbf{' + self.panel[self.idx] + '}$',
                     fontsize=20., horizontalalignment='left', color='k')

    def load_and_plot_observational_spectrum(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        with open(directory + texp2date[self.t_exp] + '.pkl', 'r') as inp:
            pkl = cPickle.load(inp)
            self.ax.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                         color='k', ls='-', lw=3., zorder=2.,
                         label=r'$\mathrm{SN\ 2011fe}$')
            self.axins.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                         color='k', ls='-', lw=3., zorder=2.)            

        #Plot observed spectra at top plot.
        if self.idx == 0:
            self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                           handletextpad=0.5, loc=9) 

        #Plot Si feature label.
        x = pkl['wavelength_minima_f7']
        y = pkl['flux_minima_f7']
        self.ax.plot([x, x], [y + 0.4, y + 0.8], ls='-', marker='None',
                         color='grey', lw=2.)

        self.ax.text(x, y + 1., r'Si', fontsize=20.,
                         horizontalalignment='center', color='grey')

        #Plot C feature label.
        x = pkl['wavelength_minima_f7'] + 220.
        
        self.ax.plot([x, x], [y + 0.7, y + 1.2], ls='-', marker='None',
                         color='grey', lw=2.)

        self.ax.text(x, y + 1.4, r'C', fontsize=20.,
                     horizontalalignment='center', color='grey')

        #Plot C feature label.
        x = pkl['wavelength_minima_f7'] + 855.
        
        self.ax.plot([x, x], [y + 0.7, y + 1.2], ls='-', marker='None',
                         color='grey', lw=2.)

        self.ax.text(x, y + 1.4, r'C', fontsize=20.,
                     horizontalalignment='center', color='grey')
        
        #Get measured pEW.
        if np.isnan(pkl['pEW_fC']):
            pkl['pEW_fC'], pkl['pEW_unc_fC'] = 0.5, 0.5
        
        self.D['pEW_obs'] = pkl['pEW_fC']
        self.D['unc_obs'] = pkl['pEW_unc_fC']        
        
    def load_and_plot_synthetic_spectrum(self):
        
        def make_fpath(C):
            lum = lum2loglum(texp2L[self.t_exp])
            fname = 'loglum-' + lum + '_C-F1-' + C
            return (path_tardis_output + '11fe_' + self.t
                    + 'd_C-scaled_grid/' + fname + '/' + fname + '.pkl')
                    
        for i, C in enumerate(C_list):
            try:
                with open(make_fpath(C), 'r') as inp:
                    pkl = cPickle.load(inp)
                    self.pkl_list.append(pkl)
                    self.ax.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                                 color=cmap(Norm(i)), ls='-', lw=2.,
                                 zorder=1.)
                    self.axins.plot(pkl['wavelength_corr'], pkl['flux_smoothed'],
                                       color=cmap(Norm(i)), ls='-', lw=2.,
                                       zorder=1.)
                    pEW = pkl['pEW_fC']
                    if np.isnan(pEW):
                        pEW = 0.5
                    self.D['C'].append(C)
                    self.D['pEW'].append(pEW)
                    self.D['unc'].append(pkl['pEW_unc_fC'])                   
            except:
                self.D['C'].append(C)
                self.D['pEW'].append(np.nan)
                self.D['unc'].append(np.nan)
                 
    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_11fe_C-scan_' + self.t + 'd.pdf',
                        format='pdf', dpi=360)
    
    def run_make_slab(self):
        self.set_fig_frame()
        self.add_texp_text()
        self.load_and_plot_observational_spectrum()
        self.load_and_plot_synthetic_spectrum()
        return self.D

class Make_Scan(object):
    
    def __init__(self, show_fig=True, save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig         
             
        self.master = {}

        self.fig = plt.figure(figsize=(8, 16))

        self.ax_1 = plt.subplot2grid((200, 20), (0, 0), rowspan=36, colspan=20)
        self.ax_2 = plt.subplot2grid((200, 20), (37, 0), rowspan=36, colspan=20,
                                     sharex=self.ax_1, sharey=self.ax_1)
        self.ax_3 = plt.subplot2grid((200, 20), (74, 0), rowspan=36, colspan=20,
                                     sharex=self.ax_1, sharey=self.ax_1)
        self.ax_4 = plt.subplot2grid((200, 20), (111, 0), rowspan=36, colspan=20,
                                     sharex=self.ax_1, sharey=self.ax_1)
        self.ax_5 = plt.subplot2grid((200, 20), (164, 0), rowspan=36, colspan=16)
        
        self.ax_cbar = plt.subplot2grid((200, 20), (164, 16), rowspan=36, colspan=1)
        
        N = len(C_list)
        color_disc = cmap(np.linspace(0, 1, N))
        cmap_D = cmap.from_list('irrel', color_disc, N)
        
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap_D, norm=Norm)
        aux_mappable.set_array([])
        aux_mappable.set_clim(vmin=-0.5, vmax=N-0.5)
        cbar = plt.colorbar(aux_mappable, cax=self.ax_cbar)
        cbar.set_ticks(range(N))
        cbar.ax.tick_params(width=1, labelsize=fs)
        cbar.set_label(r'Scaling factor', fontsize=fs)     
        cbar.set_ticklabels(C_list)

        self.ax_ins_1 = self.fig.add_axes([0.66, 0.805, 0.22, 0.08])
        self.ax_ins_2 = self.fig.add_axes([0.66, 0.658, 0.22, 0.08])
        self.ax_ins_3 = self.fig.add_axes([0.66, 0.510, 0.22, 0.08])
        self.ax_ins_4 = self.fig.add_axes([0.66, 0.360, 0.22, 0.08])
        
        self.master['9.0'] = Add_Curves(self.ax_1, self.ax_ins_1, self.ax_5,
                                        '9.0', 0).run_make_slab()      
        self.master['12.1'] = Add_Curves(self.ax_2, self.ax_ins_2, self.ax_5,
                                         '12.1', 1).run_make_slab()    
        self.master['16.1'] = Add_Curves(self.ax_3, self.ax_ins_3, self.ax_5,
                                         '16.1', 2).run_make_slab()      
        self.master['19.1'] = Add_Curves(self.ax_4, self.ax_ins_4, self.ax_5,
                                         '19.1', 3).run_make_slab()      
    
        self.run_make_scan()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$\lambda \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'

        x_cbar_label = r'$t_{\mathrm{exp}}\ \mathrm{[days]}$'
        y_cbar_label = (r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                      + r'$ \ \lambda$6580')       

        self.fig.text(0.04, 0.6, y_label, va='center', rotation='vertical',
                      fontsize=fs)
        
        self.ax_1.set_xlim(1500., 10000.)
        self.ax_1.set_ylim(0., 3.5)      
        self.ax_1.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax_1.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_1.minorticks_on()
        self.ax_1.tick_params('both', length=8, width=1, which='major')
        self.ax_1.tick_params('both', length=4, width=1, which='minor')
        self.ax_1.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax_1.xaxis.set_major_locator(MultipleLocator(2000.))   
        self.ax_1.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax_1.yaxis.set_major_locator(MultipleLocator(1.))        
        self.ax_1.tick_params(labelleft='off')          
        self.ax_1.tick_params(labelbottom='off')          
       
        self.ax_2.tick_params(labelleft='off')          
        self.ax_2.tick_params(labelbottom='off')    
        self.ax_2.tick_params('both', length=8, width=1, which='major')
        self.ax_2.tick_params('both', length=4, width=1, which='minor')
        
        self.ax_3.tick_params(labelleft='off')          
        self.ax_3.tick_params(labelbottom='off')  
        self.ax_3.tick_params('both', length=8, width=1, which='major')
        self.ax_3.tick_params('both', length=4, width=1, which='minor')
        
        self.ax_4.set_xlabel(x_label,fontsize=fs)
        self.ax_4.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_4.tick_params(labelleft='off')          
        self.ax_4.tick_params('both', length=8, width=1, which='major')
        self.ax_4.tick_params('both', length=4, width=1, which='minor')        
        
        self.ax_5.set_yscale('log')
        self.ax_5.set_xlabel(x_cbar_label, fontsize=fs)
        self.ax_5.set_ylabel(y_cbar_label, fontsize=fs)
        self.ax_5.set_xlim(8., 20.)
        self.ax_5.set_ylim(1., 50.)
        self.ax_5.tick_params(axis='y', which='major', labelsize=fs, pad=8)      
        self.ax_5.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_5.minorticks_on()
        self.ax_5.tick_params('both', length=8, width=1, which='major')
        self.ax_5.tick_params('y', length=4, width=1, which='minor')
        self.ax_5.xaxis.set_major_locator(MultipleLocator(1.))   
        self.ax_5.xaxis.set_minor_locator(MultipleLocator(1.))   
        #self.ax_5.yaxis.set_minor_locator(MultipleLocator(1.))
        #self.ax_5.yaxis.set_major_locator(MultipleLocator(2.))  

    def make_bottom_plot(self):
        
        t_exp_list = ['9.0', '12.1', '16.1', '19.1']
        t_exp_values = np.array([9.0, 12.1, 16.1, 19.1])
        offset = 0.03
        
        #Initialize lists to be plotted.
        self.master['pEW_obs'], self.master['unc_obs'] = [], []
        for C in C_list:
            self.master['C-' + C + '_pEW'] = []
            self.master['C-' + C + '_unc'] = []
        
        #Retrieve values.
        for t in t_exp_list: 
            self.master['pEW_obs'].append(self.master[t]['pEW_obs'])
            self.master['unc_obs'].append(self.master[t]['unc_obs'])
            for i, C in enumerate(self.master[t]['C']):
                self.master['C-' + C + '_pEW'].append(self.master[t]['pEW'][i])
                self.master['C-' + C + '_unc'].append(self.master[t]['unc'][i])

        #Plot velocity curves and observed data.
        self.ax_5.errorbar(
          t_exp_list, self.master['pEW_obs'],
          yerr=self.master['unc_obs'],
          color='k', ls='-', lw=2., marker='p', markersize=15., capsize=0.,
          label=r'$\mathrm{SN\ 2011fe}$')        

        for i, C in enumerate(C_list):
            self.ax_5.errorbar(
              t_exp_values + (-2. + i) * offset, self.master['C-' + C + '_pEW'],
              yerr=self.master['C-' + C + '_unc'],
              color=cmap(Norm(i)), ls='-', lw=2., marker=None, capsize=0.)

        self.ax_5.legend(frameon=False, fontsize=fs - 4., numpoints=1,
                         ncol=1, labelspacing=0.05, handletextpad=0., loc=1)  
        
    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_11fe_C-scaled.pdf',
                        format='pdf', dpi=360)
    
    def run_make_scan(self):
        self.set_fig_frame()
        self.make_bottom_plot()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()
                
if __name__ == '__main__':
    Make_Scan(show_fig=True, save_fig=True)

    
