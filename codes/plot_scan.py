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

t_exp_list = np.array(['5.9', '9.0', '12.1', '16.1', '19.1'])
v_stop = (list(np.arange(10500, 14499., 500.).astype('int').astype('str')))   
cmap_mock = plt.cm.get_cmap('seismic')
cmap = cmaps.plasma
Norm = colors.Normalize(vmin=0., vmax=len(v_stop) - 1.)
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
    
    def __init__(self, ax, axins, ax_bot, t_exp, idx, add_7d):
        
        self.t_exp = t_exp        
        self.ax = ax
        self.axins = axins
        self.ax_bot = ax_bot
        self.idx = idx
        self.add_7d = add_7d
        self.t = str(int(round(float(self.t_exp))))
        self.panel = ['a', 'b', 'c', 'd', 'e']

        self.pkl_list = []
        self.D = {}
        self.D['vel'], self.D['pEW'], self.D['unc'] = [], [], []
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        self.axins.set_xlim(6200., 6550.)
        self.axins.set_ylim(0.60, 1.15)
        self.axins.tick_params(axis='y', which='major', labelsize=fs - 4., pad=8)      
        self.axins.tick_params(axis='x', which='major', labelsize=fs - 4., pad=8)
        self.axins.minorticks_on()
        self.axins.tick_params('both', length=8, width=1, which='major', pad=2)
        self.axins.tick_params('both', length=4, width=1, which='minor', pad=2)
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
            flux_raw = pkl['flux_raw'] / pkl['norm_factor']
            self.ax.plot(pkl['wavelength_corr'], flux_raw,
                         color='k', ls='-', lw=3., zorder=2.,
                         label=r'$\mathrm{SN\ 2011fe}$')
            self.axins.plot(pkl['wavelength_corr'], flux_raw,
                         color='k', ls='-', lw=3., zorder=2.)            

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
        
        self.ax.plot([x, x], [y + 0.7, y + 1.2], ls='-', marker='None',
                         color='grey', lw=2.)

        self.ax.text(x, y + 1.4, r'C', fontsize=20.,
                     horizontalalignment='center', color='grey')
        
        #Get measured pEW.
        if np.isnan(pkl['pEW_fC']):
            pkl['pEW_fC'], pkl['pEW_unc_fC'] = 0., 0.
        
        self.D['pEW_obs'] = pkl['pEW_fC']
        self.D['unc_obs'] = pkl['pEW_unc_fC']        
 
        #Add spectrum at 7.2days for comparison.
        if self.add_7d:
            with open(directory + '2011_08_30.pkl', 'r') as inp:
                pkl = cPickle.load(inp)
                flux_raw = pkl['flux_raw'] / pkl['norm_factor']
                self.ax.plot(pkl['wavelength_corr'], flux_raw,
                             color='g', ls='-', lw=3., zorder=2.,
                             alpha=0.3, label=r'@7.2 days')
                self.axins.plot(pkl['wavelength_corr'], flux_raw,
                                color='g', ls='-', lw=3.,
                                alpha=0.3, zorder=2.)  

        #Plot observed spectra at top plot.
        if self.idx == 0:
            self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                           handletextpad=0.5, labelspacing=0, loc=9) 

    def load_and_plot_synthetic_spectrum(self):
        
        def make_fpath(v):
            lum = lum2loglum(texp2L[self.t_exp])
            fname = 'loglum-' + lum + '_v_stop_F1-' + v
            return (path_tardis_output + '11fe_' + self.t
                    + 'd_C-scan/' + fname + '/' + fname + '.pkl')
                    
        for i, v in enumerate(v_stop):
            try:
                with open(make_fpath(v), 'r') as inp:
                    pkl = cPickle.load(inp)
                    self.pkl_list.append(pkl)
                    flux_raw = pkl['flux_raw'] / pkl['norm_factor']
                    self.ax.plot(pkl['wavelength_corr'], flux_raw,
                                 color=cmap(Norm(i)), ls='-', lw=2.,
                                 zorder=1.)
                    self.axins.plot(pkl['wavelength_corr'], flux_raw,
                                       color=cmap(Norm(i)), ls='-', lw=2.,
                                       zorder=1.)
                    pEW = pkl['pEW_fC']
                    if np.isnan(pEW):
                        pEW = 0.
                    self.D['vel'].append(float(v))
                    self.D['pEW'].append(pEW)
                    self.D['unc'].append(pkl['pEW_unc_fC'])                   
            except:
                self.D['vel'].append(float(v))
                self.D['pEW'].append(np.nan)
                self.D['unc'].append(np.nan)
                 
    def run_make_slab(self):
        self.set_fig_frame()
        self.add_texp_text()
        self.load_and_plot_observational_spectrum()
        self.load_and_plot_synthetic_spectrum()
        return self.D

class Make_Scan(object):
    
    def __init__(self, compare_7d=False, show_fig=True, save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig
        self.compare_7d = compare_7d         
             
        v_stop_label = [str(float(v_in) / 1.e3) for v_in in v_stop]   

        self.master = {}

        self.fig = plt.figure(figsize=(8, 18))

        self.master['ax_0'] = plt.subplot2grid(
          (240, 20), (0, 0), rowspan=37, colspan=20)
        
        self.master['ax_1'] = plt.subplot2grid(
          (240, 20), (38, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_2'] = plt.subplot2grid(
          (240, 20), (76, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_3'] = plt.subplot2grid(
          (240, 20), (114, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_4'] = plt.subplot2grid(
          (240, 20), (152, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_bot'] = plt.subplot2grid(
          (240, 20), (203, 0), rowspan=37, colspan=16)
        
        self.master['ax_cbar'] = plt.subplot2grid(
          (240, 20), (203, 16), rowspan=37, colspan=1)
                
        N = len(v_stop)
        color_disc = cmap(np.linspace(0, 1, N))
        cmap_D = cmap_mock.from_list('irrel', color_disc, N)
        
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap_D, norm=Norm)
        aux_mappable.set_array([])
        aux_mappable.set_clim(vmin=-0.5, vmax=N-0.5)
        cbar = plt.colorbar(aux_mappable, cax=self.master['ax_cbar'])
        cbar.set_ticks(range(N))
        cbar.ax.tick_params(width=1, labelsize=fs)
        cbar.set_label(r'$v_{\mathrm{cut}} \ \mathrm{[10^3\ km\ s^{-1}]}$',
                       fontsize=fs)     
        cbar.set_ticklabels(v_stop_label)

        self.master['ax_ins_0'] = self.fig.add_axes([0.66, 0.810, 0.22, 0.08])
        self.master['ax_ins_1'] = self.fig.add_axes([0.66, 0.684, 0.22, 0.08])
        self.master['ax_ins_2'] = self.fig.add_axes([0.66, 0.558, 0.22, 0.08])
        self.master['ax_ins_3'] = self.fig.add_axes([0.66, 0.430, 0.22, 0.08])
        self.master['ax_ins_4'] = self.fig.add_axes([0.66, 0.304, 0.22, 0.08])
        
        for i, time in enumerate(t_exp_list):
            idx = str(i)
            self.master[time] = Add_Curves(
              self.master['ax_' + idx], self.master['ax_ins_' + idx],
              self.master['ax_bot'], time, i, compare_7d).run_make_slab()        
    
        self.run_make_scan()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$\lambda \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'

        x_cbar_label = r'$t_{\mathrm{exp}}\ \mathrm{[days]}$'
        y_cbar_label = (r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                      + r'$ \ \lambda$6580')       

        self.master['ax_2'].set_ylabel(y_label, fontsize=fs, labelpad=8)

        
        self.master['ax_0'].set_xlim(1500., 10000.)
        self.master['ax_0'].set_ylim(0., 3.5)      
        self.master['ax_0'].tick_params(axis='y', which='major',
          labelsize=fs, pad=8)      
        self.master['ax_0'].tick_params(axis='x', which='major',
          labelsize=fs, pad=8)
        self.master['ax_0'].minorticks_on()
        self.master['ax_0'].tick_params('both', length=8, width=1,
          which='major')
        self.master['ax_0'].tick_params('both', length=4, width=1,
          which='minor')
        self.master['ax_0'].xaxis.set_minor_locator(MultipleLocator(500.))
        self.master['ax_0'].xaxis.set_major_locator(MultipleLocator(2000.))   
        self.master['ax_0'].yaxis.set_minor_locator(MultipleLocator(0.5))
        self.master['ax_0'].yaxis.set_major_locator(MultipleLocator(1.))        
        self.master['ax_0'].tick_params(labelleft='off')          
        self.master['ax_0'].tick_params(labelbottom='off')          
       
        for idx in ['1', '2', '3', '4']:
            self.master['ax_' + idx].tick_params(labelleft='off')          
            self.master['ax_' + idx].tick_params(labelbottom='off')    
            self.master['ax_' + idx].tick_params('both', length=8, width=1,
              which='major')
            self.master['ax_' + idx].tick_params('both', length=4, width=1,
              which='minor')
            if idx == '4':
                self.master['ax_' + idx].set_xlabel(x_label, fontsize=fs,
                                                    labelpad=2.)    
                self.master['ax_' + idx].tick_params(labelbottom='on')
                self.master['ax_' + idx].tick_params(axis='x', which='major',
                  labelsize=fs, pad=4)
                        
        self.master['ax_bot'].set_xlabel(x_cbar_label, fontsize=fs)
        self.master['ax_bot'].set_ylabel(y_cbar_label, fontsize=fs)
        self.master['ax_bot'].set_xlim(5.0, 20.)
        self.master['ax_bot'].set_ylim(-0.5, 15.)
        self.master['ax_bot'].tick_params(axis='y', which='major',
          labelsize=fs, pad=8)      
        self.master['ax_bot'].tick_params(axis='x', which='major',
          labelsize=fs, pad=8)
        self.master['ax_bot'].minorticks_on()
        self.master['ax_bot'].tick_params('both', length=8, width=1,
          which='major')
        self.master['ax_bot'].tick_params('y', length=4, width=1,
          which='minor')
        self.master['ax_bot'].xaxis.set_major_locator(MultipleLocator(1.))   
        self.master['ax_bot'].xaxis.set_minor_locator(MultipleLocator(1.))   
        self.master['ax_bot'].yaxis.set_minor_locator(MultipleLocator(1.))
        self.master['ax_bot'].yaxis.set_major_locator(MultipleLocator(2.))  

    def make_bottom_plot(self):
        
        
        t_exp_values = t_exp_list.astype(float)
        offset = 0.03
        
        #Initialize lists to be plotted.
        self.master['pEW_obs'], self.master['unc_obs'] = [], []
        for v in v_stop:
            self.master['v-' + str(v) + '_pEW'] = []
            self.master['v-' + str(v) + '_unc'] = []
        
        #Retrieve values.
        for t in t_exp_list: 
            self.master['pEW_obs'].append(self.master[t]['pEW_obs'])
            self.master['unc_obs'].append(self.master[t]['unc_obs'])
            for i, v in enumerate(self.master[t]['vel']):
                self.master['v-' + str(int(v)) + '_pEW'].append(self.master[t]['pEW'][i])
                self.master['v-' + str(int(v)) + '_unc'].append(self.master[t]['unc'][i])

        #Plot velocity curves and observed data.
        self.master['ax_bot'].errorbar(
          t_exp_list, self.master['pEW_obs'],
          yerr=self.master['unc_obs'],
          color='k', ls='-', lw=2., marker='p', markersize=15., capsize=0.,
          label=r'$\mathrm{SN\ 2011fe}$')        

        for i, v in enumerate(v_stop):
            self.master['ax_bot'].errorbar(
              t_exp_values + (-2. + i) * offset, self.master['v-' + str(int(v)) + '_pEW'],
              yerr=self.master['v-' + str(int(v)) + '_unc'],
              color=cmap(Norm(i)), ls='-', lw=2., marker=None, capsize=0.)

        self.master['ax_bot'].legend(
          frameon=False, fontsize=fs - 4., numpoints=1, ncol=1,
          labelspacing=0.05, handletextpad=0., loc=1)  

        self.master['ax_bot'].text(5.6, 10., r'$\mathbf{f}$',
                    fontsize=20., horizontalalignment='left', color='k')
        
    def save_figure(self):        
        if self.save_fig:
            if self.compare_7d:
                fname = 'Fig_11fe_C-scan_comp.pdf'
            else:
                fname = 'Fig_11fe_C-scan.pdf'
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + fname, format='pdf', dpi=360,
                        bbox_inches='tight')
    
    def run_make_scan(self):
        self.set_fig_frame()
        self.make_bottom_plot()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()
                
if __name__ == '__main__':
    Make_Scan(compare_7d=False, show_fig=True, save_fig=True)
    Make_Scan(compare_7d=True, show_fig=True, save_fig=True)

    