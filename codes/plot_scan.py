#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import matplotlib.pyplot as plt
import matplotlib as mpl
import cPickle
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from matplotlib import cm
from matplotlib import colors

from mpl_toolkits.axes_grid.inset_locator import InsetPosition

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

t_exp_list = np.array(['3.7', '5.9', '9.0', '12.1', '16.1', '19.1'])
#v_stop = (list(np.arange(10500, 14499., 500.).astype('int').astype('str')))   
v_stop = (list(np.arange(10500, 14001., 500.).astype('int').astype('str')))   
cmap_mock = plt.cm.get_cmap('seismic')
cmap = plt.get_cmap('Reds')

N = len(v_stop)
Norm = colors.Normalize(vmin=0., vmax=N - 1.)
palette = [cmap(value) for value in
           np.arange(0.2, 1.001, (1. - 0.2) / (N - 1.))]
palette = palette[0:N]   

cmap_D = cmap_mock.from_list('irrel', palette, N)

fs = 20.

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

def w2v(w, w_rest):
    return 3.e5 * (w_rest / w - 1.)

def v2w(v, w_rest):
    return w_rest * 3.e5 / (3.e5 + v)

texp2date = {'3.7': '2011_08_25', '5.9': '2011_08_28', '9.0': '2011_08_31',
             '12.1': '2011_09_03', '16.1': '2011_09_07', '19.1': '2011_09_10',
             '22.4': '2011_09_13', '28.3': '2011_09_19'}

texp2v = {'3.7': '13300', '5.9': '12400', '9.0': '11300',
          '12.1': '10700', '16.1': '9000', '19.1': '7850',
          '22.4': '6700', '28.3': '4550'}

texp2L = {'3.7': 0.08e9, '5.9': 0.32e9, '9.0': 1.1e9,
          '12.1': 2.3e9, '16.1': 3.2e9, '19.1': 3.5e9,
          '22.4': 3.2e9, '28.3': 2.3e9}

texp2w_b = {'3.7': np.nan, '5.9': np.nan, '9.0': 6260.,
          '12.1': 6270., '16.1': 6280., '19.1': 6290.,
          '22.4': np.nan, '28.3': np.nan}

class Make_Scan(object):
    """
    Description:
    ------------
    Makes figure 1 in the paper. Plots the spectra of 11fe TARDIS simulations
    where the carbon profile of Mazzali+ 2014 is adopted. The color is coded
    by the depth (in velocity space) below which all carbon is removed.
    Multiple epochs are ploted and the bottom panels shows the respective
    pEW of the carbon trough in the optical. 
    
    Parameters:
    -----------
    compare_7d : ~bool
        If True: Also plot the observed spectra of 11fe at 7 days post-explosion.
                 used for comparison purposes only--this epoch seems to have
                 the strongest carbon trough in Parrent+ 2012.
        If False (default): Does not include the spectra at 7 days.
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_C-scan.pdf
    ./../OUTPUT_FILES/FIGURES/Fig_C-scan_with-7d-spectra.pdf

    References:
    -----------
    Parrent+ 2012: http://adsabs.harvard.edu/abs/2012ApJ...752L..26P
    MAzzali+ 2014: http://adsabs.harvard.edu/abs/2014MNRAS.439.1959M
    """
        
    def __init__(self, compare_7d=False, feature_mark=False, show_fig=True,
                 save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig
        self.compare_7d = compare_7d         
             
        v_stop_label = [str(float(v_in) / 1.e3) for v_in in v_stop]   

        self.master = {}

        self.fig = plt.figure(figsize=(8, 18))

        self.master['ax_0'] = plt.subplot2grid(
          (278, 20), (0, 0), rowspan=37, colspan=20)
        
        self.master['ax_1'] = plt.subplot2grid(
          (278, 20), (38, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_2'] = plt.subplot2grid(
          (278, 20), (76, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_3'] = plt.subplot2grid(
          (278, 20), (114, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_4'] = plt.subplot2grid(
          (278, 20), (152, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])

        self.master['ax_5'] = plt.subplot2grid(
          (278, 20), (190, 0), rowspan=37, colspan=20,
          sharex=self.master['ax_0'], sharey=self.master['ax_0'])
        
        self.master['ax_bot'] = plt.subplot2grid(
          (278, 20), (241, 0), rowspan=37, colspan=16)
        
        self.master['ax_cbar'] = plt.subplot2grid(
          (278, 20), (241, 16), rowspan=37, colspan=1)
                
        #N = len(v_stop)
        #color_disc = cmap(np.linspace(0, 1, N))
        #cmap_D = cmap_mock.from_list('irrel', color_disc, N)
        
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap_D, norm=Norm)
        aux_mappable.set_array([])
        aux_mappable.set_clim(vmin=-0.5, vmax=N-0.5)
        cbar = plt.colorbar(aux_mappable, cax=self.master['ax_cbar'])
        cbar.set_ticks(range(N))
        cbar.ax.tick_params(width=1, labelsize=fs)
        cbar.set_label(r'$v_{\mathrm{cut}} \ \mathrm{[10^3\ km\ s^{-1}]}$',
                       fontsize=fs)     
        cbar.set_ticklabels(v_stop_label)

        xloc = 0.70
        dx = 0.19
        dy = 0.055
        
        self.master['axi_o0'] = self.fig.add_axes([xloc, 0.816, dx, dy])
        self.master['axi_o1'] = self.fig.add_axes([xloc, 0.710, dx, dy])
        self.master['axi_o2'] = self.fig.add_axes([xloc, 0.605, dx, dy])
        self.master['axi_o3'] = self.fig.add_axes([xloc, 0.499, dx, dy])
        self.master['axi_o4'] = self.fig.add_axes([xloc, 0.394, dx, dy])
        self.master['axi_o5'] = self.fig.add_axes([xloc, 0.288, dx, dy])
        
        for i, time in enumerate(t_exp_list):
            idx = str(i)
            self.master[time] = Add_Curves(
              self.master['ax_' + idx], self.master['axi_o' + idx],
              self.master['ax_bot'], time, i, compare_7d, feature_mark).run_make_slab()        
    
        self.run_make_scan()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$\lambda \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{log \ f}_{\lambda}\ \mathrm{[arb.\ units]}$'

        x_cbar_label = r'$t_{\mathrm{exp}}\ \mathrm{[days]}$'
        y_cbar_label = (r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                      + r'$ \ \lambda$6580')       

        self.fig.text(0.07, 0.57, y_label, va='center', rotation='vertical',
                      fontsize=fs) 
        
        self.master['ax_0'].set_yscale('log')      

        self.master['ax_0'].set_ylim(0.05, 10.)      
        
        #self.master['ax_2'].set_ylabel(y_label, fontsize=fs, labelpad=8)
        self.master['ax_0'].set_xlim(3000., 12000.)
        self.master['ax_0'].tick_params(axis='y', which='major',
          labelsize=fs, pad=8)      

        self.master['ax_0'].tick_params('both', length=8, width=1,
          which='major', direction='in')
        self.master['ax_0'].tick_params('both', length=4, width=1,
          which='minor', direction='in')
        self.master['ax_0'].xaxis.set_minor_locator(MultipleLocator(500.))
        self.master['ax_0'].xaxis.set_major_locator(MultipleLocator(2000.))        
        self.master['ax_0'].tick_params(labelleft='off')          
        self.master['ax_0'].tick_params(labelbottom='off')
       
        for idx in ['1', '2', '3', '4', '5']:
            self.master['ax_' + idx].set_yscale('log')      
            self.master['ax_' + idx].tick_params(labelleft='off')          
            self.master['ax_' + idx].tick_params(labelbottom='off')    
            self.master['ax_' + idx].tick_params('both', length=8, width=1,
              which='major', direction='in')
            self.master['ax_' + idx].tick_params('both', length=4, width=1,
              which='minor', direction='in')

        self.master['ax_5'].set_xlabel(x_label, fontsize=fs, labelpad=2.)    
        self.master['ax_5'].tick_params(labelbottom='on')
        self.master['ax_5'].tick_params(axis='x', which='major',
                                        labelsize=fs, pad=4)

        self.master['ax_bot'].set_xlabel(x_cbar_label, fontsize=fs)
        self.master['ax_bot'].set_ylabel(y_cbar_label, fontsize=fs)
        self.master['ax_bot'].set_xlim(8., 20.)
        self.master['ax_bot'].set_ylim(-0.5, 15.)
        self.master['ax_bot'].tick_params(axis='y', which='major',
          labelsize=fs, pad=8)      
        self.master['ax_bot'].tick_params(axis='x', which='major',
          labelsize=fs, pad=8)
        self.master['ax_bot'].minorticks_on()
        self.master['ax_bot'].tick_params('both', length=8, width=1,
          which='major', direction='in')
        self.master['ax_bot'].tick_params('both', length=4, width=1,
          which='minor', direction='in')
        self.master['ax_bot'].xaxis.set_major_locator(MultipleLocator(2.))   
        self.master['ax_bot'].xaxis.set_minor_locator(MultipleLocator(1.))   
        self.master['ax_bot'].yaxis.set_major_locator(MultipleLocator(5.))  
        self.master['ax_bot'].yaxis.set_minor_locator(MultipleLocator(1.))
        self.master['ax_bot'].text(8.5, 12., r'$\mathbf{g}$',
                    fontsize=20., horizontalalignment='left', color='k') 
                    
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
        epochs = [float(t) for t in t_exp_list]
        self.master['ax_bot'].errorbar(
          epochs[2:], self.master['pEW_obs'][2:],
          yerr=np.asarray(self.master['unc_obs'][2:]),
          color='k', ls='-', lw=2., marker='p', markersize=15., capsize=0.,
          label=r'$\rm{SN\ 2011fe}$')        

        for i, v in enumerate(v_stop):
            self.master['ax_bot'].errorbar(
              t_exp_values + (-2. + i) * offset, self.master['v-' + str(int(v)) + '_pEW'],
              yerr=self.master['v-' + str(int(v)) + '_unc'],
              color=palette[i], ls='-', lw=2., marker=None, capsize=0.)

        self.master['ax_bot'].legend(
          frameon=False, fontsize=fs - 8., numpoints=1, ncol=1,
          labelspacing=0.05, handletextpad=0., loc=1)  

    def manage_output(self):
        if self.save_fig:
            fname = 'Fig_scan'
            if self.compare_7d:
                fname += '_with-7d-spectra'
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + fname + '.pdf', format='pdf', dpi=360,
                        bbox_inches='tight')
        if self.show_fig:
            plt.show()
        plt.close()
    
    def run_make_scan(self):
        self.set_fig_frame()
        self.make_bottom_plot()
        self.manage_output()

class Add_Curves(object):
    
    def __init__(self, ax, axi_o, ax_bot, t_exp, idx, add_7d, f_markers):
        
        self.t_exp = t_exp        
        self.ax = ax
        self.axi_o = axi_o
        self.ax_bot = ax_bot
        self.idx = idx
        self.add_7d = add_7d
        self.f_markers = f_markers
        self.t = str(int(round(float(self.t_exp))))
        self.panel = ['a', 'b', 'c', 'd', 'e', 'f']

        self.pkl_list = []
        self.D = {}
        self.D['vel'], self.D['pEW'], self.D['unc'] = [], [], []
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""

        x_label_o = r'$\lambda \ \mathrm{[\AA}]}$'
        x_label_n = r'$\lambda \ \mathrm{[\mu m}]}$'


        #self.axi_o.set_xlabel(x_label_o, fontsize=fs - 8., labelpad=-2.)
        self.axi_o.set_xlim(6200., 6550.)
        self.axi_o.set_ylim(0.70, 1.2)
        self.axi_o.tick_params(axis='y', which='major', labelsize=fs - 8.)      
        self.axi_o.tick_params(axis='x', which='major', labelsize=fs - 8.)
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

        self.axi_top = self.axi_o.twiny()
        self.axi_top.set_xlim(self.axi_o.get_xlim())
        self.axi_top.set_xticks(v2w(np.array([5000., 10000., 15000.]), 6580.))
        self.axi_top.set_xticklabels(['5', '10', '15'])
        self.axi_top.tick_params(
          axis='x', which='both', labelsize=fs - 8., pad=0., direction='in')      
        
    def add_texp_text(self):
        self.ax.text(3300., 0.08, r'$t_{\rm{exp}}=' + self.t_exp + '\\ \\rm{d}$',
                     fontsize=20., horizontalalignment='left', color='k')
        self.ax.text(3300., 0.2, r'$\mathbf{' + self.panel[self.idx] + '}$',
                     fontsize=20., horizontalalignment='left', color='k')

    def load_and_plot_synthetic_spectrum(self):
        
        def make_fpath(v):
            lum = lum2loglum(texp2L[self.t_exp])
            fname = 'v_stop_F1-' + v
            return (path_tardis_output + '11fe_' + self.t
                    + 'd_C-scan/' + fname + '/' + fname + '.pkl')
                    
        for i, v in enumerate(v_stop):
            try:
                with open(make_fpath(v), 'r') as inp:
                    pkl = cPickle.load(inp)
                    self.pkl_list.append(pkl)
                    flux_raw = pkl['flux_normalized']
                    #flux_raw = pkl['flux_smoothed']
                    self.ax.plot(pkl['wavelength_corr'], flux_raw,
                                 color=palette[i], ls='-', lw=2., zorder=1.)
                    self.axi_o.plot(pkl['wavelength_corr'], flux_raw,
                                       color=palette[i], ls='-', lw=2.)

                    if self.f_markers:
                        self.axi_o.plot(
                          pkl['wavelength_maxima_blue_fC'],
                          pkl['flux_maxima_blue_fC'], color=palette[i],
                          ls='None', marker='+')
                        self.axi_o.plot(
                          pkl['wavelength_maxima_red_fC'],
                          pkl['flux_maxima_red_fC'], color=palette[i],
                          ls='None', marker='x')

                    if np.isnan(pkl['pEW_fC']):
                        pkl['pEW_fC'], pkl['pEW_unc_fC'] = 0., 0.

                    if float(self.t_exp) > 7.:
                        pEW = pkl['pEW_fC']
                        self.D['pEW'].append(pEW)
                        self.D['unc'].append(pkl['pEW_unc_fC'])
                    else:
                        self.D['pEW'].append(np.nan)
                        self.D['unc'].append(np.nan)                        
                    self.D['vel'].append(float(v))
                
           
            except:
                self.D['vel'].append(float(v))
                self.D['pEW'].append(np.nan)
                self.D['unc'].append(np.nan)

    def load_and_plot_observational_spectrum(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        with open(directory + texp2date[self.t_exp] + '.pkl', 'r') as inp:
            pkl = cPickle.load(inp)
            flux_raw = pkl['flux_raw'] / pkl['norm_factor']
            self.ax.plot(
              pkl['wavelength_corr'], flux_raw, color='k', ls='-', lw=3.,
              zorder=2., label=r'$\mathrm{SN\ 2011fe}$')
            self.axi_o.plot(
              pkl['wavelength_corr'], flux_raw, color='k', ls='-', lw=3.,
              zorder=2.)            

            if self.f_markers:
                self.axi_o.plot(
                  pkl['wavelength_maxima_blue_fC'], pkl['flux_maxima_blue_fC'],
                  color='k', ls='None', marker='+')
                self.axi_o.plot(
                  pkl['wavelength_maxima_red_fC'], pkl['flux_maxima_red_fC'],
                  color='k', ls='None', marker='x')

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
 
        #Add spectrum at 7.2days for comparison.
        if self.add_7d:
            with open(directory + '2011_08_30.pkl', 'r') as inp:
                pkl = cPickle.load(inp)
                flux_raw = pkl['flux_raw'] / pkl['norm_factor']
                self.ax.plot(pkl['wavelength_corr'], flux_raw,
                             color='g', ls='-', lw=3., zorder=2.,
                             alpha=0.3, label=r'@7.2 days')
                self.axi_o.plot(pkl['wavelength_corr'], flux_raw,
                                color='g', ls='-', lw=3.,
                                alpha=0.3, zorder=2.)  

        #Plot observed spectra at top plot.
        if self.idx == 0:
            self.ax.legend(
              frameon=False, fontsize=fs, numpoints=1, ncol=1,
              loc=2, bbox_to_anchor=(0.,1.10), labelspacing=-0.1,
              handlelength=1.5, handletextpad=0.2)  
                 
    def run_make_slab(self):
        self.set_fig_frame()
        self.add_texp_text()
        self.load_and_plot_synthetic_spectrum()
        self.load_and_plot_observational_spectrum()
        return self.D
                
if __name__ == '__main__':
    Make_Scan(compare_7d=False, feature_mark=False, show_fig=True, save_fig=True)
    #Make_Scan(compare_7d=True, feature_mark=False, show_fig=True, save_fig=True)

    
