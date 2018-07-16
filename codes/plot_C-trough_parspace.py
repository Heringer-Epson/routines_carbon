#!/usr/bin/env python

import os                                                               
path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cPickle
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import NullFormatter
from astropy import units as u
from matplotlib import cm
from matplotlib import colors

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'  

fs = 20.

texp2date = {'3.7': '2011_08_25', '5.9': '2011_08_28', '9.0': '2011_08_31',
             '12.1': '2011_09_03', '16.1': '2011_09_07', '19.1': '2011_09_10',
             '22.4': '2011_09_13', '28.3': '2011_09_19'}

texp2v = {'3.7': '13300', '5.9': '12400', '9.0': '11300',
          '12.1': '10700', '16.1': '9000', '19.1': '7850',
          '22.4': '6700', '28.3': '4550'}

texp2L = {'3.7': 0.08e9, '5.9': 0.32e9, '9.0': 1.1e9,
          '12.1': 2.3e9, '16.1': 3.2e9, '19.1': 3.5e9,
          '22.4': 3.2e9, '28.3': 2.3e9}

class Ctrough_Spectra(object):
    """
    Description:
    ------------
    Makes figure 2 in the paper. Plots the spectra in the region of the carbon
    trough for a suit of carbon profiles at multiple photospheric
    epochs. 
    
    Parameters:
    -----------
    region : ~str
        Either 'optical' or 'NIR'. This determines if the carbon trough to be
        plotted is the one in the optical (~6300 angs) or NIR (~10300 angs).
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_C-trough-spectra_optical.pdf
    ./../OUTPUT_FILES/FIGURES/Fig_C-trough-spectra_NIR.pdf
    """
    
    def __init__(self, region='optical', show_fig=True, save_fig=False):
        self.region = region
        self.show_fig = show_fig
        self.save_fig = save_fig   

        #Make adjustments depending on region of the spectra to be plotted.
        self.date_list = np.array(['5.9', '9.0', '12.1', '16.1', '19.1'])        
        self.Nd = len(self.date_list)
        
        if self.region == 'optical':
            self.s1_list = [
              '0.00', '0.05', '0.1', '0.2', '0.5', '1.00', '2.00', '5.00']
            self.s1_labels = [
              '\mathrm{0}', '\mathrm{5\\times 10^{-4}}', '\mathrm{10^{-3}}',
              '\mathrm{2\\times 10^{-3}}', '\mathrm{5\\times 10^{-3}}',
              '\mathrm{10^{-2}}', '\mathrm{2\\times 10^{-2}}',
              '\mathrm{5\\times 10^{-2}}']        
            self.s2_list = ['0.2', '0.5', '1.00', '2.00', '5.00']
            self.s2_labels = [
              r'$\mathrm{2\times 10^{-3}}$', r'$\mathrm{5\times 10^{-3}}$',
              r'$\mathrm{10^{-2}}$', r'$\mathrm{2\times 10^{-2}}$',
              r'$\mathrm{5\times 10^{-2}}$']
       
        elif self.region == 'NIR': 
            self.s1_list = [
              '0.00', '0.1', '0.2', '0.5', '1.00', '2.00', '5.00', '10.00']
            self.s1_labels = [
              '\mathrm{0}', '\mathrm{10^{-3}}',
              '\mathrm{2\\times 10^{-3}}', '\mathrm{5\\times 10^{-3}}',
              '\mathrm{10^{-2}}', '\mathrm{2\\times 10^{-2}}',
              '\mathrm{5\\times 10^{-2}}', '\mathrm{10^{-1}}']        
            self.s2_list = ['0.00', '1.00', '2.00', '5.00', '10.00']
            self.s2_labels = [
              r'$0$', r'$\mathrm{10^{-2}}$', r'$\mathrm{2\times 10^{-2}}$',
              r'$\mathrm{5\times 10^{-2}}$', r'$\mathrm{1\times 10^{-1}}$']
        
        self.N_s1 = len(self.s1_list)
        self.N_s2 = len(self.s2_list)

        #Set Figure.
        self.F = {}
        self.F['fig'] = plt.figure(figsize=(18,14))
        for j in range(self.Nd):
            for i in range(self.N_s1):
                idx = j * self.N_s1 + i + 1
                self.F['ax' + str(idx)] = plt.subplot(self.Nd, self.N_s1, idx)
        
        self.F['fig'].subplots_adjust(left=0.08, right=0.85, top=0.91, bottom=0.07) 
        self.F['bar_ax'] = self.F['fig'].add_axes([0.87, 0.32, 0.02, 0.35])    
        
        plt.subplots_adjust(wspace=0.15, hspace=0.05)        

        #Set colormap preferences.
        self.cmap_mock = plt.cm.get_cmap('seismic')
        self.cmap = plt.get_cmap('viridis')
        
        #Two option of color schemes. The viridis one does not use the full color range
        #on purpose (yellow rejected). Set this by upper value - default=0.9
        self.palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                        '#ffff33', '#a65628', '#f781bf', '#999999']
        self.palette = [self.cmap(value) for value in
                        np.arange(0., 0.901, 0.90 / (self.N_s2 - 1.))]
        self.palette = self.palette[0:self.N_s2]       
                
        self.run_make()

    def set_frame(self):

        #Make label.
        x_label_XC = (r'$X(\rm{C})$ at 7850 $\leq\ v \ \leq$'\
                   r' 13300$ \ \mathrm{[km\ s^{-1}]}$')     
        x_label_wl = r'$\lambda \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'

        self.F['fig'].text(0.35, 0.95, x_label_XC, va='center',
                           rotation='horizontal', fontsize=fs)  
        self.F['fig'].text(0.45, 0.03, x_label_wl, va='center',
                           rotation='horizontal', fontsize=fs)  
        self.F['fig'].text(0.02, 0.51, y_label, va='center',
                           rotation='vertical', fontsize=fs)  
                                   
        #Frame Settings.
        for j in range(self.Nd):
            for i in range(self.N_s1):
                idx = str(j * self.N_s1 + i + 1)

                if self.region == 'optical':
                    self.F['ax' + idx].set_xlim(6200., 6550.)
                    self.F['ax' + idx].set_ylim(0.60, 1.15)
                    self.F['ax' + idx].xaxis.set_minor_locator(MultipleLocator(50.))
                    self.F['ax' + idx].xaxis.set_major_locator(MultipleLocator(200.))
                    self.F['ax' + idx].yaxis.set_minor_locator(MultipleLocator(0.05))
                    self.F['ax' + idx].yaxis.set_major_locator(MultipleLocator(0.2))            
                elif self.region == 'NIR': 
                    self.F['ax' + idx].set_yscale('log')
                    self.F['ax' + idx].set_xlim(1., 1.1)
                    self.F['ax' + idx].set_ylim(0.05, 0.5)
                    self.F['ax' + idx].xaxis.set_minor_locator(MultipleLocator(0.01))
                    self.F['ax' + idx].xaxis.set_major_locator(MultipleLocator(0.05))
                    xticks = ['1', '0.05', '1.1']
                    self.F['ax' + idx].set_xticklabels(xticks)

                self.F['ax' + idx].tick_params(
                  axis='y', which='major', labelsize=fs, pad=8)      
                self.F['ax' + idx].tick_params(
                  axis='x', which='major', labelsize=fs, pad=8)
                self.F['ax' + idx].minorticks_on()
                self.F['ax' + idx].tick_params(
                  'both', length=8, width=1, which='major', pad=8, direction='in')
                self.F['ax' + idx].tick_params(
                  'both', length=4, width=1, which='minor', pad=8, direction='in')
                self.F['ax' + idx].xaxis.set_ticks_position('both')
                self.F['ax' + idx].yaxis.set_ticks_position('both')
                self.F['ax' + idx].yaxis.set_major_formatter(NullFormatter())
                self.F['ax' + idx].yaxis.set_minor_formatter(NullFormatter())
                
                if i == 0:
                    y_label = (r'$\mathrm{t_{exp}}=' + self.date_list[j]
                    + '\ \mathrm{d}$')
                    self.F['ax' + idx].set_ylabel(
                      y_label, fontsize=fs, labelpad=10.)
                
                if j != self.Nd - 1:
                    self.F['ax' + idx].tick_params(labelbottom='off')          
             
                if j == 0:
                    x_label = r'$X(\rm{C})=' + self.s1_labels[i] + '$'
                    self.F['ax' + idx].set_xlabel(
                      x_label, fontsize=fs)                    
                    self.F['ax' + idx].xaxis.set_label_position('top')

    def load_observational_data(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        
        for j, date in enumerate(self.date_list):
            with open(directory + texp2date[date] + '.pkl', 'r') as inp:
                pkl = cPickle.load(inp)

                for i in range(self.N_s1):
                    idx = str(j * self.N_s1 + i + 1)
                    
                    flux_raw = pkl['flux_raw'] / pkl['norm_factor']
                    
                    if self.region == 'optical':
                        self.F['ax' + idx].plot(
                          pkl['wavelength_corr'], flux_raw,
                          color='k', ls='-', lw=3., zorder=2.)
                    elif self.region == 'NIR':
                        self.F['ax' + idx].plot(
                          pkl['wavelength_corr'] / 1.e4, flux_raw,
                          color='k', ls='-', lw=3., zorder=2.)

    def main_loop(self):

        def get_fname(date, s1, s2): 
            t = str(int(round(float(date))))
            case_folder = '11fe_' + t + 'd_C-plateaus_scaling/'
            fname = 'C-F2-' + s2 + '_C-F1-' + s1
            fname = case_folder + fname + '/' + fname + '.pkl'        
            return path_tardis_output + fname     
        
        for j, date in enumerate(self.date_list):
            for i, s1 in enumerate(self.s1_list):
                idx = str(j * self.N_s1 + i + 1)
                for k, s2 in enumerate(self.s2_list):
            
                    if float(s2) >= float(s1):
                        fpath = get_fname(date, s1, s2)
                        with open(fpath, 'r') as inp:
                            pkl = cPickle.load(inp)

                            flux_raw = pkl['flux_raw'] / pkl['norm_factor']
                            
                            if self.region == 'optical':
                                self.F['ax' + idx].plot(
                                  pkl['wavelength_corr'], flux_raw, ls='-', 
                                  color=self.palette[k], lw=2., zorder=1.)  
                            elif self.region == 'NIR':
                                self.F['ax' + idx].plot(
                                  pkl['wavelength_corr'] / 1.e4, flux_raw, ls='-',
                                  color=self.palette[k], lw=2., zorder=1.)  

    def make_colorbars(self):

        cmap_D = self.cmap_mock.from_list('irrel', self.palette, self.N_s2)
        
        Norm_mock = colors.Normalize(vmin=0., vmax=1.) 
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap_D, norm=Norm_mock)
        aux_mappable.set_array([])
        aux_mappable.set_clim(vmin=-0.5, vmax=self.N_s2-0.5)
        cbar = plt.colorbar(aux_mappable, cax=self.F['bar_ax'])
        cbar.set_ticks(range(self.N_s2))
        cbar.ax.tick_params(width=1, labelsize=fs)
        cbar_label = (r'$X(\rm{C}) \ \mathrm{[\%]}$ at 13300 $\leq\ v \ \leq$'\
                      r' 16000$ \ \mathrm{[km\ s^{-1}]}$')
        cbar.set_label(cbar_label, fontsize=fs)     
        cbar.set_ticklabels(self.s2_labels)

    def save_figure(self):        
        if self.save_fig:
            fname = 'Fig_C-trough-spectra_' + self.region + '.pdf'
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + fname, format='pdf',
                        dpi=360, bbox_inches='tight')

    def run_make(self):
        self.set_frame()
        self.load_observational_data()
        self.main_loop()
        self.make_colorbars()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()        

if __name__ == '__main__':
    Ctrough_Spectra(region='optical', show_fig=True, save_fig=False)
    #Ctrough_Spectra(region='NIR', show_fig=True, save_fig=True)

    
