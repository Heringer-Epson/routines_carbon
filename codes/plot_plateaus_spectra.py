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

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

date_list = np.array(['5.9', '9.0', '12.1', '16.1', '19.1'])
scales = ['0.00', '0.05', '0.1', '0.2', '0.5', '1.00', '2.00', '5.00',
          '10.00', '20.00']
          
s1_list = ['0.00', '0.05', '0.1', '0.2', '0.5']
s2_list = ['0.00', '0.05', '0.1', '0.2', '0.5', '1.00', '2.00']

s1_list = ['0.00', '0.05', '0.1', '0.2', '0.5', '1.00', '2.00', '5.00']
s2_list = ['0.2', '0.5', '1.00', '2.00', '5.00']

#s1_list = ['0.1', '0.2', '0.5', '1.00', '2.00']
#s2_list = ['0.1', '0.2', '0.5', '1.00', '2.00']


N_s1 = len(s1_list)
N_s2 = len(s2_list)
Nd = len(date_list)

#labels = ['0', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20']
mass_fractions = np.asarray(scales).astype(float) * 0.01

cmap_mock = plt.cm.get_cmap('seismic')
cmap = cmaps.viridis
Norm = colors.Normalize(vmin=0., vmax=N_s2 - 1.)
fs = 20.

dist = (6.4e6 * u.pc).to(u.cm)
distance_factor = (4. * np.pi * dist**2.).value

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

class Plateaus_Spectra(object):
    
    def __init__(self, show_fig=True, save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig   
            
        self.F = {}
        self.F['fig'] = plt.figure(figsize=(24,12))
        for j in range(Nd):
            for i in range(N_s1):
                idx = j * N_s1 + i + 1
                self.F['ax' + str(idx)] = plt.subplot(Nd, N_s1, idx)
        
        self.F['fig'].subplots_adjust(left=0.08, right=0.85, bottom=0.12) 
        self.F['bar_ax'] = self.F['fig'].add_axes([0.87, 0.32, 0.02, 0.35])    
        
        plt.subplots_adjust(wspace=0.15, hspace=0.05)
        
        self.run_make()

    def set_frame(self):

        #Make label.
        x_label = (r'$X(\rm{C}) \ \mathrm{[\%]}$ at 7850 $\leq\ v \ \leq$'\
                   r' 13300$ \ \mathrm{[km\ s^{-1}]}$')     
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'

        self.F['fig'].text(0.35, 0.04, x_label, va='center',
                           rotation='horizontal', fontsize=fs)  
        self.F['fig'].text(0.02, 0.51, y_label, va='center',
                           rotation='vertical', fontsize=fs)  
                                   
        #Frame Settings.
        for j in range(Nd):
            for i in range(N_s1):
                idx = str(j * N_s1 + i + 1)


                self.F['ax' + idx].set_xlim(6200., 6550.)
                self.F['ax' + idx].set_ylim(0.60, 1.15)
                self.F['ax' + idx].tick_params(axis='y', which='major', labelsize=fs, pad=8)      
                self.F['ax' + idx].tick_params(axis='x', which='major', labelsize=fs, pad=8)
                self.F['ax' + idx].minorticks_on()
                self.F['ax' + idx].tick_params('both', length=8, width=1, which='major', pad=8)
                self.F['ax' + idx].tick_params('both', length=4, width=1, which='minor', pad=8)
                self.F['ax' + idx].xaxis.set_minor_locator(MultipleLocator(50.))
                self.F['ax' + idx].xaxis.set_major_locator(MultipleLocator(100.))
                self.F['ax' + idx].yaxis.set_minor_locator(MultipleLocator(0.05))
                self.F['ax' + idx].yaxis.set_major_locator(MultipleLocator(0.2)) 
                self.F['ax' + idx].tick_params(labelleft='off')          

                if i == 0:
                    y_label = r'$\mathrm{t_{exp}}=' + date_list[j] + '\ \mathrm{d}$'
                    self.F['ax' + idx].set_ylabel(y_label, fontsize=fs,
                                                  labelpad=10.)
                if j == Nd - 1:
                    x_label = r'$X(\rm{C})=' + s1_list[i] + '$'
                   
                    self.F['ax' + idx].set_xlabel(x_label, fontsize=fs,
                                                  labelpad=10.)                    
                else:
                    self.F['ax' + idx].tick_params(labelbottom='off')          

    def load_observational_data(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        
        for j, date in enumerate(date_list):
            with open(directory + texp2date[date] + '.pkl', 'r') as inp:
                pkl = cPickle.load(inp)

                for i in range(N_s1):
                    idx = str(j * N_s1 + i + 1)

                    #self.F['ax' + idx].plot(
                    #  pkl['wavelength_corr'], pkl['flux_smoothed'],
                    #  color='k', ls='-', lw=3., zorder=2.)

                    flux_raw = pkl['flux_raw'] / pkl['norm_factor']
                    self.F['ax' + idx].plot(
                      pkl['wavelength_corr'], flux_raw,
                      color='k', ls='-', lw=3., zorder=2.)

    def main_loop(self):

        def get_fname(date, s1, s2): 
            t = str(int(round(float(date))))
            case_folder = '11fe_' + t + 'd_C-plateaus_scaling_SN/'
            fname = 'C-F2-' + s2 + '_C-F1-' + s1
            fname = case_folder + fname + '/' + fname + '.pkl'        
            return path_tardis_output + fname     
        
        for j, date in enumerate(date_list):
            for i, s1 in enumerate(s1_list):
                idx = str(j * N_s1 + i + 1)
                for k, s2 in enumerate(s2_list):
            
                    if float(s2) >= float(s1):
                        fpath = get_fname(date, s1, s2)
                        with open(fpath, 'r') as inp:
                            pkl = cPickle.load(inp)

                            #self.F['ax' + idx].plot(
                            #  pkl['wavelength_corr'], pkl['flux_smoothed'],
                            #  color=cmap(Norm(k)), ls='-', lw=2., zorder=1.)                

                            flux_raw = pkl['flux_raw'] / pkl['norm_factor']
                            self.F['ax' + idx].plot(
                              pkl['wavelength_corr'], flux_raw,
                              color=cmap(Norm(k)), ls='-', lw=2., zorder=1.)  


    def make_colorbars(self):

        color_disc = cmap(np.linspace(0, 1, N_s2))
        cmap_D = cmap_mock.from_list('irrel', color_disc, N_s2)
        
        aux_mappable = mpl.cm.ScalarMappable(cmap=cmap_D, norm=Norm)
        aux_mappable.set_array([])
        aux_mappable.set_clim(vmin=-0.5, vmax=N_s2-0.5)
        cbar = plt.colorbar(aux_mappable, cax=self.F['bar_ax'])
        cbar.set_ticks(range(N_s2))
        cbar.ax.tick_params(width=1, labelsize=fs)
        cbar_label = (r'$X(\rm{C}) \ \mathrm{[\%]}$ at 13300 $\leq\ v \ \leq$'\
                      r' 16000$ \ \mathrm{[km\ s^{-1}]}$')
        cbar.set_label(cbar_label, fontsize=fs)     
        cbar.set_ticklabels(s2_list)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_11fe_C-plateaus-spectra.pdf',
                        format='pdf', dpi=360)

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
    Plateaus_Spectra(show_fig=True, save_fig=True)

    
