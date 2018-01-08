#!/usr/bin/env python

import os                                                               
import sys
import itertools                                                        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import constants as const
from astropy import units as u
from scipy.integrate import trapz, cumtrapz

M_sun = const.M_sun.to('g').value

mass_fractions = ['0.00', '0.05', '0.1', '0.2', '0.5', '1.00', '2.00', '5.00', '10.00']
velocities = [7850., 9000., 10700., 11300., 12400., 13300., 16000.]

fs = 26.

class Plot_Models(object):
    '''Note, metallicity seems to have a nearly negligible impact on the
    location of carbon
    '''
    
    def __init__(self, show_fig=True, save_fig=True):
        
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.fig = plt.figure(figsize=(16.,10.))
        self.ax = plt.subplot(111) 
        
        self.run_make()

    def set_fig_frame(self):
        
        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = r'$X(\rm{C})$'

        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_xlim(velocities[0], velocities[-1])
        self.ax.set_ylim(0, len(mass_fractions) - 1)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.set_yticklabels(mass_fractions)
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(2000.))  

        for i in np.arange(1, len(mass_fractions)):
            plt.axhline(y=i, ls='--', lw=2., color='gray')

        for v in velocities[1:-1]:
            plt.axvline(x=v, ls='--', lw=2., color='gray')

    def add_patches(self):
        
        color = 'm'
        
        self.ax.add_patch(mpl.patches.Rectangle((13300., 0), 16000. - 13300., 4,
                          hatch='//', fill=False, snap=False, color=color, lw=3))

        self.ax.add_patch(mpl.patches.Rectangle((13300., 7), 16000. - 13300., 1,
                          hatch='//', fill=False, snap=False, color=color, lw=3))

        self.ax.add_patch(mpl.patches.Rectangle((7850., 7), 13300. - 7850., 1,
                          hatch='//', fill=False, snap=False, color=color, lw=3))

        self.ax.add_patch(mpl.patches.Rectangle((7850., 4), 9000. - 7850., 3,
                          hatch='//', fill=False, snap=False, color=color, lw=3))

        self.ax.add_patch(mpl.patches.Rectangle((9000., 4), 10700. - 9000., 3,
                          hatch='//', fill=False, snap=False, color=color, lw=3))

        self.ax.add_patch(mpl.patches.Rectangle((10700., 4), 11300. - 10700., 3,
                          hatch='//', fill=False, snap=False, color=color, lw=3))

        self.ax.add_patch(mpl.patches.Rectangle((11300., 5), 12400. - 11300., 2,
                          hatch='//', fill=False, snap=False, color=color, lw=3))

    def add_texts(self):
       
        string = '$t=6,\ 9\ \mathrm{and}\ 12\ \mathrm{days}$\n'\
                 'req: $X(C)_{\mathrm{i}}=X(C)_{\mathrm{o}}$'
        self.ax.text(
          14700., 2, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=9\ \mathrm{days}$'
        self.ax.text(
          14700., 7.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = 'consequence of $X(C)_{\mathrm{i}}\leq X(C)_{\mathrm{o}}$'
        self.ax.text(
          10700., 7.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=19$\n$\mathrm{days}$'
        self.ax.text(
          8450., 5.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=16\ \mathrm{days}$'
        self.ax.text(
          9850., 5.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=12\ \mathrm{days}$'
        self.ax.text(
          11000., 5.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'), rotation=90)                                               

        string = '$t=9$\n$\mathrm{days}$'
        self.ax.text(
          11800., 6, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w')) 

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_model_constraints.pdf', format='pdf',
                        dpi=360)
    
    def run_make(self):
        self.set_fig_frame()
        self.add_patches()
        self.add_texts()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
if __name__ == '__main__':
    Plot_Models(show_fig=True, save_fig=True)
    
#Try to obtain the following predictions:
#https://arxiv.org/pdf/1706.01898.pdf (Shen+ 2017)
#https://arxiv.org/pdf/1002.2173.pdf (Fink+ 2010)
