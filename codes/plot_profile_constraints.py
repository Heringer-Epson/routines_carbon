#!/usr/bin/env python
                                                 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

mpl.rcParams['hatch.color'] = '#e41a1c'

mass_fractions = ['0.00', '0.05', '0.1', '0.2', '0.5', '1.00', '2.00', '5.00', '10.00']
labels = [r'$\mathrm{0}$', r'$\mathrm{5\times 10^{-4}}$', r'$\mathrm{10^{-3}}$',
          r'$\mathrm{2\times 10^{-3}}$', r'$\mathrm{5\times 10^{-3}}$',
          r'$\mathrm{10^{-2}}$', r'$\mathrm{2\times 10^{-2}}$',
          r'$\mathrm{5\times 10^{-2}}$', r'$\mathrm{10^{-1}}$']
velocities = [7850., 9000., 10700., 11300., 12400., 13300., 16000.]
fs = 22.

class Plot_Constraints(object):
    """
    Description:
    ------------
    Makes Fig. 3 in the carbon paper. This code plots the the region of the
    carbon mass fraction - velocity space that is ruled out by the analysis
    in the paper.
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_model_constraints.pdf
    """
    
    def __init__(self, show_fig=True, save_fig=True):
        
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.fig = plt.figure(figsize=(16.,16.))
        self.ax = plt.subplot(111) 
        
        self.run_make()

    def set_fig_frame(self):
        
        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = r'$X(\rm{C})$'

        self.ax.set_xlabel(x_label, fontsize=fs + 6.)
        self.ax.set_ylabel(y_label, fontsize=fs + 6.)
        self.ax.set_xlim(velocities[0], velocities[-1])
        self.ax.set_ylim(0, len(mass_fractions) - 1)
        self.ax.set_yticklabels(labels)
        self.ax.tick_params(axis='y', which='major', labelsize=fs + 6., pad=12)       
        self.ax.tick_params(axis='x', which='major', labelsize=fs + 6., pad=12)
        self.ax.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax.tick_params(
          'both', length=4, width=1, which='minor', direction='in')
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(2000.))  

        for i in np.arange(1, len(mass_fractions)):
            plt.axhline(y=i, ls='--', lw=2., color='gray')

        for v in velocities[1:-1]:
            plt.axvline(x=v, ls='--', lw=2., color='gray')

    def add_patches(self):
        
        color = '#e41a1c'
        
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
       
        string = '$t=5.9,\ 9\ \mathrm{and}\ 12.1\ \mathrm{d}$'
        self.ax.text(
          14630., 2, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=9\ \mathrm{d}$'
        self.ax.text(
          14700., 7.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = 'consequence of $X(C)_{\mathrm{i}}\leq X(C)_{\mathrm{o}}$'
        self.ax.text(
          10700., 7.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=19.1\ \mathrm{d}$'
        self.ax.text(
          8410., 5.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=16.1\ \mathrm{d}$'
        self.ax.text(
          9850., 5.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'))

        string = '$t=12.1\ \mathrm{d}$'
        self.ax.text(
          11000., 5.5, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w'), rotation=90)                                               

        string = '$t=9\ \mathrm{d}$'
        self.ax.text(
          11850., 6, string, color='k', fontsize=fs, ha='center', va='center',
          bbox=dict(facecolor='w', edgecolor='w')) 

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_model_constraints.pdf', format='pdf',
                        dpi=360, bbox_inches='tight')
    
    def run_make(self):
        self.set_fig_frame()
        self.add_patches()
        self.add_texts()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
if __name__ == '__main__':
    Plot_Constraints(show_fig=True, save_fig=True)
