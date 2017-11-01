#!/usr/bin/env python

import os                                                               
import sys
import time

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LogFormatter
import new_colormaps as cmaps
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

class Plot_C_Scaled_Grid(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, show_fig=True, save_fig=False):
        self.show_fig = show_fig
        self.save_fig = save_fig 
            
        self.L_scal = np.arange(0.2, 1.61, 0.1)[1::]
        self.L = [lum2loglum(2.3e9 * l) for l in self.L_scal]              
        self.C = ['0.00', '0.20',
                  '0.50', '1.00', '2.00', '5.00', '10.00', '20.00']

        self.Nx = len(self.C)
        self.Ny = len(self.L)
        self.T_inner_1D, self.pEW_1D = [], []
        self.vmin, self.vmax = None, None
        
        self.pEW = None
        self.qtty = None
        self.norm = None
        self._im = None
        self.lower_limit = None 
        self.tick_range = None
        self.title = None
        self.cbar_label = None
        
        self.fig, self.ax = plt.subplots(figsize=(10,10))

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
        
        self.make_plot()
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        
        x_label = r'$\rm{C \ scaling}$'
        y_label = r'$L\ /\ L_{\mathrm{11fe}}$'
        
        self.ax.set_xlabel(x_label, fontsize=self.fs_label)
        self.ax.set_ylabel(y_label, fontsize=self.fs_label)
        
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')    

    def load_data(self):
        
        case_folder = '11fe_12d_C-scaled_v0/'
        def get_fname(C, L): 
            fname = 'C-' + C + '_loglum-' + L
            fname = case_folder + fname + '/' + fname + '.pkl'
            return path_tardis_output + fname     
            
        for L in self.L:
            for C in self.C:
                with open(get_fname(C, L), 'r') as inp:
                    pkl = cPickle.load(inp)
                    
                    pEW = pkl['pEW_fC']
                    if np.isnan(pEW):
                        pEW = 0.5 
                    self.pEW_1D.append(pEW)                    
                    self.T_inner_1D.append(pkl['t_inner'].value)

    def plotting(self):

        pEW_2D = np.reshape(self.pEW_1D, (self.Ny, self.Nx))
        self.vmin = 0.5
        self.vmax = int(max(self.pEW_1D)) + 1.
                
        #imshow
        self._im = plt.imshow(pEW_2D, interpolation='none', aspect='auto',
                            extent=[1., self.Nx + 1., 1., self.Ny + 1.],
                            origin='lower', cmap=cmaps.viridis,
                            #norm=colors.Normalize(
                            norm=colors.LogNorm(
                            vmin=self.vmin, vmax=self.vmax))
        
        xticks_pos = np.arange(1.5, self.Nx + 0.51, 1.)
        xticks_label = self.C
        plt.xticks(xticks_pos, xticks_label, rotation='vertical')
        
        yticks_pos = np.arange(1.5, self.Ny + 0.51, 1.)
        yticks_label = self.L_scal
        #yticks_label = self.L
        plt.yticks(yticks_pos, yticks_label)
        
    '''
    def plot_T_contours(self):
        x = np.arange(1.5, 10.6, 1.)
        y = np.arange(1.5, 15.6, 1.)
        X, Y = np.meshgrid(x, y)
        Z = self.T
        levels = np.arange(8000., 13501., 500)
        CS = plt.contour(X, Y, Z, levels, colors='white') 
        
        fmt = {}
        labels = ['T=' + str(format(lvl, '.0f')) + ' K' for lvl in levels]
        for l, label in zip(CS.levels, labels):
            fmt[l] = label
            
        plt.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=12.)
    '''

    def add_colorbar(self):
        
        #Plot the color bar.
        #formatter = LogFormatter(10, labelOnlyBase=True) 
        #ticks = np.arange(self.vmin, self.vmax, 1)
        cbar = self.fig.colorbar(self._im, orientation='vertical', pad=0.03,
                                 fraction=0.1, aspect=20)
        
        #Make 'regular' ticks from 2. to the maximum pEW value.
        #In addition, put an extra tick at the bottom for value <= 0.5 (non-
        #detections. The also rename the bottom tick to inclue the '<' symbol.
        #regular_ticks = np.arange(self.vmin + 1., self.vmax + 0.01, 1.)
        #ticks = [self.vmin] + list(regular_ticks)        
        #tick_labels = ([r'$\leq\ ' + str(self.vmin) + '$']
        #               + list(regular_ticks.astype(str)))
        #cbar.set_ticks(ticks)
        #cbar.set_ticklabels(tick_labels)

        cbar.ax.tick_params(width=1, labelsize=self.fs_label)

        #Set label.
        cbar_label = (r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                      + r'$ \ \lambda$6580')
        cbar.set_label(cbar_label, fontsize=self.fs_label)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_12d_C-scaled-grid.png',
                        format='png', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_data()
        self.plotting()
        #self.plot_T_contours()
        self.add_colorbar()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()


if __name__ == '__main__':
    Plot_C_Scaled_Grid(show_fig=True, save_fig=True)                    


