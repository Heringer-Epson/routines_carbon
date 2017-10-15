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

import colormaps as cmaps
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

class Plot_L_Fe(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, lm='downbranch', interp='none',
                 show_fig=True, save_fig=False):

        self.lm = lm
        self.interp = interp
        self.show_fig = show_fig
        self.save_fig = save_fig 

        L_scal = np.arange(0.6, 2.01, 0.1)
        self.L = [lum2loglum(0.23e9 * l) for l in L_scal]                
        
        self.Fe = ['0.00', '0.0001', '0.0002', '0.0005', '0.001',
                   '0.002', '0.005', '0.01', '0.02', '0.05']
        
        self.C = None
        self.norm = None
        self._im = None
        self.pEW_max = None 
        
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
        
        x_label = r'$X(\rm{Fe})$'
        y_label = r'$L\ /\ {\mathrm{11fe}}$'
        
        self.ax.set_xlabel(x_label, fontsize=self.fs_label)
        self.ax.set_ylabel(y_label, fontsize=self.fs_label)
        
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')    

    def load_data(self):
        
        case_folder = path_tardis_output + 'hypo-carbon-grid/'
        def get_fname(lm, L, Fe): 
            Si = str(0.60 - float(Fe))
            fname = 'loglum-' + L + '_line_interaction-' + lm + '_Fes-' + Fe
            fname = 'abun_Fe-' + Fe + '_abun_Si-' + Si + '_loglum-' + L
            fname = case_folder + fname + '/' + fname + '.pkl'
            return fname     
            
        carbon_1D, T_inner = [], []
        #[::-1] in L is because we want the top row to contain the highest L,
        #since imshow (I think) plot the first row on top.
        for L in self.L:
            for Fe in self.Fe:
                with open(get_fname(self.lm, L, Fe), 'r') as inp:
                    pkl = cPickle.load(inp)
                    pEW = pkl['pEW_fC']
                    if np.isnan(pEW) or pEW < 0.5:
                        pEW = 0.5                    
                    carbon_1D.append(pEW)
                    T_inner.append(pkl['t_inner'].value)

        carbon_1D = np.array(carbon_1D)
        carbon_1D = np.nan_to_num(carbon_1D)
        self.pEW_max = np.ceil(max(carbon_1D))
        self.C = np.reshape(carbon_1D, (15, 10))
        self.T = np.reshape(T_inner, (15, 10))

    def plotting(self):
                
        #imshow
        self._im = plt.imshow(self.C, interpolation=self.interp, aspect='auto',
                            extent=[1., 11., 1., 16.], origin='lower',
                            cmap=cmaps.viridis,
                            norm=colors.Normalize(vmin=0.5, vmax=self.pEW_max))
        
        #Format ticks. Note that the range of values plotted in x (and y) is
        #defined when calling imshow, under the 'extent' argument.
        xticks_pos = np.arange(1.5, 10.6, 1.)
        xticks_label = self.Fe
                   
        plt.xticks(xticks_pos, xticks_label, rotation='vertical')
        
        yticks_pos = np.arange(1.5, 15.6, 1.)
        yticks_label = [str(format(10.**float(L) / 0.23e9, '.2f'))
                        for L in self.L]
        plt.yticks(yticks_pos, yticks_label)        
        
    def plot_T_contours(self):
        x = np.arange(1.5, 10.6, 1.)
        y = np.arange(1.5, 15.6, 1.)
        X, Y = np.meshgrid(x, y)
        Z = self.T
        levels = np.arange(8000., 12001., 500)
        CS = plt.contour(X, Y, Z, levels, colors='white') 
        
        fmt = {}
        labels = ['T=' + str(format(lvl, '.0f')) + ' K' for lvl in levels]
        for l, label in zip(CS.levels, labels):
            fmt[l] = label
            
        plt.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=12.)

    def add_colorbar(self):
        
        #Plot the color bar.
        cbar = self.fig.colorbar(self._im, orientation='vertical',
                                 fraction=0.1, pad=0.03, aspect=20)
        
        #Make 'regular' ticks from 2. to the maximum pEW value.
        #In addition, put an extra tick at the bottom for value <= 0.5 (non-
        #detections. The also rename the bottom tick to inclue the '<' symbol.
        regular_ticks = np.arange(2., self.pEW_max + 0.1, 2.)
        ticks = [0.5] + list(regular_ticks)
        tick_labels = ([r'$\leq\ 0.5$'] + list(regular_ticks.astype(str)))
        
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(width=1, labelsize=self.fs_label)

        #Set label.
        cbar.set_label(r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'
                       + r'$ \ \lambda$6580', fontsize=self.fs_label)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_hypo-grid_' + self.lm 
                        + '.pdf', format='pdf', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_data()
        self.plotting()
        self.plot_T_contours()
        self.add_colorbar()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
Plot_L_Fe(lm='downbranch', interp='none' ,show_fig=True, save_fig=True)
#Plot_L_Fe(lm='macroatom', interp='none' ,show_fig=True, save_fig=True)
#Plot_L_Fe(lm='downbranch', interp='spline16' ,show_fig=True, save_fig=True)






