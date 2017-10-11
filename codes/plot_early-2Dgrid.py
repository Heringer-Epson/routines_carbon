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
    
    def __init__(self, case='C', quantity='pEW', lm='downbranch',
                 show_fig=True, save_fig=False):

        self.case = case
        self.quantity = quantity
        self.lm = lm
        self.show_fig = show_fig
        self.save_fig = save_fig 

        L_scal = np.arange(0.6, 2.01, 0.1)
        self.L = [lum2loglum(0.32e9 * l) for l in L_scal]                
        
        self.Fe = ['0.00', '0.05', '0.10', '0.20', '0.50', '1.00', '2.00',
                   '5.00', '10.00', '20.00']
        
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
        
        if self.case == 'C':
            self.case_folder = path_tardis_output + 'early-carbon-grid_new/'
        elif self.case == 'no-C':
            self.case_folder = path_tardis_output + 'early-carbon-grid_no-C_new/'

        self.make_plot()

    def set_quantity_vars(self):
        
        if self.quantity == 'pEW_fC':
            self.lower_limit = 0.5
            self.tick_range = [2.0, 29., 2]
            #self.tick_range = [2.0, 12., 2]
            self.title = r'11fe_5.9days_pEW_' + self.case
            self.cbar_label = (r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                       + r'$ \ \lambda$6580')
        
        elif self.quantity == 'diff_depth-fC':
            self.lower_limit = 0.01
            self.tick_range = [0.02, 0.2, 0.05]
            self.title = r'11fe_5.9days_depth_' + self.case
            self.cbar_label = r'Differential depth'

        elif self.quantity == 'Fe_trough_packets':
            self.lower_limit = 10
            self.tick_range = [50, 500, 100]
            self.title = r'11fe_5.9days_Fe-packets_' + self.case
            self.cbar_label = r'#Fe packets'

        elif self.quantity == 'C_trough_packets':
            self.lower_limit = 10
            self.tick_range = [20, 220, 20]
            self.title = r'11fe_5.9days_C-packets_' + self.case
            self.cbar_label = r'#C packets'
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        
        x_label = r'$\rm{Fe \ scaling}$'
        y_label = r'$L\ /\ L_{\mathrm{11fe}}$'
        
        self.ax.set_xlabel(x_label, fontsize=self.fs_label)
        self.ax.set_ylabel(y_label, fontsize=self.fs_label)
        
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')    

    def load_data(self):
        
        def get_fname(lm, L, Fe): 
            fname = 'loglum-' + L + '_line_interaction-' + lm + '_Fe0-' + Fe
            fname = self.case_folder + fname + '/' + fname + '.pkl'
            return fname     
            
        qtty_1D, T_inner, color_1D, pEW_1D = [], [], [], []
        for L in self.L:
            for Fe in self.Fe:
                with open(get_fname(self.lm, L, Fe), 'r') as inp:
                    pkl = cPickle.load(inp)
                    
                    pEW = pkl['pEW_fC']
                    if np.isnan(pEW):
                        pEW = 0. 
                    pEW_1D.append(pEW)
                    
                    qtty = pkl[self.quantity]
                    #qtty = sum(pkl[self.quantity])
                    if np.isnan(qtty) or qtty < self.lower_limit:
                        qtty = self.lower_limit   
                    qtty_1D.append(qtty)
                    
                    T_inner.append(pkl['t_inner'].value)
                    color_1D.append(pkl['filter_Johnson-U']
                                    - pkl['filter_Johnson-V'])
                                    
        qtty_1D = np.array(qtty_1D)
        #qtty_1D = np.nan_to_num(qtty_1D)
        
        pEW_1D = np.array(pEW_1D)
        #pEW_1D = np.nan_to_num(pEW_1D)
                
        
        self.qtty = np.reshape(qtty_1D, (15, 10))
        self.T = np.reshape(T_inner, (15, 10))
        self.color = np.reshape(color_1D, (15, 10))
        #self.pEW = np.reshape(pEW_1D, (15, 10))
        self.pEW = pEW_1D

    def plotting(self):
                
        #imshow
        self._im = plt.imshow(self.qtty, interpolation='none', aspect='auto',
                            extent=[1., 11., 1., 16.], origin='lower',
                            cmap=cmaps.viridis,
                            norm=colors.Normalize(vmin=self.lower_limit,
                                                  vmax=self.tick_range[1]))
        
        #Format ticks. Note that the range of values plotted in x (and y) is
        #defined when calling imshow, under the 'extent' argument.
        xticks_pos = np.arange(1.5, 10.6, 1.)
        xticks_label = self.Fe
                   
        plt.xticks(xticks_pos, xticks_label, rotation='vertical')
        
        yticks_pos = np.arange(1.5, 15.6, 1.)
        yticks_label = [str(format(10.**float(L) / 0.32e9, '.2f'))
                        for L in self.L]
        yticks_label = self.L
        #yticks_label = yticks_label
        plt.yticks(yticks_pos, yticks_label)
        
        #self.ax.set_title(self.title, fontsize=26)                  

    def plot_color_contours(self):
        x = np.arange(1.5, 10.6, 1.)
        y = np.arange(1.5, 15.6, 1.)
        X, Y = np.meshgrid(x, y)
        Z = self.color
        levels = np.arange(-1., 3., 0.5)
        CS = plt.contour(X, Y, Z, levels, ls='--', colors='red') 
        
        fmt = {}
        labels = ['U-B=' + str(format(lvl, '.1f')) for lvl in levels]
        for l, label in zip(CS.levels, labels):
            fmt[l] = label
            
        plt.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=12.)

        for c in CS.collections:
            c.set_linestyle('dashed') 
        
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

    def plot_hatched_region(self):
        i = 0
        for l in range(len(self.L)):
            for f in range(len(self.Fe)):
                if self.pEW[i] < 2.:
                    self.ax.add_patch(
                      mpl.patches.Rectangle((f + 1, l + 1), 1., 1., hatch='//',
                      fill=False, snap=False, lw=0.))
                i += 1


    def add_colorbar(self):
        
        #Plot the color bar.
        cbar = self.fig.colorbar(self._im, orientation='vertical',
                                 fraction=0.1, pad=0.03, aspect=20)
        
        #Make 'regular' ticks from 2. to the maximum pEW value.
        #In addition, put an extra tick at the bottom for value <= 0.5 (non-
        #detections. The also rename the bottom tick to inclue the '<' symbol.
        regular_ticks = np.arange(
          self.tick_range[0], self.tick_range[1] + 0.001, self.tick_range[2])
        ticks = [self.lower_limit] + list(regular_ticks)
        tick_labels = ([r'$\leq\ ' + str(self.lower_limit) + '$']
                       + list(regular_ticks.astype(str)))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(width=1, labelsize=self.fs_label)

        #Set label.
        cbar.set_label(self.cbar_label, fontsize=self.fs_label)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_early-grid_' + self.case + '_'
                        + self.quantity + '_new.png', format='png')

    def make_plot(self):
        self.set_quantity_vars()
        self.set_fig_frame()
        self.load_data()
        self.plotting()
        #self.plot_color_contours()
        self.plot_T_contours()
        #self.plot_hatched_region()
        self.add_colorbar()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
#Plot_L_Fe(case='C', quantity='diff_depth-fC', lm='macroatom', show_fig=True, save_fig=True)
Plot_L_Fe(case='C', quantity='pEW_fC', lm='macroatom', show_fig=True, save_fig=True)
#Plot_L_Fe(case='C', quantity='Fe_trough_packets', lm='macroatom', show_fig=True, save_fig=False)
#Plot_L_Fe(case='C', quantity='C_trough_packets', lm='macroatom', show_fig=True, save_fig=False)






