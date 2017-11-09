#!/usr/bin/env python

import os                                                               
import sys
import time

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LogFormatter
import new_colormaps as cmaps
from astropy import units as u, constants
                                     
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

def retrieve_C_velocity(hdf):
    C_level_dens = []
    lvl_dens = pd.read_hdf(hdf, '/simulation/plasma/level_number_density')     
    t_rad = pd.read_hdf(hdf, '/simulation/model/t_radiative').values   
    v_inner = pd.read_hdf(hdf, '/simulation/model/v_inner').values
    v_inner = v_inner * (u.cm / u.s).to(u.km / u.s)    
    for j in range(len(v_inner)):
        C_level_dens.append(lvl_dens.loc[6,1,10][j]
                                 + lvl_dens.loc[6,1,11][j])
    v_at_C_max = v_inner[np.array(C_level_dens).argmax()]
    return v_at_C_max / (-1.e3)

texp2L = {'3': 0.08e9, '6': 0.32e9, '9': 1.1e9,
          '12': 2.3e9, '16': 3.2e9, '19': 3.5e9,
          '22': 3.2e9, '28': 2.3e9}

class Plot_C_Scaled_Grid(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, date, show_fig=True, save_fig=False):
        self.date = date
        self.show_fig = show_fig
        self.save_fig = save_fig 
            
        self.L_scal = np.arange(0.2, 1.61, 0.1)[1::]
        self.L = [lum2loglum(texp2L[self.date] * l) for l in self.L_scal]              
        self.C = ['0.00', '0.01', '0.02', '0.05', '0.10', '0.20',
                  '0.50', '1.00', '2.00', '5.00', '10.00', '20.00']

        self.Nx = len(self.C)
        self.Ny = len(self.L)
        self.qtty_1D, self.pEW_1D = [], []
        self.vmin, self.vmax = None, None
        
        self.pEW = None
        self.qtty = None
        self.norm = None
        self._im = None
        self.lower_limit = None 
        self.tick_range = None
        self.title = None
        self.cbar_label = None
        self.norm = None
        
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
        
        case_folder = '11fe_' + self.date + 'd_C-scaled/'
        def get_fname(C, L): 
            fname = 'loglum-' + L + '_C-F1-' + C
            fname = case_folder + fname + '/' + fname + '.pkl'
            return path_tardis_output + fname     
            
        for L in self.L:
            for C in self.C:
                fpath = get_fname(C, L)
                with open(fpath, 'r') as inp:
                    pkl = cPickle.load(inp)
                    
                    pEW = pkl['pEW_fC']
                    if np.isnan(pEW):
                        pEW = 0.5 
                    self.pEW_1D.append(pEW)                    
                    #self.qtty_1D.append(pkl['velocity_fC'])
                
                #Get velocity from C trough forming shell.
                if float(C) >= 0.2:
                    hdf_path = fpath.split('pkl')[0] + 'hdf'
                    self.qtty_1D.append(retrieve_C_velocity(hdf_path))
                else:
                    self.qtty_1D.append(np.nan)
                        

    def plotting(self):

        pEW_2D = np.reshape(self.pEW_1D, (self.Ny, self.Nx))
        self.vmin = 0.5
        self.vmax = int(max(self.pEW_1D)) + 1.
        self.norm = colors.LogNorm(vmin=self.vmin, vmax=self.vmax)
                
        #imshow
        self._im = plt.imshow(pEW_2D, interpolation='none', aspect='auto',
                            extent=[1., self.Nx + 1., 1., self.Ny + 1.],
                            origin='lower', cmap=cmaps.viridis,
                            #norm=colors.Normalize(
                            norm=self.norm)
        
        xticks_pos = np.arange(1.5, self.Nx + 0.51, 1.)
        xticks_label = self.C
        plt.xticks(xticks_pos, xticks_label, rotation='vertical')
        
        yticks_pos = np.arange(1.5, self.Ny + 0.51, 1.)
        yticks_label = self.L_scal
        #yticks_label = self.L
        plt.yticks(yticks_pos, yticks_label)
        
    def plot_qtty_contours(self):
        
        self.qtty_1D = np.asarray(self.qtty_1D) * (-1000.)
        qtty_2D = np.reshape(self.qtty_1D, (self.Ny, self.Nx))
        
        x = np.arange(1.5, self.Nx + 0.51, 1.)
        y = np.arange(1.5, self.Ny + 0.51, 1.)
        X, Y = np.meshgrid(x, y)
        Z = qtty_2D
        
        if self.date == '12':
            levels = np.arange(8000., 13501., 500)
        if self.date == '19':
            levels = [8000., 9000., 10000., 11000., 11500., 12000.]
        CS = plt.contour(X, Y, Z, levels, colors='white') 
        
        fmt = {}
        labels = ['v=' + str(format(lvl, '.0f')) + ' km/s' for lvl in levels]
        for l, label in zip(CS.levels, labels):
            fmt[l] = label
            
        plt.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=12.)

    def add_colorbar(self):
        
        #Plot the color bar.
        minorticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      20, 30, 40, 50, 60, 70, 80, 90, 100]

        tick_labels = [r'$\leq$ 0.5', '', '', '', '', '1', '2', '', '', '5',
        '', '', '', '', '10', '20', '', '', '50', '', '', '', '', '100']

        minorticks = [self.norm(t) for t in minorticks if t <= self.vmax]
        tick_labels = [tl for (tl, t) in zip(tick_labels, minorticks)
                       if t <= self.vmax]
                
        cbar = self.fig.colorbar(self._im, orientation='vertical', pad=0.03,
                                 fraction=0.1, aspect=20)
        
        cbar.ax.yaxis.set_ticks(minorticks, minor=True)                         
        cbar.ax.tick_params(width=1, labelsize=self.fs_label)
        cbar.ax.tick_params('y', length=8, width=1, which='major')
        cbar.ax.tick_params('y', length=4, width=1, which='minor')    
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.set_yticklabels(['1', '10'])

        #Set label.
        cbar_label = (r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                      + r'$ \ \lambda$6580')
        cbar.set_label(cbar_label, fontsize=self.fs_label)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_' + self.date + 'd_C-scaled-grid.png',
                        format='png', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_data()
        self.plotting()
        self.plot_qtty_contours()
        self.add_colorbar()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()

if __name__ == '__main__':
    #Plot_C_Scaled_Grid(date='12', show_fig=True, save_fig=False)                    
    Plot_C_Scaled_Grid(date='19', show_fig=True, save_fig=False)                    


