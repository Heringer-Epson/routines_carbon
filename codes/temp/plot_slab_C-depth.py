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
import pickle

from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
from scipy.optimize import curve_fit
from astropy.analytic_functions import blackbody_lambda
from astropy import units as u
from astropy import constants as const

import colormaps as cmaps
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

distance_10pc = 10. * u.parsec.to(u.cm)      
area = 4. * np.pi * distance_10pc**2.

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

def dens2texp(rho_0, t_0, rho):
    #get t_exp in seconds.
    t_exp = t_0 * (rho_0 / rho)**(1. / 3.)
    #convert to days (default unit in make_inputs).
    t_exp /= (3600. * 24.)
    #format for readability.
    t_exp = format(t_exp, '.3f') 
    return t_exp

#case_folder = path_tardis_output + 'slab_carbon-only/'
case_folder = path_tardis_output + 'slab_carbon-II/'
def get_fname(lm, L, t_exp, has_depth): 
    fname = ('loglum-' + L + '_line_interaction-' + lm
             + '_time_explosion-' + t_exp)
    fname = case_folder + fname + '/' + fname
    if has_depth:
        fname += '_up.pkl'
    else:
        fname += '.pkl'                    
    return fname    

class Plot_L_Fe(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, lm='downbranch', compute_C=False, show_fig=True,
                 save_fig=False):

        self.lm = lm
        self.compute_C = compute_C
        self.show_fig = show_fig
        self.save_fig = save_fig 

        #L_list = np.logspace(8., 9.7, 16)
        L_list = np.logspace(7., 9.7, 16)
        self.L = [lum2loglum(l) for l in L_list]     
            
        self.density = np.logspace(-18, -11.,8)
        self.t_exp = [dens2texp(0.01, 100., dens) for dens in self.density]
        
        self.fig, self.ax = plt.subplots(figsize=(10,10))

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
    
        self.make_plot()

    def get_C_depth(self):

        def compute_C_depth(w, f_s):
           
            C_window = ((w >= 5800.) & (w <= 6580.))
            w_window, f_window = w[C_window], f_s[C_window]
            popt = np.polyfit(w_window, f_window, 2)                                    
            flux_bb = np.polyval(popt, w_window) 
            diff = (f_window - flux_bb) / flux_bb
            depth = min(diff)
            
            if depth >= 0.:
                depth = np.nan
            else:
                depth = abs(depth)    
            
            return depth

        for L in self.L:
            for t in self.t_exp:
                with open(get_fname(self.lm, L, t, False), 'r') as inp:        
                    pkl = cPickle.load(inp)                  
                    wavelength = pkl['wavelength_corr']
                    flux_smoothed = pkl['flux_smoothed']
                    flux_norm_factor = pkl['norm_factor']
                    flux_smoothed *= flux_norm_factor / area
                    depth = compute_C_depth(wavelength, flux_smoothed)
                    pkl['C_depth'] = depth
                
                with open(get_fname(self.lm, L, t, True), 'w') as out:        
                    pickle.dump(pkl, out, protocol=pickle.HIGHEST_PROTOCOL)        

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        
        x_label = r'$\rm{log} \ \rho \ \rm{[g \ cm^{-3}]}$'
        y_label = r'$\rm{log} \ L \ \rm{[erg \ s^{-1}]}$'
        
        self.ax.set_xlabel(x_label, fontsize=self.fs_label)
        self.ax.set_ylabel(y_label, fontsize=self.fs_label)
        
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')    

    def load_data(self):
        
        C_depth, T_inner, density = [], [], []
        #[::-1] in L is because we want the top row to contain the highest L,
        #since imshow (I think) plot the first row on top.
        for L in self.L:
            for t, dens in zip(self.t_exp, self.density):
                with open(get_fname(self.lm, L, t, True), 'r') as inp:
                    pkl = cPickle.load(inp)                  
                    depth = pkl['C_depth']
                    T = pkl['t_inner'].value
                    
                    #Clean up spurious values, which will be plotted differently.
                    if T >= 20000. or T <= 4000.:
                        depth = np.nan
                    #if depth <= 0.02:
                    #   depth = 0.02
                    #elif depth >= 0.2:
                    #   depth = 0.2 

                    C_depth.append(depth)
                    T_inner.append(pkl['t_inner'].value)
                    
                    #Test whether dens used in tardis matches the requested one.
                    tardis_dens = pkl['density'][0].value                    
                    if (tardis_dens - dens) / dens > 0.01:
                        raise ValueError(
                          'Density in input array ("%s") and density used in '\
                          'TARDIS ("%s") do not match.\n\n'
                          %(dens, tardis_dens))  
      
        C_depth = np.array(C_depth)
        C_depth = np.fabs(C_depth)
        #C_depth = np.nan_to_num(C_depth)
        x = C_depth[~np.isnan(C_depth)]

        self.depth_max = np.ceil(max(C_depth))
        self.C = np.reshape(C_depth, (16, 8))
        self.T = np.reshape(T_inner, (16, 8))
        
    def plotting(self):
                
        #imshow
        self._im = plt.imshow(self.C, interpolation='none', aspect='auto',
                            extent=[1., 9., 1., 17.], origin='lower',
                            cmap=cmaps.viridis,
                            norm=colors.Normalize(vmin=0.01, vmax=0.1))
        
        #Format ticks. Note that the range of values plotted in x (and y) is
        #defined when calling imshow, under the 'extent' argument.
        xticks_pos = np.arange(1.5, 8.6, 1.)
        xticks_label = np.arange(-18, -10.9, 1.).astype(int)
                   
        plt.xticks(xticks_pos, xticks_label, rotation='horizontal')
        
        yticks_pos = np.arange(1.5, 16.6, 1.)
        yticks_label = self.L
        plt.yticks(yticks_pos, yticks_label)        
        
    def plot_T_contours(self):
        x = np.arange(1.5, 8.6, 1.)
        y = np.arange(1.5, 16.6, 1.)
        X, Y = np.meshgrid(x, y)
        Z = self.T
        levels = np.arange(4000., 20001., 2000)
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
        #regular_ticks = np.arange(0.2, 1.01, 0.1)
        #ticks = [0.1] + list(regular_ticks)
        #tick_labels = ([r'$\leq\ 0.1$'] + list(regular_ticks.astype(str)))
        
        #cbar.set_ticks(ticks)
        #cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(width=1, labelsize=self.fs_label)

        #Set label.
        cbar.set_label(r'pEW $\mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'
                       + r'$ \ \lambda$6580', fontsize=self.fs_label)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_slab_C-depth_' + self.lm 
                        + '.pdf', format='pdf', dpi=360)

    def make_plot(self):
        if self.compute_C:
            self.get_C_depth()
        self.set_fig_frame()
        self.load_data()
        self.plotting()
        self.plot_T_contours()
        self.add_colorbar()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
Plot_L_Fe(lm='downbranch', compute_C=True, show_fig=True, save_fig=True)







