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
from astropy.analytic_functions import blackbody_lambda
from astropy import units as u
from astropy import constants as const
from scipy.optimize import curve_fit
import math
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

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

class Plot_Blackbody(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, lm, L, dens, show_fig=True, save_fig=False):

        self.lm = lm
        self.L = L
        self.dens = dens
        self.show_fig = show_fig
        self.save_fig = save_fig 
        
        self.T = None
        
        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
    
        self.make_plot()
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""

        self.FIG = plt.figure(figsize=(8,12))
        self.ax_top = plt.subplot(211) 
        self.ax_bot = plt.subplot(212, sharex=self.ax_top)
        
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label_top = r'$\mathrm{Relative \ f}_{\lambda}$'
        y_label_bot = r'$\partial \mathrm{f}_{\lambda}$'
        
        self.ax_bot.set_xlabel(x_label, fontsize=self.fs_label)
        self.ax_bot.set_ylabel(y_label_bot, fontsize=self.fs_label)
        self.ax_top.set_ylabel(y_label_top, fontsize=self.fs_label)
        
        self.ax_bot.tick_params(axis='y', which='major',
                                labelsize=self.fs_ticks, pad=8)       
        self.ax_bot.tick_params(axis='x', which='major',
                                labelsize=self.fs_ticks, pad=8)
        self.ax_bot.minorticks_off()
        self.ax_bot.tick_params('both', length=8, width=1, which='major')
        self.ax_bot.tick_params('both', length=4, width=1, which='minor')    
        
        self.ax_top.tick_params(axis='y', which='major',
                                labelsize=self.fs_ticks, pad=8)       
        self.ax_top.tick_params(axis='x', which='major',
                                labelsize=self.fs_ticks, pad=8)
        self.ax_top.minorticks_off()
        self.ax_top.tick_params('both', length=8, width=1, which='major')
        self.ax_top.tick_params('both', length=4, width=1, which='minor')  

        self.ax_bot.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax_bot.xaxis.set_major_locator(MultipleLocator(1000.))

        #self.ax_bot.set_xlim(4000.,8000.)
        self.ax_bot.set_ylim(-.1, .1)

    def load_data(self):

        lum = lum2loglum(self.L)
        t_exp = dens2texp(0.01, 100., self.dens)

        with open(get_fname(self.lm, lum, t_exp, False), 'r') as inp:
            pkl = cPickle.load(inp)
            self.T = pkl['t_inner']
            self.r = pkl['r_inner']
            self.flux_r = pkl['flux_raw']
            self.flux_n = pkl['flux_normalized']
            self.flux_s = pkl['flux_smoothed']
            self.norm_factor = pkl['norm_factor']
            self.wavelength = pkl['wavelength_corr']
            self.w = pkl['w']
            print self.T
                                    
    def plot_spectra(self):

        #Note that TARDIS's flux here is actually a luminosity, i.e. need
        #be divided by an area.
        
        distance_10pc =  10. * u.parsec.to(u.cm)      
        area = 4. * np.pi * distance_10pc**2.
        self.flux_r /= area
        self.flux_s *= self.norm_factor / area       
        
        self.ax_top.plot(self.wavelength, self.flux_r, color='gray', ls='-',
                         lw=1., alpha=0.5)

        self.ax_top.plot(self.wavelength, self.flux_s, color='k', ls='-',
                         lw=2., alpha=1.)    

    def plot_blackbody(self):
        self.flux_bb = blackbody_lambda(self.wavelength * u.AA, self.T).value
        explosion_surface = 4. * np.pi * self.r**2.
        distance_10pc =  10. * u.parsec.to(u.cm)      
        area = 4. * np.pi * distance_10pc**2.
        #solid_angle = 4. * np.pi
        solid_angle = 1.
        self.flux_bb *= explosion_surface * solid_angle / area
        #Correct for the diluation factor
        self.flux_bb /= self.w
        
        self.ax_top.plot(self.wavelength, self.flux_bb, color='m', ls='--',
                         lw=1., alpha=1.)

    def fit_BB(self):
        T = self.T.value
        def BB(x, scaling):
            return scaling * blackbody_lambda(x * u.AA, T).value
                
        popt, pcov = curve_fit(BB, self.wavelength, self.flux_r)
        self.flux_bb = BB(self.wavelength, popt[0])
        self.ax_top.plot(self.wavelength, self.flux_bb, color='g', ls='--',
                         lw=1., alpha=1.)
    
    def fit_poly(self):
        C_window = ((self.wavelength >= 5800.) & (self.wavelength <= 6580.))
        w_window, f_window = self.wavelength[C_window], self.flux_s[C_window]
        popt = np.polyfit(w_window, f_window, 2)                                    
        flux_bb = np.polyval(popt, w_window)
        self.ax_top.plot(w_window, flux_bb, color='g', ls='--',
                         lw=3., alpha=1.)     

        diff = (f_window - flux_bb) / flux_bb
        
            
        self.ax_bot.plot(w_window, diff, color='k', ls='-', lw=1., alpha=1.)        

        self.ax_bot.plot([500., 20000.], [0., 0.], color='gray', ls='--',
                         lw=1., alpha=1.)   

        #depth = min(diff[C_window])
        
    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_hypo-grid_' + self.lm 
                        + '.pdf', format='pdf', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.load_data()
        self.plot_spectra()
        #self.plot_blackbody()
        #self.fit_BB()
        self.fit_poly()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()

#L_list = np.logspace(8., 9.7, 16)
L_list = np.logspace(7., 9.7, 16)
        
Plot_Blackbody(lm='downbranch', L=L_list[0], dens=1.e-18, show_fig=True, save_fig=False)
#Plot_Blackbody(lm='downbranch', L=L_list[0], dens=1.e-17, show_fig=True, save_fig=False)
#Plot_Blackbody(lm='downbranch', L=L_list[0], dens=1.e-9, show_fig=True, save_fig=False)
#Plot_Blackbody(lm='downbranch', L=L_list[-1], dens=1.e-17, show_fig=True, save_fig=False)
#Plot_Blackbody(lm='downbranch', L=L_list[-1], dens=1.e-9, show_fig=True, save_fig=False)





