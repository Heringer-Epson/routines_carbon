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

def read_hesma(fpath):
    v, dens, C = np.loadtxt(fpath, skiprows=1, usecols=(0, 1, 19), unpack=True)
    return v * u.km / u.s, dens * u.g / u.cm**3, C
    
def get_mass(v, dens, X, time):
    r = v.to(u.cm / u.s) * time    
    vol = 4. / 3. * np.pi * r**3.
    vol_step = np.diff(vol)
    mass_step = np.multiply(vol_step, dens[1:]) / const.M_sun.to('g')
    
    #Add zero value to array so that the length is preserved.
    mass_step = np.array([0] + list(mass_step))
    
    mass_cord = np.cumsum(mass_step)
    m_X = mass_step * X
    return mass_cord, m_X


class Plot_Models(object):
    '''Note, metallicity seems to have a nearly negligible impact on the
    location of carbon
    '''
    
    def __init__(self, add_tomography=False, show_fig=True, save_fig=True):
        
        self.add_tomography = add_tomography
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.fig = plt.figure(figsize=(14.,16.))
        self.ax_top = plt.subplot(211) 
        self.ax_bot = plt.subplot(212)
        self.fs = 26.

        
        self.top_dir = './../INPUT_FILES/model_predictions/'

        self.run_make()

    def set_fig_frame(self):
        
        x_top_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_top_label = r'$X(\rm{C})$'

        x_bot_label = r'$M\ \ \rm{[M_\odot]}$'
        #y_bot_label = r'$m(\rm{C})\ \ \rm{[M_\odot]}$'
        y_bot_label = r'$X(\rm{C})$'
                
        self.ax_top.set_xlabel(x_top_label, fontsize=self.fs)
        self.ax_top.set_ylabel(y_top_label, fontsize=self.fs)
        self.ax_top.set_yscale('log')
        self.ax_top.set_xlim(5000., 30000.)
        self.ax_top.set_ylim(1.e-4, 1.1)
        self.ax_top.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)       
        self.ax_top.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax_top.tick_params('both', length=8, width=1, which='major')
        self.ax_top.tick_params('both', length=4, width=1, which='minor')
        self.ax_top.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax_top.xaxis.set_major_locator(MultipleLocator(5000.))  

        self.ax_bot.set_xlabel(x_bot_label, fontsize=self.fs)
        self.ax_bot.set_ylabel(y_bot_label, fontsize=self.fs)
        self.ax_bot.set_yscale('log')
        self.ax_bot.set_xlim(0.6, 1.45)
        self.ax_bot.set_ylim(1.e-4, 1.1)
        #self.ax_bot.set_ylim(1.e-5, 2.e-2)
        self.ax_bot.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)       
        self.ax_bot.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax_bot.tick_params('both', length=8, width=1, which='major')
        self.ax_bot.tick_params('both', length=4, width=1, which='minor')
        self.ax_bot.xaxis.set_minor_locator(MultipleLocator(0.1))
        self.ax_bot.xaxis.set_major_locator(MultipleLocator(0.2))  

    def add_tomography_models(self):

        #SN2005bl.
        '''
        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Hachinger_2005bl/models-05bl-w7e0.7/SN2005bl_p129/abuplot.dat')               
        v, m, C = np.loadtxt(fpath, skiprows=2, usecols=(1, 2, 9), unpack=True)
        
        v = v[::2]
        m = m[::2]
        C = C[::2]
        
        v = v * u.km / u.s
        m_step = np.array([0.] + list(np.diff(m)))
        m_C = np.multiply(m_step, C)         

        self.ax_top.step(v, C, ls='-', color='purple', lw=3., where='post',
                         label=r'SN 2005bl')        
        self.ax_bot.step(m, C, ls='-', color='purple', lw=3., where='post',
                         label=r'SN 2005bl')   
        '''                 
        
        #SN 2011fe.
        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Mazzali_2011fe/ejecta_layers.dat')                   
        v, m, C = np.loadtxt(fpath, skiprows=2, usecols=(1, 2, 9), unpack=True)
        v = v * u.km / u.s
        m_step = np.array([0.] + list(np.diff(m)))
        m_C = np.multiply(m_step, C)        
    
        #Plot original work.
        '''
        self.ax_top.step(v, C, ls='-', color='k', lw=3., where='post',
                         label=r'SN 2011fe')
        self.ax_bot.step(m, C, ls='-', color='k', lw=3., where='post',
                         label=r'SN 2011fe')
        '''
        
        #SN minimum condition from analysis.
        v_cut = 13000. * u.km / u.s
        v_phot = 7850. * u.km / u.s

        self.ax_top.step(v[v > v_cut], C[v > v_cut] / 2., ls='-', color='k',
                          lw=3., where='post', label=r'SN 2011fe minimal')
        self.ax_bot.step(m[v > v_cut], C[v > v_cut], ls='-', color='k',
                          lw=3., where='post', label=r'SN 2011fe minimal')
        
        #Plot upper limit based on t_exp=19days.
        C_cond = C[(v <= v_cut) & (v > v_phot)] * 0.1
        v_cond = v[(v <= v_cut) & (v > v_phot)]
        self.ax_top.plot(v_cond, C_cond, ls='--', color='k', lw=3.)
        self.ax_top.arrow(v_cond[12].value, C_cond[12], 0., -0.0004,
                          head_width=100., head_length=0.0001, fc='k', ec='k')
         
                         
    def load_Shen_2017_ddet_models(self):
        
        #models = ['0.85_5050', '1.00_5050', '1.10_5050']
        models = ['1.00_5050']
        labels = [r'DDet: $M=1M_\odot$']
        ls = ['-']
        
        for k, model in enumerate(models):
            fpath = self.top_dir + 'Shen_2017_ddet/' + model + '.dat'
            m, v, C = np.loadtxt(fpath, skiprows=2, usecols=(0, 1, 17),
                                 unpack=True)
            v = (v * u.cm / u.s).to(u.km / u.s)
            m_step = np.array([0.] + list(np.diff(m)))
            m_C = np.multiply(m_step, C)

            self.ax_top.step(v, C, ls=ls[k], color='r', lw=3., where='mid',
                             label=labels[k])
            self.ax_bot.step(m, C, ls=ls[k], color='r', lw=3.,
                             where='mid', label=labels[k])
        

    def plot_Seitenzahl_2013_ddt_models(self):

        time = 100.22 * u.s
        models = ['n100']
        labels = ['N100']
        ls = ['-.']
        
        for k, model in enumerate(models):
            fpath = (self.top_dir + 'Seitenzahl_2013_ddt/ddt_2013_' + model
                     + '_isotopes.dat')
            
            v, dens, C = read_hesma(fpath)
            m, m_C = get_mass(v, dens, C, time)
            
            self.ax_top.step(v, C, ls=ls[k], color='g', lw=3., where='post',
                             label=labels[k])
            self.ax_bot.step(m, C, ls=ls[k], color='g', lw=3.,
                             where='post', label=labels[k])

    def plot_Seitenzahl_2016_gcd_models(self):

        time = 100.22 * u.s
        fpath = self.top_dir + 'Seitenzahl_2016_gcd/gcd_2016_gcd200_isotopes.dat'
        v, dens, C = read_hesma(fpath)
        m, m_C = get_mass(v, dens, C, time)
        
        self.ax_top.step(v, C, ls='--', color='b', lw=3., where='post',
                         label='GCD200')
        self.ax_bot.step(m, C, ls='--', color='b', lw=3.,
                         where='post', label='GCD200')

        
    def plot_W7_models(self):
        
        time = (0.000231481 * u.d).to(u.s)
        fpath = self.top_dir + 'W7/W7_model.dat'
        v, dens = np.loadtxt(fpath, skiprows=2, usecols=(1, 2), unpack=True)
        v = v * u.km / u.s
        dens = 10.**dens * u.g / u.cm**3

        fpath = self.top_dir + 'W7/W7_abundances.dat'
        C = np.loadtxt(fpath, skiprows=0, usecols=(6,), unpack=True)
        m, m_C = get_mass(v, dens, C, time)

        self.ax_top.step(v, C, ls=':', color='gold', lw=3., where='post',
                         label='W7')
        self.ax_bot.step(m, C, ls=':', color='gold', lw=3.,
                         where='post', label='W7')

    def add_legend(self):
        self.ax_top.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=2) 
        plt.tight_layout()    
        plt.subplots_adjust(hspace=0.3)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_model_C.png', format='png', dpi=360)
    
    def run_make(self):
        self.set_fig_frame()
        if self.add_tomography:
            self.add_tomography_models()
        self.load_Shen_2017_ddet_models()
        self.plot_Seitenzahl_2013_ddt_models()
        self.plot_Seitenzahl_2016_gcd_models()
        self.plot_W7_models()
        self.add_legend()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
if __name__ == '__main__':
    Plot_Models(add_tomography=True, show_fig=True, save_fig=True)
    
#Try to obtain the following predictions:
#https://arxiv.org/pdf/1706.01898.pdf (Shen+ 2017)
#https://arxiv.org/pdf/1002.2173.pdf (Fink+ 2010)
