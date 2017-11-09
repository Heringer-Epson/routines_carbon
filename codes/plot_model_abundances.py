#!/usr/bin/env python

import os                                                               
import sys
import itertools                                                        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import constants as const
from scipy.integrate import cumtrapz

M_sun = const.M_sun.to('g').value

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

        self.models = ['0.85_5050', '1.00_5050', '1.10_5050', 'N100', 'W7', 
                       'Fink_model_3']
        #Create and initialize dict.
        self.D = {}
        for model in self.models:
            for key in ['vel', 'm', 'C']:
                self.D[model + '_' + key] = []
        
        self.top_dir = './../INPUT_FILES/model_predictions/'

        self.run_make()

    def set_fig_frame(self):
        
        x_top_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_top_label = r'$X(\rm{C})$'

        x_bot_label = r'$M\ \ \rm{[M_\odot]}$'
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
        self.ax_bot.set_xlim(0.2
        , 1.5)
        self.ax_bot.set_ylim(1.e-4, 1.1)
        self.ax_bot.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)       
        self.ax_bot.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax_bot.tick_params('both', length=8, width=1, which='major')
        self.ax_bot.tick_params('both', length=4, width=1, which='minor')
        self.ax_bot.xaxis.set_minor_locator(MultipleLocator(0.1))
        self.ax_bot.xaxis.set_major_locator(MultipleLocator(0.2))  
        
    def load_W7_N100_data(self):
        
        for model in ['W7', 'N100']:
            
            dens = []
            fpath = self.top_dir + model + '/' + model + '_model.dat'
            with open(fpath, 'r') as inp:
                header1 = inp.readline()
                time = float(inp.readline()) * 3600. * 24.
                for line in inp:
                    column = line.rstrip('\n').split(' ') 
                    column = filter(None, column)
                    self.D[model + '_vel'].append(float(column[1]))
                    dens.append(float(column[2]))
            
            #Compute mass.
            dens = 10.**np.array(dens)
            r_array = np.array(self.D[model + '_vel']) * time * 1.e5
            int_array = 4. * np.pi * r_array**2. * dens
            self.D[model + '_m'] = [0.] + list(cumtrapz(int_array, r_array) / M_sun)
            
            fpath = self.top_dir + model + '/' + model + '_abundances.dat'
            with open(fpath, 'r') as inp:            
                for line in inp:
                    column = line.rstrip('\n').split(' ') 
                    column = filter(None, column)
                    self.D[model + '_C'].append(float(column[6]))
                   
    def load_SubChandra_data(self):
        subC_models = ['0.85_5050', '1.00_5050_0.5Zsol', '1.00_5050',
                       '1.00_5050_2.0Zsol', '1.10_5050']
        
        for model in subC_models:
            fpath = self.top_dir + 'Sub-Chandra/' + model + '.dat'
            m, v, C = zip(*np.loadtxt(fpath, skiprows=2, usecols=(0, 1, 17)))
            self.D[model + '_m'] = m
            self.D[model + '_vel'] = np.array(v) / 1.e5
            self.D[model + '_C'] = C

    def load_Fink_DDet_data(self):
        fpath = self.top_dir + 'DDet_Fink_2010/abundance_C.csv'
        v, C = zip(*np.loadtxt(fpath, skiprows=0,  delimiter=',',))
        self.D['Fink_model_3_vel'] = v 
        self.D['Fink_model_3_C'] = C
        self.D['Fink_model_3_m'] = np.zeros(len(v))
        
    def add_tomography_models(self):

        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Mazzali_2011fe/ejecta_layers.dat')                   
        v_11fe, m_11fe, C_11fe = [], [], []
        with open(fpath, 'r') as inp:
            for line in itertools.islice(inp, 2, None, 1):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                v_11fe.append(column[1])  
                m_11fe.append(column[2])  
                C_11fe.append(column[9])  
        self.ax_top.step(v_11fe, C_11fe, ls='-', color='k', label=r'SN 2011fe',
                 lw=3., where='post')
        self.ax_bot.step(m_11fe, C_11fe, ls='-', color='k', lw=3.,
                         where='post', label=r'SN 2011fe')

        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Hachinger_2005bl/models-05bl-w7e0.7/SN2005bl_p129/abuplot.dat')               
        v_05bl, m_05bl, C_05bl = [], [], []
        with open(fpath, 'r') as inp:
            for line in itertools.islice(inp, 2, None, 2):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                v_05bl.append(column[1])  
                m_05bl.append(column[2])  
                C_05bl.append(column[9])  
        self.ax_top.step(v_05bl, C_05bl, ls='-', color='purple',
                         lw=3., where='post', label=r'SN 2005bl')        
        self.ax_bot.step(m_05bl, C_05bl, ls='-', color='purple',
                         lw=3., where='post', label=r'SN 2005bl')     
                                     
    def plot_model_predictions(self):
        colors = ['g', 'g', 'g', 'r', 'b', 'orange']
        ls = ['--', '-', '-.', '-', '-', ':']
        labels = [r'Shen DDet: $M=0.85M_\odot$', r'Shen DDet: $M=M_\odot$',
                  r'Shen DDet: $M=1.1M_\odot$', 'Artis - DDet?', 'W7',
                  r'Fink DDet: $M=1.03M_\odot$']
        where = ['mid'] * 3 + ['post'] * 2 + ['mid']
        for i, model in enumerate(self.models):
            self.ax_top.step(self.D[model + '_vel'], self.D[model + '_C'], ls=ls[i],
                         color=colors[i], label=labels[i], lw=3., where=where[i])         
            self.ax_bot.step(self.D[model + '_m'], self.D[model + '_C'],
                             ls=ls[i], color=colors[i], label=labels[i],
                             lw=3., where=where[i])         

    def add_legend(self):
        self.ax_bot.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=2) 
        plt.tight_layout()    
        plt.subplots_adjust(hspace=0.3)

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_model_C.png', format='png', dpi=360)
    
    def run_make(self):
        self.set_fig_frame()
        self.load_W7_N100_data()
        self.load_SubChandra_data()
        self.load_Fink_DDet_data()
        if self.add_tomography:
            self.add_tomography_models()
        self.plot_model_predictions()
        self.add_legend()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
if __name__ == '__main__':
    Plot_Models(add_tomography=True, show_fig=True, save_fig=True)
    
#Try to obtain the following predictions:
#https://arxiv.org/pdf/1706.01898.pdf (Shen+ 2017)
#https://arxiv.org/pdf/1002.2173.pdf (Fink+ 2010)
