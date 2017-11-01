#!/usr/bin/env python

import os                                                               
import sys
import itertools                                                        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

class Plot_Models(object):
    '''Note, metallicity seems to have a nearly negligible impact on the
    location of carbon
    '''
    
    def __init__(self, add_tomography=False, show_fig=True, save_fig=True):
        
        self.add_tomography = add_tomography
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.models = ['0.85_5050', '1.00_5050', '1.10_5050', 'N100', 'W7']
        #Create and initialize dict.
        self.D = {}
        for model in self.models:
            for key in ['vel', 'C']:
                self.D[model + '_' + key] = []
        
        self.fs = 26.
        self.fig, self.ax = plt.subplots(figsize=(14,8))
        self.top_dir = './../INPUT_FILES/model_predictions/'

        self.run_make()

    def set_fig_frame(self):
        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = r'$X(\rm{C})$'
        self.ax.set_xlabel(x_label, fontsize=self.fs)
        self.ax.set_ylabel(y_label, fontsize=self.fs)
        self.ax.set_yscale('log')
        self.ax.set_xlim(3500., 30000.)
        self.ax.set_ylim(1.e-4, 1.1)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(5000.))  
        
    def load_W7_N100_data(self):
        
        for model in ['W7', 'N100']:
            
            fpath = self.top_dir + model + '/' + model + '_model.dat'
            with open(fpath, 'r') as inp:
                header1, header2 = next(inp), next(inp)
                for line in inp:
                    column = line.rstrip('\n').split(' ') 
                    column = filter(None, column)
                    self.D[model + '_vel'].append(float(column[1]))

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
            
            v, C = zip(*np.loadtxt(fpath, skiprows=2, usecols=(1, 17)))
            self.D[model + '_vel'], self.D[model + '_C'] = np.array(v) / 1.e5, C

    def add_tomography_models(self):

        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Mazzali_2011fe/ejecta_layers.dat')                   
        v_11fe, C_11fe = [], []
        with open(fpath, 'r') as inp:
            for line in itertools.islice(inp, 2, None, 1):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                v_11fe.append(column[1])  
                C_11fe.append(column[9])  
        self.ax.step(v_11fe, C_11fe, ls='-', color='k', label=r'SN 2011fe', lw=3.,
                where='post')

        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Hachinger_2005bl/models-05bl-w7e0.7/SN2005bl_p129/abuplot.dat')               
        v_05bl, C_05bl = [], []
        with open(fpath, 'r') as inp:
            for line in itertools.islice(inp, 2, None, 2):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                v_05bl.append(column[1])  
                C_05bl.append(column[9])  
        self.ax.step(v_05bl, C_05bl, ls='-', color='purple', label=r'SN 2005bl',
                lw=3., where='post')        
            
    def plot_model_predictions(self):
        colors = ['g', 'g', 'g', 'r', 'b']
        ls = ['--', '-', '-.', '-', '-', ]
        labels = [r'DDet: $M=0.85M_\odot$', r'DDet: $M=M_\odot$',
                  r'DDet: $M=1.1M_\odot$', 'N100', 'W7']
        where = ['mid'] * 3 + ['post'] * 2
        for i, model in enumerate(self.models):
            self.ax.step(self.D[model + '_vel'], self.D[model + '_C'], ls=ls[i],
                         color=colors[i], label=labels[i], lw=3., where=where[i])         

    def add_legend(self):
        self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=2) 
        plt.tight_layout()          

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_model_C.png', format='png', dpi=360)
    
    def run_make(self):
        self.set_fig_frame()
        self.load_W7_N100_data()
        self.load_SubChandra_data()
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
