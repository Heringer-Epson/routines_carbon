#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import matplotlib.pyplot as plt
import matplotlib as mpl
import colormaps as cmaps
import cPickle
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from matplotlib import cm
from matplotlib import colors

                                        
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


class Make_Slab(object):
    """Runs a TARDIS simulation with a specified density profile to analyse and
    plot how the ion fraction of a given element depends on the density and
    temperature.    
    """
    
    def __init__(self):
        
        self.density = np.logspace(-19, -12, 8.)
        self.temperature = np.arange(5000., 20001, 1000.)


        self.velocity = np.linspace(10000., 20001, len(self.density))
        self.fig, self.ax = plt.subplots(figsize=(16,6))
        
        #Shells are defined in between the specified boundaries. Innermost
        #value is not actually used.
        self.N_shells = len(self.velocity) - 1
        
        self.atom_data_dir = os.path.join(os.path.split(tardis.__file__)[0],
          'data', 'kurucz_cd23_chianti_H_He.h5')

        self.sim = None
        self.fs = 26

        self.run_make_slab()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        
        x_label = r'$T\ \rm{[K]}$'
        y_label = r'$\rm{log}\ \rho\ \rm{[g\ cm^{-3}]}$'
        
        self.ax.set_xlabel(x_label, fontsize=self.fs)
        self.ax.set_ylabel(y_label, fontsize=self.fs)
        
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.minorticks_off()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')    

    
    def make_input_files(self):
        
        with open('./density.dat', 'w') as inp:
            inp.write('1 d\n')
            inp.write('# index velocity (km/s) density (g/cm^3)')
        
            for i, (vel, dens) in enumerate(zip(self.velocity, self.density)):
                inp.write('\n' + str(i) + '    ' + str(vel) + '    ' + str(dens))
            
        
        abun_line = ['0.0'] * 5 + ['1.0'] + ['0.0'] * 24
        
        with open('./abundance.dat', 'w') as inp:
            inp.write('# index Z=1 - Z=30\n')        
            for i in range(len(self.density)):
                inp.write('\n' + str(i))
                for abun in abun_line:
                    inp.write(' ' + abun)
                            
    def create_ejecta(self):
        self.sim = tardis.run_tardis('./slab.yml', self.atom_data_dir)

    def collect_states(self):
        
        D = {}

        ion_I = []
        lvl_10 = []
        lvl_11 = []

        #print self.sim.plasma.ion_number_density.loc[6,1]
        #print self.sim.model.v_inner
        #print self.sim.model.t_inner
        print self.sim.model.t_rad
        print self.sim.model.density
        #print dir(self.sim.plasma)
        for i, T in enumerate(self.temperature):
            self.sim.plasma.update(t_rad=np.ones(self.N_shells) * T * u.K)
            
            for j in range(self.N_shells):
                level_number = (self.sim.plasma.level_number_density.loc[6,1,10][j]
                                + self.sim.plasma.level_number_density.loc[6,1,11][j])
                
                total_ions = sum(self.sim.plasma.ion_number_density.loc[6][j])
                #print sum(self.sim.plasma.ion_number_density.loc[6][i])
                #print self.sim.plasma.ion_number_density.loc[6,1][i]
                fraction = level_number / total_ions
                #if fraction_ion_I < 1.e-4:
                #    fraction_ion_I = 1.e-4


                D['T' + str(i) + 'rho' + str(j)] = fraction
        
        #Create 1D array in the 'ciorrect' order.        
        for j in range(self.N_shells):
            for i in range(len(self.temperature)):
                ion_I.append(D['T' + str(i) + 'rho' + str(j)])
        
        ion_I = np.array(ion_I)
        print min(ion_I), max(ion_I)
        self.ion_I = np.reshape(ion_I, (self.N_shells, len(self.temperature)))
        #print self.ion_I


    def plotting(self):
                
        #imshow
        self._im = plt.imshow(self.ion_I, interpolation='none', aspect='auto',
                            extent=[1., len(self.temperature) + 1, 1., self.N_shells + 1], origin='lower',
                            cmap=cmaps.viridis,
                            norm=colors.LogNorm(vmin=1.e-12,  vmax=1.e-7))
        
        #Format ticks. Note that the range of values plotted in x (and y) is
        #defined when calling imshow, under the 'extent' argument.
        yticks_pos = np.arange(1.5, self.N_shells + 0.6, 1.)
        yticks_label = [str(int(np.log10(dens))) for dens in self.density[1:]]
                   
        plt.yticks(yticks_pos, yticks_label, rotation='vertical')
        
        xticks_pos = np.arange(1.5, len(self.temperature) + 0.6, 1.)
        xticks_label = [str(format(T, '.0f')) for T in self.temperature]
        plt.xticks(xticks_pos, xticks_label)

    def add_colorbar(self):
        
        #Plot the color bar.
        cbar = self.fig.colorbar(self._im, orientation='vertical',
                                 fraction=0.1, pad=0.03, aspect=20)
        
        #Make 'regular' ticks from 2. to the maximum pEW value.
        #In addition, put an extra tick at the bottom for value <= 0.5 (non-
        #detections. The also rename the bottom tick to inclue the '<' symbol.
        ticks = [1.e-12, 1.e-11, 1.e-10, 1.e-9, 1.e-8, 1.e-7]
        tick_labels = ([r'$\leq\ 10^{-12}$', r'$10^{-11}$', r'$10^{-10}$', r'$10^{-9}$', r'$10^{-8}$', r'$\geq\ 10^{-7}$'])
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(width=1, labelsize=self.fs)

        #Set label.
        cbar.set_label(r'Number Fraction of C$_{\rm{II}}$ at levels 10-11', fontsize=self.fs)
        
    #Add model trad v dens for different epoches.
    def add_synthetic_models(self):
        
        def dens2axis(dens):
            #Note the first shell is for density[1], note [0]
            a = ((len(self.density) - 2.) 
				 / (np.log10(self.density[-1]) - np.log10(self.density[1])))
            b = 1.5 - a * np.log10(self.density[1])
            return a * np.log10(dens) + b
        def T2axis(T):
            a = ((len(self.temperature) - 1.) / 
				 (self.temperature[-1] - self.temperature[0]))
            b = 1.5 - a * self.temperature[0]
            return a * T + b
        
        #at t_exp = 5.9 days
        case_folder = path_tardis_output + '11fe_test_5.9d/'
        fname = 'loglum-8.505'
        fname = case_folder + fname + '/' + fname + '.pkl'
        
        with open(fname, 'r') as inp:
            pkl = cPickle.load(inp)
            t_rad = pkl['t_rad'].value
            density = pkl['density'].value
            self.ax.plot(T2axis(t_rad), dens2axis(density))
        
        
    #add explosion models?  
    
    def clean_up(self):
        os.remove('./abundance.dat')
        os.remove('./density.dat')
        
    def run_make_slab(self):
        self.set_fig_frame()
        self.make_input_files()
        self.create_ejecta()
        self.collect_states()
        self.plotting()
        self.add_colorbar()
        self.add_synthetic_models()
        self.clean_up()
        plt.tight_layout()
        plt.show()
        
        
        
    
Make_Slab()
