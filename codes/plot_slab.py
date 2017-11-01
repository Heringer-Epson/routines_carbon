#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    
    def __init__(self, Z=6, lvl_list=[10, 11]):
        
        self.Z = Z
        self.lvl_list = lvl_list      
        
        self.density = 10. ** np.arange(-18.5, -11.9, 0.5)
        self.temperature = np.arange(5000., 17001, 500.)
        self.velocity = np.linspace(10000., 20001, len(self.density))
        
        self.F = {}
        self.F['fig'] = plt.figure(figsize=(12,12))
        self.F['ax1'] = plt.subplot(311) 
        self.F['ax2'] = plt.subplot(312)
        self.F['ax3'] = plt.subplot(313)
        
        #Shells are defined in between the specified boundaries. Innermost
        #value is not actually used.
        self.N_shells = len(self.velocity) - 1
        
        self.atom_data_dir = os.path.join(os.path.split(tardis.__file__)[0],
          'data', 'kurucz_cd23_chianti_H_He.h5')

        Z2el = {6: 'C', 14: 'Si', 26: 'Fe'}
        self.el = Z2el[self.Z]

        self.sim = None
        self.lvl = None
        self.ion_II = None
        self.ion_III = None
        self.fs = 26
        self.cbar_range = 1.e4

        self.run_make_slab()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$T\ \rm{[K]}$'
        y_label = r'$\rm{log}\ \rho\ \rm{[g\ cm^{-3}]}$'
        for ax in [self.F['ax1'], self.F['ax2'], self.F['ax3']]:
            ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)       
            ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
            ax.minorticks_off()
            ax.tick_params('both', length=8, width=1, which='major')
            ax.tick_params('both', length=4, width=1, which='minor')    
        self.F['ax3'].set_xlabel(x_label, fontsize=self.fs)
        self.F['ax2'].set_ylabel(y_label, fontsize=self.fs)
                
    def make_input_files(self):
        
        with open('./inputs/density.dat', 'w') as inp:
            inp.write('1 d\n')
            inp.write('# index velocity (km/s) density (g/cm^3)')
            for i, (vel, dens) in enumerate(zip(self.velocity, self.density)):
                inp.write('\n' + str(i) + '    ' + str(vel) + '    ' + str(dens))
            
        abun_line = ['0.0'] * (self.Z - 1) + ['1.0'] + ['0.0'] * (30 - self.Z)
        with open('./inputs/abundance.dat', 'w') as inp:
            inp.write('# index Z=1 - Z=30\n')        
            for i in range(len(self.density)):
                inp.write('\n' + str(i))
                for abun in abun_line:
                    inp.write(' ' + abun)
                            
    def create_ejecta(self):
        self.sim = tardis.run_tardis('./inputs/slab.yml', self.atom_data_dir)

    def collect_states(self):
        
        D = {}
        ion_II, ion_III, lvl = [], [], []

        for i, T in enumerate(self.temperature):
            self.sim.plasma.update(t_rad=np.ones(self.N_shells) * T * u.K)
            
            for j in range(self.N_shells):
                lvl_number = 0.
                for level in self.lvl_list:
                    lvl_number += (
                      self.sim.plasma.level_number_density.loc[self.Z,1,level][j])

                ion_II_number = self.sim.plasma.ion_number_density.loc[self.Z,1][j]
                ion_III_number = self.sim.plasma.ion_number_density.loc[self.Z,2][j]
                                
                total_ions = sum(self.sim.plasma.ion_number_density.loc[self.Z][j])
                                
                D['lvl_T' + str(i) + 'rho' + str(j)] = lvl_number / total_ions
                D['ion_II_T' + str(i) + 'rho' + str(j)] = ion_II_number / total_ions
                D['ion_III_T' + str(i) + 'rho' + str(j)] = ion_III_number / total_ions
        
        #Create 1D array in the 'correct' order.        
        for j in range(self.N_shells):
            for i in range(len(self.temperature)):
                lvl.append(D['lvl_T' + str(i) + 'rho' + str(j)])
                ion_II.append(D['ion_II_T' + str(i) + 'rho' + str(j)])
                ion_III.append(D['ion_III_T' + str(i) + 'rho' + str(j)])
        
        self.lvl = np.array(lvl)
        self.ion_II = np.array(ion_II)
        self.ion_III = np.array(ion_III)
        
    def plotting(self):

        for k, (ax, qtty, cmap) in enumerate(
          zip([self.F['ax1'], self.F['ax2'], self.F['ax3']],
          [self.ion_II, self.ion_III, self.lvl],
          [plt.get_cmap('Oranges'), plt.get_cmap('Blues'), plt.get_cmap('Purples')])):
          #[cmaps.inferno, cmaps.plasma, cmaps.viridis])):

            qtty_max = 10. ** int(np.log10(max(qtty)))
            qtty_min = qtty_max / self.cbar_range
            qtty = np.reshape(qtty, (self.N_shells, len(self.temperature)))
                
            #imshow
            _im = ax.imshow(
              qtty, interpolation='none', aspect='auto',
              extent=[1., len(self.temperature) + 1, 1., self.N_shells + 1],
              origin='lower', cmap=cmap,
              norm=colors.LogNorm(vmin=qtty_min,  vmax=qtty_max))
            
            #Format ticks. Note that the range of values plotted in x (and y) is
            #defined when calling imshow, under the 'extent' argument.
            yticks_pos = np.arange(1.5, self.N_shells + 0.6, 1.)
            yticks_label = [''] * self.N_shells
            for i in xrange(0, self.N_shells, 2):
                yticks_label[i] = int(np.log10(self.density[i + 1]))
            ax.set_yticks(yticks_pos)
            ax.set_yticklabels(yticks_label, rotation='vertical')
            
            xticks_pos = np.arange(1.5, len(self.temperature) + 0.6, 1.)
            if k == 2:
                xticks_label = [''] * len(self.temperature)
                for i in xrange(0, len(self.temperature), 4):
                    xticks_label[i] = int(self.temperature[i])
            else:
                xticks_label = []        
            ax.set_xticks(xticks_pos)
            ax.set_xticklabels(xticks_label, rotation='vertical')
            
            cbar = self.F['fig'].colorbar(_im, orientation='vertical', pad=0.01,
                                          fraction=0.15, aspect=10, ax=ax)

            ticks = np.logspace(np.log10(qtty_max / self.cbar_range),
                                np.log10(qtty_max), np.log10(self.cbar_range) + 1)            
            tick_labels = [r'$10^{' + str(int(np.log10(v))) + '}$' for v in ticks]
            tick_labels[0] = r'$\leq\ 10^{' + str(int(np.log10(ticks[0]))) + '}$'
            #tick_labels[-1] = r'$\geq\ 10^{' + str(int(np.log10(ticks[-1]))) + '}$'
                        
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(width=1, labelsize=self.fs)
            
            #labels = [
            #  r'n$(\\rm{' + self.el + '_{II}})/n(\\rm{' + self.el + '})$',
            #  r'n$(\\rm{' + self.el + '_{III}})/n(\\rm{' + self.el + '})$',
            #  r'n$(\\rm{' + self.el + '_{III}^{n=10-11}})/n(\\rm{' + self.el + '})$']

            #Make part of label string containing level info.
            if len(self.lvl_list) == 1:
                lvl_str = str(self.lvl_list[0])
            else:
                lvl_str = str(self.lvl_list[0]) + '-' + str(self.lvl_list[-1])
            
            #Make label string containing the denominator.
            den_str = '/\mathrm{n}(\mathrm{' + self.el + '})$'
            
            
            labels = [
              r'n$(\mathrm{' + self.el + '_{II}})' + den_str,
              r'n$(\mathrm{' + self.el + '_{III}})' + den_str,
              r'n$(\mathrm{' + self.el + '_{II}^{' + lvl_str + '}})' + den_str]
                        
            cbar.set_label(labels[k], fontsize=self.fs)

    def dens2axis(self, dens):
        #Note the first shell is for density[1], note [0]
        a = ((len(self.density) - 2.) 
             / (np.log10(self.density[-1]) - np.log10(self.density[1])))
        b = 1.5 - a * np.log10(self.density[1])
        return a * np.log10(dens) + b
    
    def T2axis(self, T):
        a = ((len(self.temperature) - 1.) / 
             (self.temperature[-1] - self.temperature[0]))
        b = 1.5 - a * self.temperature[0]
        return a * T + b
        
    def clean_up(self):
        os.remove('./inputs/abundance.dat')
        os.remove('./inputs/density.dat')
        
    def run_make_slab(self):
        self.set_fig_frame()
        self.make_input_files()
        self.create_ejecta()
        self.collect_states()
        self.plotting()
        self.clean_up()
        plt.tight_layout()
        
if __name__ == '__main__': 

    save_fig = True
    
    case_folder = path_tardis_output + '11fe_default_L-scaled_UP/'
    L_list = ['8.942', '9.063', '9.243', '9.544']
    f1 = 'line_interaction-downbranch_loglum-'
    f2 = '_velocity_start-7850_time_explosion-19.1'
    syn_list = [case_folder + (f1 + L + f2) + '/' + (f1 + L + f2) + '.pkl'
                for L in L_list]

    slab = Make_Slab(Z=6, lvl_list=[10, 11])
    #slab = Make_Slab(Z=14, lvl_list=[7]) #Weak Si feature
    #slab = Make_Slab(Z=14, lvl_list=[15]) #Strong Si feature
    #slab = Make_Slab(Z=26, lvl_list=[1, 6, 9, 11, 13, 14])

    for ax in [slab.F['ax1'], slab.F['ax2'], slab.F['ax3']]:
        for fpath in syn_list:
            with open(fpath, 'r') as inp:
                pkl = cPickle.load(inp)
                t_rad = pkl['t_rad'].value
                density = pkl['density'].value
                ax.plot(slab.T2axis(t_rad), slab.dens2axis(density),
                        color='k', ls='-', lw=2.)
            
                #Add reference points at given velocities
                #i=1 : v~8000km/s, i=20 : v~11500km/s
                #In the future, improve this to accept velocity list as input.
                ax.plot(slab.T2axis(t_rad[1]), slab.dens2axis(density[1]),
                        color='k', ls='None', marker='o',
                        markersize=8.)
                ax.plot(slab.T2axis(t_rad[20]), slab.dens2axis(density[20]),
                        color='k', ls='None', marker='o',
                        markersize=8.)

    if save_fig:
        directory = './../OUTPUT_FILES/FIGURES/'
        plt.savefig(directory + 'Fig_slab_' + slab.el + '.png', format='png', dpi=360)

    plt.show()

