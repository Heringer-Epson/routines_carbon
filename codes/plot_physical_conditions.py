#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import cPickle
from binning import make_bin
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from astropy import units as u
from matplotlib import cm
from matplotlib import colors

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

L_list = ['8.505', '9.041', '9.362', '9.505', '9.544']
t_list = ['5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['6', '9', '12', '16', '19']
v_list = ['12400', '11300', '10700', '9000', '7850']

#05bl models
L_05bl = ['8.520', '8.617', '8.745', '8.861', '8.594']
t_05bl = ['11.0', '12.0', '14.0', '21.8', '29.9']
v_05bl = ['8350', '8100', '7600', '6800', '3350']

color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
marker = ['s', 'p', '^', 'o', 'D']

mark_velocities = [12500., 13300., 16000.] * u.km / u.s
vel_markers = ['x', '+']

fs = 20

class Make_Slab(object):
    """Runs a TARDIS simulation with a specified density profile to analyse and
    plot how the ion fraction of a given element depends on the density and
    temperature.    
    """
    
    def __init__(self, syn_list, syn_05bl, Z=6, lvl_list=[10, 11], 
                 show_fig=True, save_fig=False):
        
        self.syn_list = syn_list
        self.syn_05bl = syn_05bl
        self.Z = Z
        self.lvl_list = lvl_list
        self.show_fig = show_fig
        self.save_fig = save_fig      
        
        self.density = 10. ** np.arange(-17.5, -11.9, 0.5)
        self.temperature = 10. ** np.arange(3.75, 4.251, 0.025)
        #self.temperature = np.arange(6000, 17000.1, 500.)
        self.velocity = np.linspace(10000., 20001, len(self.density))
        self.vel_cb_center = None

        self.F = {}
        self.D = {}
        self.fig = plt.figure(figsize=(10,14))
        self.ax_top = plt.subplot(211) 
        self.ax_bot = plt.subplot(212) 
        
        #Shells are defined in between the specified boundaries. Innermost
        #value is not actually used. Therefore density[0] not used.
        #Note, however, that each T is set constant across the pseudo ejecta
        #and therefore all value (including [0]) are used.
        self.N_shells = len(self.velocity) - 1
        
        self.atom_data_dir = os.path.join(os.path.split(tardis.__file__)[0],
          'data', 'kurucz_cd23_chianti_H_He.h5')

        Z2el = {6: 'C', 14: 'Si', 26: 'Fe'}
        self.el = Z2el[self.Z]

        self.sim = None
        self.lvl = None
        self.ion_II = None
        self.ion_III = None
        self.cbar_range = 1.e4

        #Make part of label string containing level info.
        if len(self.lvl_list) == 1:
            self.lvl_str = str(self.lvl_list[0])
        else:
            self.lvl_str = str(self.lvl_list[0]) + '-' + str(self.lvl_list[-1])

        self.run_make_slab()

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""

        x_label_top = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label_top = (r'$m(\rm{' + self.el + '^{+}_{n=' + self.lvl_str + '}})'
                       + '\ \ \mathrm{[M_\odot]}$')

        x_label_bot = r'$\rm{log}\ T\ \rm{[K]}$'
        y_label_bot = r'$\rm{log}\ \rho\ \rm{[g\ cm^{-3}]}$'       

        self.ax_top.set_xlabel(x_label_top, fontsize=fs)
        self.ax_top.set_ylabel(y_label_top, fontsize=fs)
        self.ax_top.set_yscale('log')
        self.ax_top.set_xlim(8500., 15000.)
        self.ax_top.set_ylim(1.e-14, 1.e-11)
        self.ax_top.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_top.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_top.tick_params('both', length=8, width=1, which='major')
        self.ax_top.tick_params('both', length=4, width=1, which='minor')
        self.ax_top.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax_top.xaxis.set_major_locator(MultipleLocator(1000.)) 
        
        self.ax_bot.set_xlabel(x_label_bot, fontsize=fs)
        self.ax_bot.set_ylabel(y_label_bot, fontsize=fs)
        self.ax_bot.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_bot.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_bot.minorticks_off()
        self.ax_bot.tick_params('both', length=8, width=1, which='major')
        self.ax_bot.tick_params('both', length=4, width=1, which='minor')    

    def retrieve_number_dens(self):
        
        for i, syn in enumerate(self.syn_list):
            
            self.D[str(i) + '_lvldens'] = []
            self.D[str(i) + '_Cdens'] = []
            self.D[str(i) + '_iondens'] = []
            
            lvldens = pd.read_hdf(
              syn + '.hdf', '/simulation/plasma/level_number_density')     
            numdens = pd.read_hdf(
              syn + '.hdf', '/simulation/plasma/number_density')
            iondens = pd.read_hdf(
              syn + '.hdf', '/simulation/plasma/ion_number_density')
            self.D[str(i) + '_vinner'] = (pd.read_hdf(
              syn + '.hdf', '/simulation/model/v_inner').values * u.cm /
              u.s).to(u.km / u.s)
            self.D[str(i) + '_density'] = (pd.read_hdf(
              syn + '.hdf', '/simulation/plasma/density')).values / u.cm**3 * u.g 

            for j in range(len(self.D[str(i) + '_vinner'])):
               
                #Get level density per shell.
                lvl_dens_app = 0.
                for level in self.lvl_list:
                    lvl_dens_app += lvldens.loc[self.Z,1,level][j]                
                
                self.D[str(i) + '_lvldens'].append(lvl_dens_app)                
                #Either way below leads to the same answer.
                self.D[str(i) + '_Cdens'].append(numdens.loc[self.Z][j])
                #self.D[str(i) + '_Cdens'].append(sum(iondens.loc[self.Z][j]))
                #print i, (lvldens.loc[6,1,10][j] + lvldens.loc[6,1,11][j]) / sum(iondens.loc[6][j])
                #print i, (lvldens.loc[6,1,10][j] + lvldens.loc[6,1,11][j]) / numdens.loc[6][j]
            self.D[str(i) + '_lvldens'] = np.asarray(self.D[str(i) + '_lvldens'])
            self.D[str(i) + '_Cdens'] = np.asarray(self.D[str(i) + '_Cdens'])

    def get_C_mass(self):

        fb = 1. * u.km / u.s
        cb = 200. * u.km / u.s
        A = {}
        for i in range(len(self.syn_list)):

            time = float(t_list[i]) * u.day
            A['vel'] = self.D[str(i) + '_vinner']
            A['dens'] = self.D[str(i) + '_density']
            A['Cdens'] = self.D[str(i) + '_Cdens'] * u.g * u.u.to(u.g) * 12 / u.cm**3
            A['lvldens'] = self.D[str(i) + '_lvldens'] * u.g * u.u.to(u.g) * 12 / u.cm**3 
                        
            #Re-bin the data and compute masses.
            for qtty in ['dens', 'Cdens', 'lvldens']:
            
                vel_cb,\
                self.D['m_' + qtty + str(i) + '_cb'],\
                self.D['m_' + qtty + str(i) + '_i'],\
                self.D['m_' + qtty + str(i) + '_o'],\
                self.D['m_' + qtty + str(i) + '_t'] =\
                make_bin(A['vel'], A[qtty], time, fb, cb)

            #All quantities should have the same coarse binning, so any works
            #for plotting.
            self.vel_cb_center = (vel_cb.value[0:-1] + vel_cb.value[1:]) / 2.

    def plotting_top(self):

        #Use any of the epochs to plot the mass per bin. Note that tihs is
        #conserved because the explosion is in homologous expnasion.
        #Note that earlier epochs will be cut below the photosphere.
        self.ax_top.plot(
          self.vel_cb_center, self.D['m_dens4_cb'] * 1.e-10,
          ls='--', lw=2., marker='None', color='dimgray', alpha=0.5,
          label=r'$\mathrm{10^{-10}}\times m_{\mathrm{total}}$')

        for i in range(len(self.syn_list)):
            
            #This prevents vertical drops in logscale.
            cd = (self.D['m_lvldens' + str(i) + '_cb'] > 1.e-14 * u.solMass)
            
            self.ax_top.plot(
              self.vel_cb_center[cd], self.D['m_lvldens' + str(i) + '_cb'][cd],
              ls='-', lw=2., marker=marker[i], markersize=8., color=color[i],
              label = r'$\rm{t_{exp}\ =\ ' + t_list[i] + '\ \mathrm{d}}$')
                            
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
        rho_test, T_test = [], []

        for i, T in enumerate(self.temperature):
            self.sim.plasma.update(t_rad=np.ones(self.N_shells) * T * u.K)
            #Test to make sure the temperature is updated accordingly.
            #print self.sim.plasma.t_rad, T * u.K
            
            for j in range(self.N_shells):
                lvl_number = 0.
                for level in self.lvl_list:
                    lvl_number += (
                      self.sim.plasma.level_number_density.loc[self.Z,1,level][j])

                ion_II_number = self.sim.plasma.ion_number_density.loc[self.Z,1][j]
                ion_III_number = self.sim.plasma.ion_number_density.loc[self.Z,2][j]
                
                #Testing that the density in the simulated ejecta actually
                #corresponds to the requested density. Note that the
                #model densities correspond to the requested density[1::].
                #print self.sim.model.density[j] - self.density[1::][j] * u.g / u.cm**3
                                
                total_ions = sum(self.sim.plasma.ion_number_density.loc[self.Z][j])
                                
                D['lvl_T' + str(i) + 'rho' + str(j)] = lvl_number / total_ions
                D['ion_II_T' + str(i) + 'rho' + str(j)] = ion_II_number / total_ions
                D['ion_III_T' + str(i) + 'rho' + str(j)] = ion_III_number / total_ions
        
                #Test to compare against plot.
                #print T, self.sim.model.density[j], lvl_number / total_ions
        
        #Create 1D array in the 'correct' order.        
        for j in range(self.N_shells):
            for i in range(len(self.temperature)):
                lvl.append(D['lvl_T' + str(i) + 'rho' + str(j)])
                ion_II.append(D['ion_II_T' + str(i) + 'rho' + str(j)])
                ion_III.append(D['ion_III_T' + str(i) + 'rho' + str(j)])
                rho_test.append(self.density[1::][j])
                T_test.append(self.temperature[i])
                
        self.lvl = np.array(lvl)
        self.ion_II = np.array(ion_II)
        self.ion_III = np.array(ion_III)
        #Test to compare against plot.
        #idx_max = self.lvl.argmax()
        #print idx_max, rho_test[idx_max], T_test[idx_max]
        
    def plotting_bot(self):

        cmap = plt.get_cmap('Greys')
        
        qtty_max = 10. ** int(np.log10(max(self.lvl)))
        qtty_min = qtty_max / self.cbar_range
        qtty = np.reshape(self.lvl, (self.N_shells, len(self.temperature)))
            
        #imshow
        _im = self.ax_bot.imshow(
          qtty, interpolation='nearest', aspect='auto',
          extent=[1., len(self.temperature) + 1, 1., self.N_shells + 1],
          origin='lower', cmap=cmap,
          norm=colors.LogNorm(vmin=qtty_min,  vmax=qtty_max))
        
        #Format ticks. Note that the range of values plotted in x (and y) is
        #defined when calling imshow, under the 'extent' argument.
        yticks_pos = np.arange(1.5, self.N_shells + 0.6, 1.)
        yticks_label = [''] * self.N_shells
        for i in xrange(0, self.N_shells, 2):
            #Every two licks, name the label according to thed density. Note
            #that density[0] is not actually used as it is the density
            #at the photosphere.
            yticks_label[i] = int(np.log10(self.density[i + 1]))
        self.ax_bot.set_yticks(yticks_pos)
        self.ax_bot.set_yticklabels(yticks_label, rotation='vertical')
        
        xticks_pos = np.arange(1.5, len(self.temperature) + 0.6, 1.)
        xticks_label = [''] * len(self.temperature)
        for i in xrange(0, len(self.temperature), 4):
            #Different than the density, all temperatures are used since the
            #T is set across the ejecta and the there is no "photosphere" value.
            xticks_label[i] = format(np.log10(self.temperature[i]), '.2f')
            #xticks_label[i] = format(self.temperature[i], '.2f')
        self.ax_bot.set_xticks(xticks_pos)
        self.ax_bot.set_xticklabels(xticks_label, rotation='vertical')
        
        cbar = self.fig.colorbar(_im, orientation='vertical', pad=0.01,
                                      fraction=0.10, aspect=20, ax=self.ax_bot)

        ticks = np.logspace(np.log10(qtty_max / self.cbar_range),
                            np.log10(qtty_max), np.log10(self.cbar_range) + 1)            
        tick_labels = [r'$10^{' + str(int(np.log10(v))) + '}$' for v in ticks]
        tick_labels[0] = r'$\leq\ 10^{' + str(int(np.log10(ticks[0]))) + '}$'
        #tick_labels[-1] = r'$\geq\ 10^{' + str(int(np.log10(ticks[-1]))) + '}$'
                    
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(width=1, labelsize=fs)
    
        #Make label string containing the denominator.
        den_str = '/\mathrm{n}(\mathrm{' + self.el + '})$'
        
        #label = (r'n$(\mathrm{' + self.el + '_{II}^{' + self.lvl_str + '}})'
        #         + den_str)

        label = (r'n$(\mathrm{' + self.el + '^{+}_{n=' + self.lvl_str + '}})'
                 + den_str)
                    
        cbar.set_label(label, fontsize=fs)

    def add_tracks_bot(self):

        #Add 11fe tracks
        for i, syn in enumerate(self.syn_list):
            with open(syn + '.pkl', 'r') as inp:
                pkl = cPickle.load(inp)
                t_rad = pkl['t_rad'].value
                density = pkl['density'].value
                v_inner = pkl['v_inner'].to(u.km / u.s)
                #Test to compare against plot.
                #print i, np.log10(density[0]), np.log10(t_rad[0])
                self.ax_bot.plot(self.T2axis(t_rad), self.dens2axis(density),
                        color=color[i], ls='-', lw=2.)
            
                #Find velocity reference markers.
                for j, v in enumerate(mark_velocities):
                    idx = (np.abs(v_inner - v)).argmin()
                    
                    self.ax_bot.plot(
                      self.T2axis(t_rad[idx]), self.dens2axis(density[idx]),
                      color=color[i], ls='None', marker='*',
                      markersize=8.)

        #Add 05bl track for comparison.
        '''
        for i, syn in enumerate(self.syn_05bl):
            with open(syn + '.pkl', 'r') as inp:
                pkl = cPickle.load(inp)
                t_rad = pkl['t_rad'].value
                density = pkl['density'].value
                v_inner = pkl['v_inner'].to(u.km / u.s)
                #Test to compare against plot.
                #print i, np.log10(density[0]), np.log10(t_rad[0])
                self.ax_bot.plot(self.T2axis(t_rad), self.dens2axis(density),
                        color='m', ls='--', lw=2.)
            
                #Find velocity reference markers.
                for j, v in enumerate(mark_velocities):
                    idx = (np.abs(v_inner - v)).argmin()
                    
                    self.ax_bot.plot(
                      self.T2axis(t_rad[idx]), self.dens2axis(density[idx]),
                      color='m', ls='None', marker='*',
                      markersize=8.)        
        '''
    def print_table(self):
        directory = './../OUTPUT_FILES/TABLES/'
        with open(directory + 'tb_masses.txt', 'w') as out:

            for i in range(len(self.syn_list)):
                line1 = (
                  '\multirow{3}{*}{' + t_list[i] + '} & $m_{\\rm{tot}}$ & $'
                  + format(self.D['m_dens' + str(i) + '_i'].value, '.3f')
                  + '$ & $'
                  + format(self.D['m_dens' + str(i) + '_o'].value, '.3f')
                  + '$ & $'
                  + format(self.D['m_dens' + str(i) + '_t'].value, '.3f')
                  + '$') 
                line2 = (
                  ' & $m(\\rm{C})$ & $'
                  + format(self.D['m_Cdens' + str(i) + '_i'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_Cdens' + str(i) + '_o'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_Cdens' + str(i) + '_t'].value, '.4f')
                  + '$ \\\\\n') 
                line3 = (
                  ' & $10^{11}\\times m(\\rm{C^{+}_{n=10}})$ & $'
                  + format(self.D['m_lvldens' + str(i) + '_i'].value * 1.e11, '.3f')
                  + '$ & $'
                  + format(self.D['m_lvldens' + str(i) + '_o'].value * 1.e11, '.3f')
                  + '$ & $'
                  + format(self.D['m_lvldens' + str(i) + '_t'].value * 1.e11, '.3f')
                  + '$\\Bstrut')
                if i == 0:
                    line1 += ' \\\\\n'
                else:
                    line1 += '\\Tstrut \\\\\n'                    
                if i != 4:
                    line3 += ' \\\\\n'
            
                out.write(line1)
                out.write(line2)
                out.write(line3)
                if i != 4:
                    out.write('\\hline')

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_physical_conditions.pdf',
                        format='pdf', dpi=360)

    def dens2axis(self, dens):
        #Note the first shell is for density[1], not [0]
        a = ((len(self.density) - 2.) 
             / (np.log10(self.density[-1]) - np.log10(self.density[1])))
        b = 1.5 - a * np.log10(self.density[1])
        return a * np.log10(dens) + b

    def T2axis(self, T):
        #Note the first shell is for temperature is [0].
        a = ((len(self.temperature) - 1.) 
             / (np.log10(self.temperature[-1]) - np.log10(self.temperature[0])))
        b = 1.5 - a * np.log10(self.temperature[0])
        return a * np.log10(T) + b
        
    def clean_up(self):
        os.remove('./inputs/abundance.dat')
        os.remove('./inputs/density.dat')
        
    def run_make_slab(self):
        self.set_fig_frame()
        self.retrieve_number_dens()
        self.get_C_mass()
        self.plotting_top()
        self.make_input_files()
        self.create_ejecta()
        self.collect_states()
        self.plotting_bot()
        self.add_tracks_bot()
        if self.save_fig:
            self.print_table()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        self.clean_up()
        plt.close()
        
if __name__ == '__main__': 
    
    #Mazzali profiles.
    case_folder = path_tardis_output + '11fe_default_L-scaled/'
    f1 = 'velocity_start-'
    f2 = '_loglum-'
    f3 = '_line_interaction-downbranch_time_explosion-'
    syn_list_orig = [case_folder + (f1 + v + f2 + L + f3 + t) + '/'
                     + (f1 + v + f2 + L + f3 + t)
                     for (v, L, t) in zip(v_list, L_list, t_list)]

    #Accepted model profile
    X_i = '0.2'
    X_o = '1.00'
    #X_i = '0.00'
    #X_o = '2.00'

    fname = 'line_interaction-downbranch_excitation-dilute-lte_'\
            + 'C-F2-' + X_o + '_C-F1-' + X_i
    syn_list = [path_tardis_output + '11fe_' + t + 'd_C-best/' + fname
                     + '/' + fname for t in t_label]

    #05bl profile for comparison.
    case_folder = path_tardis_output + '05bl_default/'
    f1 = 'velocity_start-'
    f2 = '_loglum-'
    f3 = '_time_explosion-'
    syn_05bl = [case_folder + (f1 + v + f2 + L + f3 + t) + '/'
                     + (f1 + v + f2 + L + f3 + t)
                     for (v, L, t) in zip(v_05bl, L_05bl, t_05bl)]    
    
    Make_Slab(syn_list, syn_05bl, Z=6, lvl_list=[10], show_fig=True,
              save_fig=True)
