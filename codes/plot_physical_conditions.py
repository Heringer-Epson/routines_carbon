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

#mark_velocities = [12500., 13300., 16000.] * u.km / u.s
mark_velocities = [7850., 13300., 16000.] * u.km / u.s
vel_markers = ['x', '+']

fs = 20

class Make_Slab(object):
    """Runs a TARDIS simulation with a specified density profile to analyse and
    plot how the ion fraction of a given element depends on the density and
    temperature.
    Paramters:
    ion indicates the ionization state: 0 for neutral, 1 for singly ionized, etc.    
    """
    
    def __init__(self, syn_list, syn_05bl, Z=6, ionization=1, transition=10, 
                 show_fig=True, save_fig=False):
        
        self.syn_list = syn_list
        self.syn_05bl = syn_05bl
        self.Z = Z
        self.ionization = ionization
        self.transition = transition
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

        self.sim = None
        self.lvl = None
        self.ion = None
        self.top_label = None
        self.bot_label = None
        self.fname = None

        self.cbar_range = 1.e4
        self.mass_range = 1.e3

        self.run_make_slab()

    def make_label(self):
        #Make part of label string containing level info.
        Z2el = {6: 'C', 14: 'Si', 26: 'Fe'}
        ion2symbol = {0: 'I', 1: 'II', 2: 'III'}
        transition2symbol = {None: '', 10: '1s^23s'}
        transition2str = {None: '', 10: '_1s2-3s'}
        
        var_label = (
          '\mathrm{' + Z2el[self.Z] + '_{' + ion2symbol[self.ionization] + '}'\
          + '^{' + transition2symbol[self.transition] + '}}')       
        
        self.top_label = r'$m(' + var_label + ')\ \ \mathrm{[M_\odot]}$'
        self.bot_label = (r'$\mathrm{n}(' + var_label + ')/'\
                          + '\mathrm{n}(\mathrm{' + Z2el[self.Z] + '})$') 

        #Make figure name.
        self.fname = (
          'Fig_conditions_' + Z2el[self.Z] + '_' + ion2symbol[self.ionization]\
          + transition2str[self.transition]  + '.pdf')

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""

        x_label_top = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label_top = self.top_label

        x_label_bot = r'$\rm{log}\ T\ \rm{[K]}$'
        y_label_bot = r'$\rm{log}\ \rho\ \rm{[g\ cm^{-3}]}$'       

        self.ax_top.set_xlabel(x_label_top, fontsize=fs)
        self.ax_top.set_ylabel(y_label_top, fontsize=fs)
        self.ax_top.set_yscale('log')
        self.ax_top.set_xlim(8500., 15000.)
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
        
        #Iterate over simulations.
        for i, syn in enumerate(self.syn_list):
            
            self.D[str(i) + '_eldens'] = []
            self.D[str(i) + '_iondens'] = []
            self.D[str(i) + '_lvldens'] = []
            
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

            #Iterate over shells.
            for j in range(len(self.D[str(i) + '_vinner'])):

                #Get el, ion and lvl number density density per shell.
                self.D[str(i) + '_eldens'].append(numdens.loc[self.Z][j])
                self.D[str(i) + '_iondens'].append(
                  iondens.loc[self.Z,self.ionization][j])
               
                if self.transition is not None:
                    self.D[str(i) + '_lvldens'].append(
                      lvldens.loc[self.Z,self.ionization,self.transition][j])                
                else:
                    self.D[str(i) + '_lvldens'].append(float('Nan'))                
                        
                #Test that sumation of ions density equals el density.
                #print numdens.loc[self.Z][j] - sum(iondens.loc[self.Z][j])

            #Convert lists to arrays.
            self.D[str(i) + '_eldens'] = np.asarray(self.D[str(i) + '_eldens'])
            self.D[str(i) + '_iondens'] = np.asarray(self.D[str(i) + '_iondens'])
            self.D[str(i) + '_lvldens'] = np.asarray(self.D[str(i) + '_lvldens'])

    def get_C_mass(self):

        fb = 1. * u.km / u.s
        cb = 200. * u.km / u.s
        A = {}
        for i in range(len(self.syn_list)):

            time = float(t_list[i]) * u.day
            A['vel'] = self.D[str(i) + '_vinner']
            A['dens'] = self.D[str(i) + '_density']
            A['eldens'] = self.D[str(i) + '_eldens'] * u.g * u.u.to(u.g) * 12. / u.cm**3.
            A['iondens'] = self.D[str(i) + '_iondens'] * u.g * u.u.to(u.g) * 12. / u.cm**3.
            A['lvldens'] = self.D[str(i) + '_lvldens'] * u.g * u.u.to(u.g) * 12. / u.cm**3. 
                        
            #Re-bin the data and compute masses.
            for qtty in ['dens', 'eldens', 'iondens', 'lvldens']:
            
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
          
        for i in range(len(self.syn_list)):
            
            if self.transition is not None:
                qtty = self.D['m_lvldens' + str(i) + '_cb']
            else:
                qtty = self.D['m_iondens' + str(i) + '_cb']
                
            #This prevents vertical drops in logscale.
            cd = (qtty > 1.e-18 * u.solMass)
            
            self.ax_top.plot(
              self.vel_cb_center[cd], qtty[cd],
              ls='-', lw=2., marker=marker[i], markersize=8., color=color[i],
              label = r'$\rm{t_{exp}\ =\ ' + t_list[i] + '\ \mathrm{d}}$')

        qtty_max = 10. ** int(np.log10(max(qtty.value)))
        qtty_min = qtty_max / self.mass_range
        self.ax_top.set_ylim(qtty_min, qtty_max)

        #Use any of the epochs to plot the mass per bin. Note that this is
        #conserved because the explosion is in homologous expnasion.
        #Note that earlier epochs will be cut below the photosphere.
        mass_max = 10. ** int(np.log10(max(self.D['m_dens4_cb'].value)))
        mass_scaling = qtty_max / mass_max
        str_scaling = str(int(np.log10(mass_scaling)))  
        
        mass_scaling_C = mass_scaling * 1.e2
        str_scaling_C = str(int(np.log10(mass_scaling_C)))  

        self.ax_top.plot(
          self.vel_cb_center, self.D['m_eldens4_cb'] * mass_scaling_C,
          ls='--', lw=4., marker='None', color='#b15928', alpha=0.7,
          label=r'$m(\mathrm{C}) \times \mathrm{10^{' + str_scaling_C + '}}$')

        self.ax_top.plot(
          self.vel_cb_center, self.D['m_dens4_cb'] * mass_scaling,
          ls=':', lw=4., marker='None', color='dimgray', alpha=0.7,
          label=r'$m_{\mathrm{total}} \times \mathrm{10^{' + str_scaling + '}}$')

        self.ax_top.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1,
                           labelspacing=-0.2, bbox_to_anchor=(0.615, 0.725),
                           bbox_transform=plt.gcf().transFigure)
                            
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
        ion, lvl = [], []
        rho_test, T_test = [], []

        for i, T in enumerate(self.temperature):
            self.sim.plasma.update(t_rad=np.ones(self.N_shells) * T * u.K)
            #Test to make sure the temperature is updated accordingly.
            #print self.sim.plasma.t_rad, T * u.K
            
            for j in range(self.N_shells):
                
                #Retrieve lvl densities.
                lvl_number = 0.
                if self.transition is not None:
                    lvl_number = self.sim.plasma.level_number_density.loc[
                      self.Z,self.ionization,self.transition][j]
                else:
                    lvl_number = float('NaN')
                
                ion_number = self.sim.plasma.ion_number_density.loc[
                  self.Z,self.ionization][j]
                
                #Testing that the density in the simulated ejecta actually
                #corresponds to the requested density. Note that the
                #model densities correspond to the requested density[1::].
                #print self.sim.model.density[j] - self.density[1::][j] * u.g / u.cm**3
                                
                total_ions = sum(self.sim.plasma.ion_number_density.loc[self.Z][j])
                                
                D['lvl_T' + str(i) + 'rho' + str(j)] = lvl_number / total_ions
                D['ion_T' + str(i) + 'rho' + str(j)] = ion_number / total_ions
        
                #Test to compare against plot.
                #print T, self.sim.model.density[j], lvl_number / total_ions
        
        #Create 1D array in the 'correct' order.        
        for j in range(self.N_shells):
            for i in range(len(self.temperature)):
                lvl.append(D['lvl_T' + str(i) + 'rho' + str(j)])
                ion.append(D['ion_T' + str(i) + 'rho' + str(j)])
                
                rho_test.append(self.density[1::][j])
                T_test.append(self.temperature[i])
                
        self.lvl = np.array(lvl)
        self.ion = np.array(ion)
        
        #Test to compare against plot.
        #idx_max = self.lvl.argmax()
        #print idx_max, rho_test[idx_max], T_test[idx_max]
        
    def plotting_bot(self):

        cmap = plt.get_cmap('Greys')
        
        if self.transition is None:
            qtty = self.ion
        else:
            qtty = self.lvl            
        
        qtty_max = 10. ** int(np.log10(max(qtty)))
        qtty_min = qtty_max / self.cbar_range
        qtty = np.reshape(qtty, (self.N_shells, len(self.temperature)))
            
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
                    
        cbar.set_label(self.bot_label, fontsize=fs)

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
                  '\multirow{4}{*}{' + t_list[i] + '} & $m_{\\rm{tot}}$ & $'
                  + format(self.D['m_dens' + str(i) + '_i'].value, '.3f')
                  + '$ & $'
                  + format(self.D['m_dens' + str(i) + '_o'].value, '.3f')
                  + '$ & $'
                  + format(self.D['m_dens' + str(i) + '_t'].value, '.3f')
                  + '$') 
                line2 = (
                  ' & $m(\\rm{C})$ & $'
                  + format(self.D['m_eldens' + str(i) + '_i'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_eldens' + str(i) + '_o'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_eldens' + str(i) + '_t'].value, '.4f')
                  + '$ \\\\\n') 
                line3 = (
                  ' & $m(\\rm{C})$ & $'
                  + format(self.D['m_iondens' + str(i) + '_i'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_iondens' + str(i) + '_o'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_iondens' + str(i) + '_t'].value, '.4f')
                  + '$ \\\\\n')                
                line4 = (
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
                out.write(line4)
                if i != 4:
                    out.write('\\hline')

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + self.fname, format='pdf', dpi=360)

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
        self.make_label()
        self.set_fig_frame()
        self.retrieve_number_dens()
        self.get_C_mass()
        self.plotting_top()
        self.make_input_files()
        self.create_ejecta()
        self.collect_states()
        self.plotting_bot()
        self.add_tracks_bot()
        #if self.save_fig:
        #    if self.transition is not None:
        #        self.print_table()
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

    #Make_Slab(syn_list, syn_05bl, Z=6, ionization=1, transition=10,
    #          show_fig=False, save_fig=True)
    
    Make_Slab(syn_list, syn_05bl, Z=6, ionization=0, transition=10,
              show_fig=False, save_fig=True)
    Make_Slab(syn_list, syn_05bl, Z=6, ionization=0, transition=None,
              show_fig=False, save_fig=True)
    Make_Slab(syn_list, syn_05bl, Z=6, ionization=1, transition=10,
              show_fig=False, save_fig=True)
    Make_Slab(syn_list, syn_05bl, Z=6, ionization=1, transition=None,
              show_fig=False, save_fig=True)
