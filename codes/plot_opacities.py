#!/usr/bin/env python

import os                                                               
path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import cPickle
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from matplotlib import cm
from matplotlib import colors

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

L_list = ['7.903', '8.505', '9.041', '9.362', '9.505', '9.544']
t_list = ['3.7', '5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['4', '6', '9', '12', '16', '19']
v_list = ['13300', '12400', '11300', '10700', '9000', '7850']

color = ['c', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
marker = ['*', '^', 's', 'p', 'D', 'v']
mark_velocities = [9000., 11000., 13500., 16000., 19000.] * u.km / u.s

Z2el = {6: 'C', 8: 'O', 14: 'Si', 26: 'Fe'}
ion2symbol = {0: 'I', 1: 'II', 2: 'III'}
line2symbol = {None: '', '6580': '2s^23s', '10693': '2s^22p3s'}
transition2str = {None: '', 10: '_2s2-3s', 11: '_2s2-2p3s', 12: '_2s2-3p', 19: '_2s2-2p3s'}

fs = 20

def make_fpath(_t, vbound=None):
    inp_dir = '11fe_' + _t + 'd_C-best'    
    fname = 'line_interaction-downbranch_' + 'C-F2-1.00_C-F1-0.2'
    if vbound is not None:
        inp_dir += '_boundary'
        fname = 'v_start_F2-' + vbound + '_' + fname + '_v_stop_F1-' + vbound      
    return path_tardis_output + inp_dir + '/' + fname + '/' + fname

def make_idx(_t, _v):
    if _v is None:
        return _t + '_13500_'
    else:
        return _t + '_' + _v + '_'
        

class Make_Slab(object):
    """
    Description:
    ------------
    Makes Figures 6 of the carbon paper. This includes two panels.
      -Top shows the opacity for the best models for 11fe.
      -Bottom shows the fraction of ions at a given population as computed under
       LTE assumptions.

    Parameters:
    -----------
    Z : ~int
        Atomic number.
    ionization : ~int
        Ion number. (0 is neutral, 1 is singly ionized and so on).
    lvl_low : ~int
        Atomic level of the lower state to be used. Values follow TARDIS usage.
    lvl_up : ~int
        Atomic level of the upper state to be used. Values follow TARDIS usage.      
    line : ~str
        String to be used to make labels.         
    
    Notes:
    ------
    The top panel will used both the lower and upper levels, as required for
    an atomic transition. The bottom panel shows the number density of a given
    atomic level (normalized by the number density of carbon). By convention,
    the number density of the *lower* level is plotted.

    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_opacity_Z_ion_lvl.pdf
    """
    
    def __init__(self, Z=6, ionization=1, lvl_low=10, lvl_up=11,
                 line='6580', show_fig=True, save_fig=False):
        
        self.Z = Z
        self.ionization = ionization
        self.lvl_low = lvl_low
        self.lvl_up = lvl_up
        self.line = line
        self.show_fig = show_fig
        self.save_fig = save_fig      
        
        self.density = 10. ** np.arange(-17.25, -11.4, 0.25)
        self.temperature = 10. ** np.arange(3.75, 4.251, 0.01)
        self.velocity = np.linspace(19000., 20001, len(self.density))

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

        self.run_make_slab()

    def make_label(self):
        var_label = (
          '\mathrm{' + Z2el[self.Z] + '_{' + ion2symbol[self.ionization] + '}'\
          + '^{' + line2symbol[self.line] + '}}')       

        var_label2 = (
          '\mathrm{' + Z2el[self.Z] + '_{' + ion2symbol[self.ionization]
          + '}\ \lambda ' + self.line + '}')   

        ion_label = '\mathrm{' + Z2el[self.Z] + '_{' + ion2symbol[self.ionization] + '}}' 
        
        self.top_label = (r'$\mathrm{\tau}\ (' + var_label2 + ')$') 
        self.bot_label = r'$n(' + var_label + ')/n(\mathrm{' + Z2el[self.Z] + '})$'

        #Make figure name.
        self.fname = (
          'Fig_opacity_' + Z2el[self.Z] + '_' + ion2symbol[self.ionization]\
          + '_' + self.line  + '.pdf')

    def set_fig_frame(self):
        """Define the configuration of the figure axes."""

        x_label_top = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label_top = self.top_label

        x_label_bot = r'$\rm{log}\ T\ \rm{[K]}$'
        y_label_bot = r'$\rm{log}\ \rho\ \rm{[g\ cm^{-3}]}$'       

        self.ax_top.set_xlabel(x_label_top, fontsize=fs)
        self.ax_top.set_ylabel(y_label_top, fontsize=fs)
        self.ax_top.set_yscale('log')
        self.ax_top.set_xlim(8000., 25000.)
        self.ax_top.set_ylim(1.e-5,1.e2)
        self.ax_top.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_top.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_top.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax_top.tick_params(
          'both', length=4, width=1, which='minor', direction='in')
        self.ax_top.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax_top.xaxis.set_major_locator(MultipleLocator(5000.)) 
        
        self.ax_bot.set_xlabel(x_label_bot, fontsize=fs)
        self.ax_bot.set_ylabel(y_label_bot, fontsize=fs)
        self.ax_bot.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_bot.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_bot.minorticks_off()
        self.ax_bot.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax_bot.tick_params(
          'both', length=4, width=1, which='minor', direction='in')    

    def retrieve_number_dens(self):
        
        #Iterate over simulations.
        for _t in t_label:
            for vbound in ['12000', None, '15000']:
                fpath = make_fpath(_t, vbound)
                idx = make_idx(_t, vbound)
                
                self.D[idx + '_eldens'] = []
                self.D[idx + '_iondens'] = []
                self.D[idx + '_lvldens'] = []
                self.D[idx + '_opacity'] = []
                
                lvldens = pd.read_hdf(
                  fpath + '.hdf', '/simulation/plasma/level_number_density')     
                numdens = pd.read_hdf(
                  fpath + '.hdf', '/simulation/plasma/number_density')
                iondens = pd.read_hdf(
                  fpath + '.hdf', '/simulation/plasma/ion_number_density')
                opacity = pd.read_hdf(
                  fpath + '.hdf', '/simulation/plasma/tau_sobolevs')
                self.D[idx + '_vinner'] = (pd.read_hdf(
                  fpath + '.hdf', '/simulation/model/v_inner').values * 1.e-5)
                self.D[idx + '_density'] = (pd.read_hdf(
                  fpath + '.hdf', '/simulation/plasma/density')).values / u.cm**3 * u.g 
                
                #Iterate over shells.
                for j in range(len(self.D[idx + '_vinner'])):

                    #Get el, ion and lvl number density density per shell.
                    self.D[idx + '_eldens'].append(numdens.loc[self.Z][j])
                    self.D[idx + '_iondens'].append(
                      iondens.loc[self.Z,self.ionization][j])
                   
                    if self.lvl_low is not None:
                        self.D[idx + '_lvldens'].append(
                          lvldens.loc[self.Z,self.ionization,self.lvl_low[0]][j])                
                        _opacity = 0.
                        for (lvl_l,lvl_u) in zip(self.lvl_low,self.lvl_up):
                            _opacity += opacity.loc[
                              self.Z,self.ionization,lvl_l,lvl_u][j].values[0]
                        self.D[idx + '_opacity'].append(_opacity)
                    else:
                        self.D[idx + '_lvldens'].append(float('Nan'))   
                        self.D[idx + '_opacity'].append(float('Nan'))                          
                            
                    #Test that sumation of ions density equals el density.
                    #print numdens.loc[self.Z][j] - sum(iondens.loc[self.Z][j])
                    #print iondens.loc[self.Z,self.ionization][j] - sum(lvldens.loc[self.Z,self.ionization][j])

                #Convert lists to arrays.
                self.D[idx + '_eldens'] = np.asarray(self.D[idx + '_eldens'])
                self.D[idx + '_iondens'] = np.asarray(self.D[idx + '_iondens'])
                self.D[idx + '_lvldens'] = np.asarray(self.D[idx + '_lvldens'])
                self.D[idx + '_opacity'] = np.asarray(self.D[idx + '_opacity'])

    def plotting_top(self):
        for i, _t in enumerate(t_label):

            #Curves where the separation between the inner and outer zones is
            #the default one, i.e. 13500 km/s
            idx = make_idx(_t, None)
            x, y = self.D[idx + '_vinner'], self.D[idx + '_opacity']
            
            self.ax_top.plot(
              x, y, ls='-', lw=3., marker=marker[i], markersize=9., markevery=2,
              color=color[i], label = r'$\rm{t_{exp}\ =\ ' + t_list[i] + '\ \mathrm{d}}$') 

            #Fill between. This shows the effects of displacing the velocity
            #boundary between the inner and outer zones.
            idx_l, idx_u = make_idx(_t, '12000'), make_idx(_t, '15000')
            y_l, y_u = self.D[idx_l + '_opacity'], self.D[idx_u + '_opacity']
            self.ax_top.fill_between(x, y_l, y_u, color=color[i], alpha=0.3)
            
            #Add linestyles for clarity.
            self.ax_top.plot(x, y_l, ls=':', lw=1., marker='None', color=color[i])                     
            self.ax_top.plot(x, y_u, ls='--', lw=1., marker='None', color=color[i])                     
        
        self.ax_top.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1,
                           labelspacing=0.1, loc='best')
    
                            
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
        ion, lvl, opacity = [], [], []
        rho_test, T_test = [], []

        for i, T in enumerate(self.temperature):
            self.sim.plasma.update(t_rad=np.ones(self.N_shells) * T * u.K)
            #Test to make sure the temperature is updated accordingly.
            #print self.sim.plasma.t_rad, T * u.K
            
            for j in range(self.N_shells):
                
                #Retrieve lvl densities.
                lvl_number = 0.
                if self.lvl_low is not None:
                    lvl_number = self.sim.plasma.level_number_density.loc[
                      self.Z,self.ionization,self.lvl_low[0]][j]
                else:
                    lvl_number = float('NaN')
                
                #print opacity_number
                #print dir(self.sim.plasma)
                ion_number = self.sim.plasma.ion_number_density.loc[
                  self.Z,self.ionization][j]
                el_number = self.sim.plasma.number_density.loc[self.Z][j]                
                #Testing that the density in the simulated ejecta actually
                #corresponds to the requested density. Note that the
                #model densities correspond to the requested density[1::].
                #print self.sim.model.density[j] - self.density[1::][j] * u.g / u.cm**3
                #print el_number - sum(self.sim.plasma.ion_number_density.loc[self.Z][j])
                                
                #total_ions = sum(self.sim.plasma.ion_number_density.loc[self.Z][j])
                total_ions = el_number
                #total_ions = ion_number
                                
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
        
        if self.lvl_low is None:
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
        #defined when calling imshow, under the 'extent' argument. Note
        #that density[0] is not used as it is the density at the photosphere.
        yticks_pos = np.arange(1.5, self.N_shells + 0.6, 1.)
        yticks_major_cond = (np.log10(self.density[1:]) % 1 == 0.)
        yticks_major_pos = yticks_pos[yticks_major_cond] 
        yticks_major_label = np.log10(self.density[1:])[yticks_major_cond].astype(int)
        self.ax_bot.set_yticks(yticks_pos, minor=True)
        self.ax_bot.set_yticks(yticks_major_pos, minor=False)
        self.ax_bot.set_yticklabels(yticks_major_label, rotation='vertical')        
        
        xticks_pos = np.arange(1.5, len(self.temperature) + 0.6, 1.)
        xticks_major_cond = (np.round((np.log10(self.temperature) * 10.),2) % 1 == 0.)        
        xticks_major_pos = xticks_pos[xticks_major_cond] 
        xticks_major_label = np.log10(self.temperature)[xticks_major_cond]
        self.ax_bot.set_xticks(xticks_pos, minor=True)
        self.ax_bot.set_xticks(xticks_major_pos, minor=False)
        self.ax_bot.set_xticklabels(xticks_major_label)
        
        cbar = self.fig.colorbar(_im, orientation='vertical', pad=0.01,
                                      fraction=0.10, aspect=20, ax=self.ax_bot)

        ticks = np.logspace(np.log10(qtty_max / self.cbar_range),
                            np.log10(qtty_max), np.log10(self.cbar_range) + 1)            
        tick_labels = [r'$10^{' + str(int(np.log10(v))) + '}$' for v in ticks]
        tick_labels[0] = r'$\leq\ 10^{' + str(int(np.log10(ticks[0]))) + '}$'
                    
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(width=1, labelsize=fs)
                    
        cbar.set_label(self.bot_label, fontsize=fs)

    def add_tracks_bot(self):

        #Add 11fe tracks
        for i, _t in enumerate(t_label):
            fpath = make_fpath(_t)
            with open(fpath + '.pkl', 'r') as inp:
                pkl = cPickle.load(inp)
                t_rad = pkl['t_rad'].value
                density = pkl['density'].value
                v_inner = pkl['v_inner'].to(u.km / u.s)
                #Test to compare against plot.
                #print i, np.log10(density[0]), np.log10(t_rad[0])
                self.ax_bot.plot(self.T2axis(t_rad), self.dens2axis(density),
                        color=color[i], ls='-', lw=3.)
                #Find velocity reference markers.
                for j, v in enumerate(mark_velocities):
                    v_diff = np.abs(v_inner - v)
                    #if min(v_diff) < 115. * u.km / u.s:
                    if min(v_diff) < 150. * u.km / u.s:
                        idx = (v_diff).argmin()
                    
                        self.ax_bot.plot(
                          self.T2axis(t_rad[idx]), self.dens2axis(density[idx]),
                          color=color[i], ls='None', marker='o',
                          markersize=6. + 2. * j)

        #Make legend for velocities.
        for j, v in enumerate(mark_velocities):
            self.ax_bot.plot(
              [np.nan], [np.nan], color='k', marker='o', markersize=6. + 2. * j,
              ls='none', label= r' $' + str(int(v.value)) + '{\\rm\\,km\\,s^{-1}}$')
       
        self.ax_bot.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1,
                           labelspacing=0.1, handletextpad=-0.1, loc=3)

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
        self.plotting_top()
        self.make_input_files()
        self.create_ejecta()
        self.collect_states()
        self.plotting_bot()
        self.add_tracks_bot()
        plt.gca().invert_xaxis()
        plt.tight_layout()
        self.save_figure()
        if self.show_fig:
            plt.show()
        self.clean_up()
        plt.close()
        
if __name__ == '__main__':  

    Make_Slab(
      Z=6, ionization=0, lvl_low=[7, 6, 8, 7, 8], lvl_up=[15, 14, 16, 14, 15],
      line='10693', show_fig=False, save_fig=True)

    Make_Slab(
      Z=6, ionization=1, lvl_low=[10,10], lvl_up=[11,12],
      line='6580', show_fig=False, save_fig=True)    
