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
    
    def __init__(self, syn_list, Z=6, ionization_list=[0,1], transition=10,
                 save_table=False):
        
        self.syn_list = syn_list
        self.Z = Z
        self.ionization_list = ionization_list
        self.transition = transition
        self.save_table = save_table              

        self.sim = None
        self.lvl = None
        self.ion = None
        self.top_label = None
        self.bot_label = None
        self.fname = None
        self.vel_cb_center = None
        self.D = {}

        self.run_make_table()

    def retrieve_number_dens(self):

        #Initiate variables.
        for i, syn in enumerate(self.syn_list):
            self.D[str(i) + '_eldens'] = []
            for m in self.ionization_list:
                idx = str(i) + str(m)
                self.D[idx + '_iondens'] = []
                self.D[idx + '_lvldens'] = []
        
        #Iterate over simulations.
        for i, syn in enumerate(self.syn_list):
            
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

            #Iterate over shells to get mass densities.
            for j in range(len(self.D[str(i) + '_vinner'])):
                self.D[str(i) + '_eldens'].append(numdens.loc[self.Z][j])
            self.D[str(i) + '_eldens'] = np.asarray(self.D[str(i) + '_eldens'])
            
            #Iterate over ionization state and shells to get ion and
            #ion at a given state densities.
            for m in self.ionization_list:
                idx = str(i) + str(m)
                for j in range(len(self.D[str(i) + '_vinner'])):
                    self.D[idx + '_iondens'].append(iondens.loc[self.Z,m][j])
                    self.D[idx + '_lvldens'].append(
                      lvldens.loc[self.Z,m,self.transition][j])                
                    #Test that sumation of ions density equals el density.
                    #print numdens.loc[self.Z][j] - sum(iondens.loc[self.Z][j])
           
            self.D[idx + '_iondens'] = np.asarray(self.D[idx + '_iondens'])
            self.D[idx + '_lvldens'] = np.asarray(self.D[idx + '_lvldens'])
           
    def get_C_mass(self):

        fb = 1. * u.km / u.s
        cb = 200. * u.km / u.s
        A = {}
      
        for i in range(len(self.syn_list)):
           
            time = float(t_list[i]) * u.day
            A['vel'] = self.D[str(i) + '_vinner']
            A['dens'] = self.D[str(i) + '_density']

            A['eldens'] = (self.D[str(i) + '_eldens']
                           * u.g * u.u.to(u.g) * 12. / u.cm**3.)

            #Get masses that don't require the ionization state.
            for qtty in ['dens', 'eldens']:
            
                vel_cb,\
                self.D['m_' + qtty + str(i) + '_cb'],\
                self.D['m_' + qtty + str(i) + '_i'],\
                self.D['m_' + qtty + str(i) + '_o'],\
                self.D['m_' + qtty + str(i) + '_t'] =\
                make_bin(A['vel'], A[qtty], time, fb, cb)            
        
            for m in self.ionization_list:
                idx = str(i) + str(m)
                
                A['iondens'] = (self.D[idx + '_iondens']
                                * u.g * u.u.to(u.g) * 12. / u.cm**3.)
                A['lvldens'] = (self.D[idx + '_lvldens']
                                * u.g * u.u.to(u.g) * 12. / u.cm**3.) 
                            
                #Get masses that require the ionization state.
                for qtty in ['iondens', 'lvldens']:
                    vel_cb,\
                    self.D['m_' + qtty + idx + '_cb'],\
                    self.D['m_' + qtty + idx + '_i'],\
                    self.D['m_' + qtty + idx + '_o'],\
                    self.D['m_' + qtty + idx + '_t'] =\
                    make_bin(A['vel'], A[qtty], time, fb, cb)

            #All quantities should have the same coarse binning, so any works
            #for plotting.
            self.vel_cb_center = (vel_cb.value[0:-1] + vel_cb.value[1:]) / 2.
    
    def print_table(self):
        directory = './../OUTPUT_FILES/TABLES/'
        with open(directory + 'tb_masses.txt', 'w') as out:
            
            for i in range(len(self.syn_list)):
                line1 = (
                  '\multirow{6}{*}{' + t_list[i] + '} & $m_{\\rm{tot}}$ & $'
                  + format(self.D['m_dens' + str(i) + '_i'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_dens' + str(i) + '_o'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_dens' + str(i) + '_t'].value, '.4f')
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
                  ' & $m$(\ion{C}{1}) & $'
                  + format(self.D['m_iondens' + str(i) + '0_i'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_iondens' + str(i) + '0_o'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_iondens' + str(i) + '0_t'].value, '.4f')
                  + '$ \\\\\n')                
                line4 = (
                  ' & $m$(\ion{C}{2}) & $'
                  + format(self.D['m_iondens' + str(i) + '1_i'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_iondens' + str(i) + '1_o'].value, '.4f')
                  + '$ & $'
                  + format(self.D['m_iondens' + str(i) + '1_t'].value, '.4f')
                  + '$ \\\\\n')                  
                line5 = (
                  ' & $m$(\ion{C}{1} $^{\dagger})$\\tablenotemark{d} & $'
                  + format(self.D['m_lvldens' + str(i) + '0_i'].value * 1.e10, '.4f')
                  + '$ & $'
                  + format(self.D['m_lvldens' + str(i) + '0_o'].value * 1.e10, '.4f')
                  + '$ & $'
                  + format(self.D['m_lvldens' + str(i) + '0_t'].value * 1.e10, '.4f')
                  + '$ \\\\\n')                  
                line6 = (
                  ' & $m$(\ion{C}{2} $^{\dagger})$\\tablenotemark{d} & $'
                  + format(self.D['m_lvldens' + str(i) + '1_i'].value * 1.e10, '.4f')
                  + '$ & $'
                  + format(self.D['m_lvldens' + str(i) + '1_o'].value * 1.e10, '.4f')
                  + '$ & $'
                  + format(self.D['m_lvldens' + str(i) + '1_t'].value * 1.e10, '.4f')
                  + '$\\Bstrut')                
                if i == 0:
                    line1 += ' \\\\\n'
                else:
                    line1 += '\\Tstrut \\\\\n'                    
                if i != 4:
                    line6 += ' \\\\\n'
            
                out.write(line1)
                out.write(line2)
                out.write(line3)
                out.write(line4)
                out.write(line5)
                out.write(line6)
                if i != 4:
                    out.write('\\hline \n')

    def run_make_table(self):
        self.retrieve_number_dens()
        self.get_C_mass()
        if self.save_table:
            self.print_table()
        
if __name__ == '__main__': 
    
    #Accepted model profile
    X_i = '0.2'
    X_o = '1.00'
    #X_i = '0.00'
    #X_o = '2.00'

    fname = 'line_interaction-downbranch_excitation-dilute-lte_'\
            + 'C-F2-' + X_o + '_C-F1-' + X_i
    syn_list = [path_tardis_output + '11fe_' + t + 'd_C-best/' + fname
                     + '/' + fname for t in t_label]

    Make_Slab(syn_list, Z=6, ionization_list=[0,1], transition=10,
              save_table=True)

