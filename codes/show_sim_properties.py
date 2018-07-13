#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import pandas as pd
import h5py

L_list = ['8.505', '9.041', '9.362', '9.505', '9.544']
t_list = ['5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['6', '9', '12', '16', '19']
v_list = ['12400', '11300', '10700', '9000', '7850']

j = 5

class Print_Physics(object):
    """TBW   
    """
    
    def __init__(self, syn_list):
        self.syn_list = syn_list
        self.print_quantities()

    def print_quantities(self):
                
        for i, syn in enumerate((self.syn_list[3:5])):
            
            I = syn + '.hdf'

            #Show available keys.
            #f = h5py.File(I, 'r')
            #data = list(f['simulation']['plasma'])
            #print data
            
            v_inner = pd.read_hdf(I, '/simulation/model/v_inner').values / 1.e5
            delta = pd.read_hdf(I, '/simulation/plasma/delta').loc[6,1].values
            t_rad = pd.read_hdf(I, '/simulation/plasma/t_rad').values
            eldens = pd.read_hdf(I, '/simulation/plasma/number_density').loc[6].values
            CII_dens = pd.read_hdf(I, '/simulation/plasma/ion_number_density').loc[6,1].values
            CIII_dens = pd.read_hdf(I, '/simulation/plasma/ion_number_density').loc[6,2].values
            lvldens_l = pd.read_hdf(I, '/simulation/plasma/level_number_density').loc[6,1,10].values
            lvldens_u = pd.read_hdf(I, '/simulation/plasma/level_number_density').loc[6,1,12].values
            opacities = pd.read_hdf(I, '/simulation/plasma/tau_sobolevs').loc[6,1,10,12].values[0]
            
            CII_frac = CII_dens /eldens
            CIII_frac = CIII_dens / eldens
            
            v_cond = (v_inner == 9157)
            
            #print t_list[3:5][i], v_inner[v_cond], t_rad[v_cond], CII_frac[v_cond],
            #print lvldens_l[v_cond] - lvldens_u[v_cond], opacities[v_cond]
            
            #print t_list[3:5][i], float((lvldens_l[v_cond][0] - lvldens_u[v_cond][0])) * float(t_list[3:5][i])
            print t_list[3:5][i], lvldens_l[v_cond] / eldens[i]
                    
        print '\n'
        
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

    fname = 'line_interaction-downbranch_excitation-dilute-lte_'\
            + 'C-F2-' + X_o + '_C-F1-' + X_i
    syn_list = [path_tardis_output + '11fe_' + t + 'd_C-best/' + fname
                     + '/' + fname for t in t_label]  
    
    #fname = 'velocity_start-10700_loglum-9.362_line_interaction-downbranch_C-F2-1.00_C-F1-0.2_time_explosion-12.1'
    #syn_list = [path_tardis_output + '11fe_best_delta/' + fname + '/' + fname]

    #fname = 'seed-23111970'
    #syn_list = [path_tardis_output + 'fast_single_blank/' + fname + '/' + fname]

    Print_Physics(syn_list)

 
