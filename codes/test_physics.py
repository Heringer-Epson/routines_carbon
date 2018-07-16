#!/usr/bin/env python

import os                                                               

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import pandas as pd

L_list = ['8.505', '9.041', '9.362', '9.505', '9.544']
t_list = ['5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['6', '9', '12', '16', '19']
v_list = ['12400', '11300', '10700', '9000', '7850']

class Print_Physics(object):
    """
    Description:
    ------------
    This code is a side routine to help understand why the opacity lines in
    Fig. 6 overlap for t=16.1 and 19.1 days. It reads data from the default
    11fe simulations to print a set of quantities that show that the difference
    in temperature (at a given velocity) is sufficient to keep the (relevant)
    level number density high, despite the ejecta expansion.    
    
    Outputs:
    --------
    ./../OUTPUT_FILES/OTHER/late_opacity.csv
    """
    
    def __init__(self, syn_list):
        self.syn_list = syn_list
        self.print_quantities()

    def print_quantities(self):

        directory = './../OUTPUT_FILES/OTHER/'
        with open(directory + 'late_opacity.csv', 'w') as out:

            out.write('Z=6\nion=1\nlevel_low=10\nlevel_up=12\n'
                      + 'Densities are in [#/cm^3]\n\n')
            out.write('t_exp_[d],v_[km/s],t_rad_[k],C_dens,CII_frac,CIII_frac,'
                      + 'lvl_l_dens,lvl_u_dens,lvl_l_frac,lvl_u_frac,'
                      + '(n_l - n_u)*t,opacity')
            for i, syn in enumerate((self.syn_list[3:5])):
                    I = syn + '.hdf'

                    v_inner = pd.read_hdf(I, '/simulation/model/v_inner').values / 1.e5
                    delta = pd.read_hdf(I, '/simulation/plasma/delta').loc[6,1].values
                    t_rad = pd.read_hdf(I, '/simulation/plasma/t_rad').values
                    eldens = pd.read_hdf(I, '/simulation/plasma/number_density').loc[6].values
                    CII_dens = pd.read_hdf(I, '/simulation/plasma/ion_number_density').loc[6,1].values
                    CIII_dens = pd.read_hdf(I, '/simulation/plasma/ion_number_density').loc[6,2].values
                    lvldens_l = pd.read_hdf(I, '/simulation/plasma/level_number_density').loc[6,1,10].values
                    lvldens_u = pd.read_hdf(I, '/simulation/plasma/level_number_density').loc[6,1,12].values
                    opacities = pd.read_hdf(I, '/simulation/plasma/tau_sobolevs').loc[6,1,10,12].values[0]

                    v_cond = (v_inner == 9157)
                    
                    CII_frac = CII_dens / eldens
                    CIII_frac = CIII_dens / eldens
                    lvl_l_frac = lvldens_l / eldens
                    lvl_u_frac = lvldens_u / eldens
                    tau_est = (
                      float((lvldens_l[v_cond][0] - lvldens_u[v_cond][0]))
                      * float(t_list[3:5][i]))
                
                    out.write(
                      '\n' + t_list[3:5][i] + ',' + str(v_inner[v_cond][0])
                      + ',' + str(t_rad[v_cond][0]) + ',' + str(eldens[v_cond][0])
                      + ',' + str(CII_frac[v_cond][0]) + ',' + str(CIII_frac[v_cond][0])
                      + ',' + str(lvldens_l[v_cond][0]) + ','
                      + str(lvldens_u[v_cond][0]) + ',' + str(lvl_l_frac[v_cond][0]) + ','
                      + str(lvl_u_frac[v_cond][0]) + ',' + str(tau_est) + ','
                      + str(opacities[v_cond][0]))
                
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
                     
    Print_Physics(syn_list)
