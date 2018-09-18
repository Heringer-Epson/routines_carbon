#!/usr/bin/env python

import os                                                               
path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import pandas as pd
from astropy import units as u

XC_o = '1.00'
XC_i = ['0.00', '0.05', '0.1', '0.2', '0.5', '1.00']

def get_fname(s1, s2): 
    case_folder = '11fe_19d_C-plateaus_SN/'
    fname = 'C-F2-' + s2 + '_C-F1-' + s1
    fname = case_folder + fname + '/' + fname + '.hdf'        
    return path_tardis_output + fname 
    
class Write_Opacities(object):
    """
    Description:
    ------------
    This code creates a short table containing the the maximum opacity at the
    inner region (7850 < v < 13300 km/s) for different choices of inner carbon
    mass fraction. These numbers are used in 5.3 of the carbon paper to give an
    idea of how the maximum opacity reflects (as expecetd) the presence or not
    of a carbon trough depending on the choice of X(C)_inner. The simulations
    adopt the 'standard' (best) X(C)_outer=0.01  

    Outputs:
    --------
    ./../OUTPUT_FILES/OTHER/max_taus_at_max.txt
    """
    
    def __init__(self):
        self.write_output()

    def write_output(self):

        directory = './../OUTPUT_FILES/OTHER/'
        with open(directory + 'max_taus_at_max.txt', 'w') as out:
            out.write('Record of the maximum opacity in the inner region '\
                      + '(7850 < v < 13300 km/s) for models at maximum using '\
                      + 'X(C)_outer=1%. Relevant for section 5.3.\n\n')
            out.write('X(C)_inner,max_tau')
            
            for XC in XC_i:
                fname = get_fname(XC, XC_o)
                velocities = pd.read_hdf(fname, '/simulation/model/v_inner')
                opacities = pd.read_hdf(fname, '/simulation/plasma/tau_sobolevs')
                
                v = velocities.values * (u.cm /u.s).to(u.km / u.s)
                tau = (opacities.loc[6,1,10,11].values[0]
                       + opacities.loc[6,1,10,12].values[0])
                
                max_tau_inner = str(max(tau[v < 13680.]))

                out.write('\n' + str(float(XC) * 0.01) + ',' + max_tau_inner)
                                    
if __name__ == '__main__': 
    Write_Opacities()
