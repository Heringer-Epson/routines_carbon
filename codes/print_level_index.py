#!/usr/bin/env python

import os                                                               
import pandas as pd
path_tardis_output = os.environ['path_tardis_output']
    
class Print_Indexes(object):
    """
    Description:
    ------------
    This code provides a quick and simple way to verify which indexes in TARDIS
    correspond to a given level transition. Pass as input the wavelength range
    in which to search for line transitions.           

    Parameters:
    -----------
    Z : ~int
        Atomic number.
    ionization : ~int
        Ion number. (0 is neutral, 1 is singly ionized and so on).
    w_l : ~float
        Lower limit of the wavelength range.
    w_u : ~float
        Upper limit of the wavelength range.

    Outputs:
    --------
    Prints (among other stuff) the 'level_number_lower' and 'level_number_upper'
    quantities, which can be used as inputs to retrieve, for instance,
    opacities via the 'plot_opacities.py' routine. 
    """
    def __init__(self, Z, ion, w_l, w_u):

        fname = 'seed-23111970'
        fpath = path_tardis_output + 'fast_single_blank/' + fname + '/' + fname + '.hdf'

        lines = pd.read_hdf(fpath, '/simulation/plasma/lines').loc[Z,ion]
        w = lines['wavelength'].values
        w_cond = ((w > w_l) & (w < w_u))
                
        print lines[w_cond]
        
if __name__ == '__main__': 
    #Print_Indexes(Z=6, ion=1, w_l=6578., w_u=6582.)
    Print_Indexes(Z=6, ion=0, w_l=10690., w_u=10696.)

 
