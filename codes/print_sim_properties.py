#!/usr/bin/env python

import os                                                               
import h5py
path_tardis_output = os.environ['path_tardis_output']

class List_Properties(object):
    """
    Description:
    ------------    
    Simple routine to print out the quantities stored in the hdf5 files
    produced by TARDIS. The simulation used was not chosen for any particular
    reason.  
    """
    def __init__(self):
        self.print_quantities()

    def print_quantities(self):
        fname = 'seed-23111970'
        fpath = path_tardis_output + 'fast_single_blank/' + fname\
                + '/' + fname + '.hdf'

        f = h5py.File(fpath, 'r')
        #print list(f['simulation'])
        #print list(f['simulation']['plasma'])
        #print list(f['simulation']['plasma']['tau_sobolevs'])
        print list(f['simulation']['plasma']['lines'])
                            
if __name__ == '__main__': 
    List_Properties()

