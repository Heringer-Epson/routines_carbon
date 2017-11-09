#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import pandas as pd
import cPickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import units as u, constants

c = constants.c.to('angstrom/s').value
c_kms = constants.c.to('km/s').value

def wav2vel(wav_rest, wav):
    return c_kms / 1.e3 * ((wav / wav_rest)**2. - 1.) / ((wav / wav_rest)**2. + 1.)



def nu2wav(hz):
    return c / hz
def wav2nu(wav):
    return c / wav    
    
class Trough_Analysis(object):
    """Given a hdf output from TARDIS as an input, compute the contribution of
    each shell to forming the carbon trough, the fraction of C atoms in the
    respective ion and level state and also the measured feature velocity.    
    """
    
    def __init__(self, hdf):
        self.hdf = hdf     
        #self.trough_wmin = 6500.
        #self.trough_wmax = 6600.

        self.trough_wmin = 6575.
        self.trough_wmax = 6585.

        self.N_interactions = []
        self.retrieve_trough_packets()

    def retrieve_trough_packets(self):
        """Retrieve the number of packets that helped to form the trough in
        each shell. These are the packets that interact with carbon inside
        the defined trough region and are re-emitted elsewhere.
        """    
        
        #print pd.HDFStore(self.hdf)
        
        nu_max =  wav2nu(self.trough_wmin)
        nu_min =  wav2nu(self.trough_wmax)

        lines = pd.read_hdf(
          self.hdf, '/simulation/plasma/lines')        
        last_line_interaction_in_id = pd.read_hdf(
          self.hdf, '/simulation/runner/last_line_interaction_in_id')
        last_line_interaction_out_id = pd.read_hdf(
          self.hdf, '/simulation/runner/last_line_interaction_out_id')
        last_line_interaction_shell_id = pd.read_hdf(
          self.hdf, '/simulation/runner/last_line_interaction_shell_id')
        #input_nu = pd.read_hdf(
        #  self.hdf, '/simulation/runner/last_interaction_in_nu')
        output_nu = pd.read_hdf(
          self.hdf, '/simulation/runner/output_nu')
        lines = pd.read_hdf(
          self.hdf, '/simulation/plasma/lines')
        input_wav = lines.wavelength.iloc[last_line_interaction_in_id].values

        model_v_inner = pd.read_hdf(self.hdf, '/simulation/model/v_inner')
        N_shells = len(model_v_inner)
                
        #Maybe improve this in the future to mask by line_id.
        mask_scatter = (last_line_interaction_out_id != -1)
        mask_absorption = ((input_wav >= self.trough_wmin)
                           & (input_wav <= self.trough_wmax))
        mask_emission = ((output_nu <= nu_min)
                         | (output_nu >= nu_max))
                        
        for s in range(N_shells):
            #Make shell mask and then combined mask.
            mask_shell = (last_line_interaction_shell_id == s)
            mask = mask_scatter & mask_absorption & mask_emission & mask_shell        
        
            #Get the indexes of the packets which satisfy the mask requirement.
            #Note, this works with either interaction_(in or out), as the
            #**indexes** are the same after the masking process.
            indexed_series = last_line_interaction_in_id[mask]
            #indexed_series = last_line_interaction_out_id[mask]

            #Compute and append how many interactions satisfying the masking
            #criteria are due to C_II ([6,1] index). 
            try:
                N_inter = len(lines.iloc[indexed_series].loc[6,1].values)
            except:
                N_inter = 0
            
            self.N_interactions.append(N_inter)

class Compare_Interaction_Level(object):
    """Compare shell-wise the number density of C_II at levels 10-11 to the
    actual carbon trough interactions as retrieved from the Trough_Analysis
    class above.
    """

    def __init__(self, hdf, show_fig=True, save_fig=False):
        self.hdf = hdf
        self.show_fig = show_fig
        self.save_fig = save_fig
                
        self.t_rad = None
        self.v_inner = None
        self.C_inter = None
        self.C_level_dens = []
        
        self.v_level = None
        self.v_inter = None
        self.v_measured = None

        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.fs = 26.
        
        self.run_comparison()

    def set_fig_frame(self):
        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        self.ax.set_xlabel(x_label, fontsize=self.fs)
        self.ax.set_yscale('log')
        #self.ax.set_xlim(6000., 15000.)
        #self.ax.set_ylim(1.e-4, 1.1)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs, pad=8)
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(5000.)) 
        
    def retrieve_number_dens(self):
        lvl_dens = pd.read_hdf(
          self.hdf, '/simulation/plasma/level_number_density')     
        self.t_rad = pd.read_hdf(
          self.hdf, '/simulation/model/t_radiative').values   
        self.v_inner = (pd.read_hdf(
          self.hdf, '/simulation/model/v_inner').values * u.cm / u.s).to(u.km / u.s)
        
        for j in range(len(self.v_inner)):
            self.C_level_dens.append(lvl_dens.loc[6,1,10][j]
                                     + lvl_dens.loc[6,1,11][j])
        self.C_level_dens = np.array(self.C_level_dens)        
        
    def retrieve_C_interaction(self):
        class_inter = Trough_Analysis(hdf=self.hdf)
        self.C_inter = np.array(class_inter.N_interactions).astype(float)

    def retrieve_velocities(self):
        
        #Measured velocity.
        fpath_pkl = self.hdf.split('.hdf')[0] + '.pkl'
        with open(fpath_pkl) as inp:
            pkl = cPickle.load(inp)
            v_measured = pkl['velocity_fC']
            w_max_red = pkl['wavelength_maxima_red_fC']
            print wav2vel(6580., w_max_red) * (-1000.)
        
        #Velocity from shell where C_II at lvls 10-11 is largest.
        self.v_level = self.v_inner[self.C_level_dens.argmax()]

        #Velocity from shell where number of trough forming interactions is largest.
        self.v_inter = self.v_inner[self.C_inter.argmax()]

        print v_measured * (-1000.), self.v_level, self.v_inter, sum(self.C_inter)

    def make_plot(self):

        #Divide quantities by the mean to make scales similar.
        inter = self.C_inter / np.mean(self.C_inter)
        levels = self.C_level_dens / np.mean(self.C_level_dens)
        
        label_inter = r'$\mathrm{N^{CII}_{inter}} / \langle \mathrm{N^{CII}_{inter}} \rangle $'
        label_levels = r'$\mathrm{n^{CII}_{10-11}} / \langle \mathrm{n^{CII}_{10-11}} \rangle $'
        
        self.ax.plot(self.v_inner, inter, ls='None', marker='o', markersize=12.,
                     color='k', label=label_inter)
        self.ax.plot(self.v_inner, levels, ls='None', marker='^', markersize=12.,
                     color='b', label=label_levels)

    def add_legend(self):
        self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc='best') 
        plt.tight_layout()          

    def save_figure(self): 
        top_dir = './../OUTPUT_FILES/FIGURES/Compare_lvldens_CII-inter/'
        if not os.path.isdir(top_dir):
            os.mkdir(top_dir)          

        L = fpath.split('loglum-')[1][0:5]
        t_exp = fpath.split('time_explosion-')[1][0:4]
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            fname = 'Fig_compare_lvldens_inter_' + t_exp + '_' + L + '.png'
            plt.savefig(top_dir + fname, format='png', dpi=360)
    
    def run_comparison(self):
        self.set_fig_frame()
        self.retrieve_number_dens()
        self.retrieve_C_interaction()
        self.retrieve_velocities()
        self.make_plot()
        self.add_legend()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()
        
if __name__ == '__main__': 

    case_folder = path_tardis_output + '11fe_default_L-scaled_UP/'
    L_list = ['8.942', '9.063', '9.243', '9.544']
    #L_list = ['9.544']
    f1 = 'line_interaction-downbranch_loglum-'
    f2 = '_velocity_start-7850_time_explosion-19.1'
    syn_list = [case_folder + (f1 + L + f2) + '/' + (f1 + L + f2) + '.hdf'
                for L in L_list]
    
    for fpath in syn_list:
        Compare_Interaction_Level(hdf=fpath, show_fig=False, save_fig=False)



