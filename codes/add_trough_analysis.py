#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import pandas as pd
import cPickle   
from astropy import units as u, constants

c = constants.c.to('m/s').value

def hz2ang(hz):
    return c / hz * 1.e10

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

class Trough_Analysis(object):

    def __init__(self, lm):

        self.lm = lm

        L_scal = np.arange(0.6, 2.01, 0.1)
        self.L = [lum2loglum(0.32e9 * l) for l in L_scal]                
        
        self.Fe = ['0.0', '0.02', '0.04', '0.1', '0.2', '0.4', '1.0', '2.0',
                   '4.0', '10.0']    

        #For the differential spectrum analyses.
        self.C_min = 6200.
        self.C_max = 6500.
                
        self.loop_L_and_Fe()

    def load_data(self, lm, L, Fe, folder_appendix):
        
        if folder_appendix == 'C':
            case_folder = path_tardis_output + 'early-carbon-grid_highres/'
        elif folder_appendix == 'no-C':
            case_folder = path_tardis_output + 'early-carbon-grid_highres_no-C/'
                
        fname = 'loglum-' + L + '_line_interaction-' + lm + '_Fes-' + Fe
        bot_dir = case_folder + fname + '/'
        fullpath = bot_dir + fname + '.pkl'                  
        
        with open(fullpath, 'r') as inp:
            pkl = cPickle.load(inp)
            
        return pkl, bot_dir, fname

    def compute_pEW_and_depth(self, w, f_C, f_noC):
        
        window = ((w >= self.C_min) & (w <= self.C_max))
        f_div = f_C / f_noC
        
        w_wd = w[window]
        f_wd = f_div[window]
        
        pEW = w_wd[-1] - w_wd[0] - sum(np.multiply(np.diff(w_wd), f_wd[0:-1]))
        depth = 1. - min(f_wd)
        
        return f_div, pEW, depth
        
    def call_line_handling_class(self, hdf_name):
                
        #Initialize variables:
        last_line_interaction_in_id = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/last_line_interaction_in_id')
        last_line_interaction_out_id = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/last_line_interaction_out_id')
        last_line_interaction_shell_id = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/last_line_interaction_shell_id')
        output_nu = pd.read_hdf(
          self.hdf_fname, '/simulation15/runner/output_nu')
        lines = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/lines')
        model_lvl_dens = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/level_number_density')
        model_dens = pd.read_hdf(
          self.hdf_fname, '/simulation15/plasma/density').values
        shells = np.arange(0, len(model_dens), 1)

        C_packets, Fe_packets = [], []
        for shell in shells:

            #Add here calculation of column density of C_II ions in lvl 10.
            #self.lvl_frac.append((model_lvl_dens[shell][6][1][10]
            #                  / sum(model_lvl_dens[shell][6][1])))
            
            #Note that packet counting of forming and filling cannot be
            #subtracted. This is because, I suspect, an ion can absorb a
            #packet multiple times, but only the last interation is recorded.
            number_packs, C_trough_forming, Fe_trough_filling = LastLineInteraction(
              shell, last_line_interaction_in_id, last_line_interaction_out_id,
              last_line_interaction_shell_id, output_nu, lines).run_analysis()

            C_packets.append(C_trough_forming)
            Fe_packets.append(Fe_trough_filling)                
        
        return number_packs, C_packets, Fe_packets

    def loop_L_and_Fe(self):

        for i, L in enumerate(self.L):
            for Fe in self.Fe:
                print '...' + str(i + 1) + '/' + str(len(self.L))

                pkl_C, bot_dir, fname = self.load_data(self.lm, L, Fe, 'C')
                pkl_noC, trash1, trash2 = self.load_data(self.lm, L, Fe, 'no-C')
                self.hdf_fname = bot_dir + fname + '.hdf'

                w_C, f_C = pkl_C['wavelength_corr'], pkl_C['flux_smoothed']
                f_noC = pkl_noC['flux_smoothed']
                
                f_div, pEW, depth = self.compute_pEW_and_depth(w_C, f_C, f_noC)
                
                #number_packs, C_packets, Fe_packets = (
                #  self.call_line_handling_class(self.hdf_fname))                
                                
                #Use pkl_C to save an updated pkl file. Note that although
                #both the simulation with and without carbon are used,
                #the updated pickle files will be stored in the folders
                #of the simulations with carbon.
                new_pkl_fullpath = bot_dir + fname + '_up.pkl'
                pkl_C['diff_pEW-fC'] = pEW
                pkl_C['diff_depth-fC'] = depth
                #pkl_C['C_trough_packets'] = C_packets
                #pkl_C['Fe_trough_packets'] = Fe_packets

                with open(new_pkl_fullpath, 'w') as out_pkl:
                    cPickle.dump(pkl_C, out_pkl, protocol=cPickle.HIGHEST_PROTOCOL)                 
                    
class LastLineInteraction(object):
    '''
    #Mode 'line_in_nu' will only take into account the interactions that
    #occur in the requested wvl (i.e absorption in a given wlv.)
    #Mode 'packet_nu' will filter by the emission in a given wvl region.
    
    #For each mode, line_in contains the atomic_levels involved in the
    #absorption process and line_out the atomic_levels involved in the re-
    #emission.
    
    #If 'line_in_nu', then 'line_in' contains the atomic_lvls
    #of all the absoprtion in the requested wvl range and 'line_out'#
    #can be emitted outside that range.
    
    #If 'packet_nu', then line_in may contain absorptions outside the
    #requested wvl range, but the re-emission will be the wvl range.
    
    #Note that the number of packets of line_in and line_out has to be
    #the same. 'packet_nu' is the upper table in gui and 'line_in_nu' the
    #bottom one.
    
    #With mode 'line_in_nu', will give number of packets that interacted
    #in a given wvl region. Does not care about re-emission wvl.
    '''
        
    def __init__(self, shell, 
                 last_line_interaction_in_id, last_line_interaction_out_id,
                 last_line_interaction_shell_id, output_nu, lines):

        mask = ((last_line_interaction_out_id != -1) &
                (last_line_interaction_shell_id == shell))
       
        self.number_packs = len(last_line_interaction_in_id.index)

        self.last_line_interaction_in_id = last_line_interaction_in_id[mask]
        self.last_line_interaction_out_id = last_line_interaction_out_id[mask]
        self.last_line_interaction_shell_id = last_line_interaction_shell_id[mask]
        self.last_line_interaction_angstrom =  output_nu.apply(hz2ang)[mask]
        
        self.lines = lines
        
        self.trough_wmin = 6200.
        self.trough_wmax = 6600.
        
        self.absoption_filter = None
        self.emission_filter = None
        self.C_fraction = None
        self.Fe_fraction = None
                
    def make_filters(self):
        
        #Make absorption filter.
        absorption_in_nu = (
          self.lines.wavelength.iloc[self.last_line_interaction_in_id].values)
        self.absoption_filter = ((absorption_in_nu >= self.trough_wmin) &
                                 (absorption_in_nu <= self.trough_wmax))

        #Filter packets by only those which in the last interaction were
        #absorbed in the C trough region. 
        self.last_line_in_abs = self.lines.iloc[
          self.last_line_interaction_in_id[self.absoption_filter]]
        self.last_line_out_abs = self.lines.iloc[
          self.last_line_interaction_out_id[self.absoption_filter]]

        #Make emission filter.
        emission_in_nu = self.last_line_interaction_angstrom
        self.emission_filter = ((emission_in_nu >= self.trough_wmin) &
                                (emission_in_nu <= self.trough_wmax))

        #Filter packets by only those which in the last interaction were
        #emitted in the C trough region by Fe. 
        self.last_line_in_emi = self.lines.iloc[
          self.last_line_interaction_in_id[self.emission_filter]]
        self.last_line_out_emi = self.lines.iloc[
          self.last_line_interaction_out_id[self.emission_filter]]

    def trough_contribution(self, atomic_number, ion_number):
        
        #Filter absorption by only those which were due to the requested
        #element and ionization state.
        if atomic_number is not None:
            #Filter by absoprtion.
            last_line_out_abs = self.last_line_out_abs[
              self.last_line_out_abs.atomic_number == atomic_number]
              
            #Filter by emission.  
            last_line_in_emi = self.last_line_in_emi[
              self.last_line_in_emi.atomic_number == atomic_number]
                                      
        if ion_number is not None:
            #Filter by absoprtion.
            last_line_out_abs = last_line_out_abs[
              last_line_out_abs.ion_number == ion_number]

            #Filter by emission.  
            last_line_in_emi = last_line_in_emi[
              last_line_in_emi.ion_number == ion_number]

        #Count packets that were absorbed in the trough region but emitted
        #elsewhere.
        last_line_out_count_abs = last_line_out_abs.wavelength.groupby(
          level=0).count()
               
        last_line_out_table_abs = self.lines[['wavelength',
          'atomic_number', 'ion_number', 'level_number_lower',
          'level_number_upper']].ix[last_line_out_count_abs.index]
        last_line_out_table_abs['count'] = last_line_out_count_abs
        last_line_out_table_abs.sort_values(
          'count', ascending=False, inplace=True)
                
        mask_emission = (
          (last_line_out_table_abs.wavelength < self.trough_wmin) |
          (last_line_out_table_abs.wavelength > self.trough_wmax))
        
        trough_forming_packets = float(sum(
          last_line_out_table_abs[mask_emission]['count'].values))

        #Count packets that were emitted in the trough region but absorbed
        #elsewhere.
        last_line_in_count_emi = last_line_in_emi.wavelength.groupby(
          level=0).count()
                                
        last_line_in_table_emi = self.lines[['wavelength',
          'atomic_number', 'ion_number', 'level_number_lower',
          'level_number_upper']].ix[last_line_in_count_emi.index]
        last_line_in_table_emi['count'] = last_line_in_count_emi
        last_line_in_table_emi.sort_values(
          'count', ascending=False, inplace=True)
             
        mask_absorption= (
          (last_line_in_table_emi.wavelength < self.trough_wmin) |
          (last_line_in_table_emi.wavelength > self.trough_wmax))
                
        trough_filling_packets = float(sum(
          last_line_in_table_emi[mask_absorption]['count'].values))
           
        return trough_forming_packets, trough_filling_packets
    
    def run_analysis(self):
        self.make_filters()
        
        C_trough_forming, C_trough_filling = self.trough_contribution(6, 1)
        Fe_trough_forming, Fe_trough_filling = self.trough_contribution(26, None)

        return self.number_packs, C_trough_forming, Fe_trough_filling

if __name__ == '__main__':
    Trough_Analysis(lm='macroatom')


        
