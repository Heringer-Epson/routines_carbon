#!/usr/bin/env python

from astropy import units as u, constants

c = constants.c.to('m/s').value

def hz2ang(hz):
    return c / hz * 1.e10

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
        
        self.trough_wmin = 6000.
        #self.trough_wmin = 6200.
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

        #Make emission filter.
        emission_in_nu = self.last_line_interaction_angstrom
        self.emission_filter = ((emission_in_nu >= self.trough_wmin) &
                                (emission_in_nu <= self.trough_wmax))

    def C_trough_formation(self):
        atomic_number = 6
        ion_number = 1
        
        #Filter packets by only those which in the last interaction were
        #absorbed in the C trough region by C. 
        self.last_line_in = self.lines.iloc[
          self.last_line_interaction_in_id[self.absoption_filter]]
        self.last_line_out = self.lines.iloc[
          self.last_line_interaction_out_id[self.absoption_filter]]

        #Filter interactions by only those which were due to C_II
        if atomic_number is not None:
            self.last_line_in = self.last_line_in[
              self.last_line_in.atomic_number == atomic_number]
            self.last_line_out = self.last_line_out[
              self.last_line_out.atomic_number == atomic_number]
                        
        if ion_number is not None:
            self.last_line_in = self.last_line_in[
              self.last_line_in.ion_number == ion_number]
            self.last_line_out = self.last_line_out[
              self.last_line_out.ion_number == ion_number]

        #Count lines for each possible interactions (between different energy
        #levels. note that the level argument is pertinent to pandas.
        last_line_in_count = self.last_line_in.wavelength.groupby(level=0).count()
        last_line_out_count = self.last_line_out.wavelength.groupby(level=0).count()
                                
        #Get line information for each possible absorption.
        self.last_line_in_table = self.lines[['wavelength', 'atomic_number', 'ion_number', 'level_number_lower',
                                              'level_number_upper']].ix[last_line_in_count.index]
        self.last_line_in_table['count'] = last_line_in_count
        self.last_line_in_table.sort_values('count', ascending=False, inplace=True)

        #Get emission information for each line that was absorbed in the
        #requested region.                
        self.last_line_out_table = self.lines[['wavelength', 'atomic_number', 'ion_number', 'level_number_lower',
                                              'level_number_upper']].ix[last_line_out_count.index]
        self.last_line_out_table['count'] = last_line_out_count
        self.last_line_out_table.sort_values('count', ascending=False, inplace=True)
                
        #Mask emission.
        mask_emission = (
          (self.last_line_out_table.wavelength < self.trough_wmin) |
          (self.last_line_out_table.wavelength > self.trough_wmax))
                
        #Number of packets that were absorbed in the trough region and
        #re-emitted outside that region.
        trough_packs = float(sum(
          self.last_line_out_table[mask_emission]['count'].values))
                
        self.C_fraction = trough_packs                        

    def C_trough_filling(self):
        atomic_number = 26
        ion_number = None
        
        #Filter packets by only those which in the last interaction were
        #emitted in the C trough region by Fe. 
        self.last_line_in = self.lines.iloc[
          self.last_line_interaction_in_id[self.emission_filter]]
        self.last_line_out = self.lines.iloc[
          self.last_line_interaction_out_id[self.emission_filter]]

        #Filter interactions by only those which were due to Fe in any
        #ionazation state.
        if atomic_number is not None:
            self.last_line_in = self.last_line_in[
              self.last_line_in.atomic_number == atomic_number]
            self.last_line_out = self.last_line_out[
              self.last_line_out.atomic_number == atomic_number]
                        
        if ion_number is not None:
            self.last_line_in = self.last_line_in[
              self.last_line_in.ion_number == ion_number]
            self.last_line_out = self.last_line_out[
              self.last_line_out.ion_number == ion_number]

        #Count lines for each possible interactions (between different energy
        #levels. note that the level argument is pertinent to pandas.
        last_line_in_count = self.last_line_in.wavelength.groupby(level=0).count()
        last_line_out_count = self.last_line_out.wavelength.groupby(level=0).count()
                                
        #Get line information for each possible absorption.
        self.last_line_in_table = self.lines[['wavelength', 'atomic_number', 'ion_number', 'level_number_lower',
                                              'level_number_upper']].ix[last_line_in_count.index]
        self.last_line_in_table['count'] = last_line_in_count
        self.last_line_in_table.sort_values('count', ascending=False, inplace=True)

        #Get emission information for each line that was absorbed in the
        #requested region.                
        self.last_line_out_table = self.lines[['wavelength', 'atomic_number', 'ion_number', 'level_number_lower',
                                              'level_number_upper']].ix[last_line_out_count.index]
        self.last_line_out_table['count'] = last_line_out_count
        self.last_line_out_table.sort_values('count', ascending=False, inplace=True)
                
        #Mask emission.
        mask_emission = (
          (self.last_line_out_table.wavelength >= self.trough_wmin) &
          (self.last_line_out_table.wavelength <= self.trough_wmax))
                
        #Number of packets that were absorbed outside the trough region, but
        #re-emitted in that region.
        trough_packs = float(sum(
          self.last_line_out_table[mask_emission]['count'].values))
        
        self.Fe_fraction = trough_packs                         

    def run_analysis(self):
        self.make_filters()
        self.C_trough_formation()
        self.C_trough_filling()
        return self.C_fraction, self.Fe_fraction, self.number_packs


        
