#!/usr/bin/env python

import os                                                               

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import pandas as pd
import cPickle
from binning import make_bin
from binning import get_binned_maxima
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from astropy import units as u

#Accepted model profile
X_i = '0.2'
X_o = '1.00'

L_list = ['8.505', '9.041', '9.362', '9.505', '9.544']
t_list = ['5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['6', '9', '12', '16', '19']
v_list = ['12400', '11300', '10700', '9000', '7850']

#05bl models
L_05bl = ['8.520', '8.617', '8.745', '8.861', '8.594']
t_05bl = ['11.0', '12.0', '14.0', '21.8', '29.9']
v_05bl = ['8350', '8100', '7600', '6800', '3350']

class Make_Slab(object):
    """
    Description:
    ------------
    Makes Table 2 in carbon paper. It uses the default 11fe TARDIS simulations
    to retrieve relevant quantities, which include the masses and of opacities
    due to carbon. There are three major zones analysed: an inner, outer and
    outskirts regions. See paper for further description. Also produces a file
    containing the range of masses that can reproduce the data (see paper for
    details of how the acceptable mass fractions are assessed).
    
    Outputs:
    --------
    ./../OUTPUT_FILES/TABLES/tb_quantities.txt 
    ./../OUTPUT_FILES/OTHER/mass_range.dat
    """
    
    def __init__(self, syn_list, Z=6, ionization_list=[0,1],
                 transitions=[[11,19], [10,12]], save_table=False):
        
        self.syn_list = syn_list
        self.Z = Z
        self.ionization_list = ionization_list
        self.transitions = transitions
        self.save_table = save_table              

        self.sim = None
        self.lvl = None
        self.ion = None
        self.top_label = None
        self.bot_label = None
        self.fname = None
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
                self.D[idx + '_taus'] = []
        
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
            taus = pd.read_hdf(
              syn + '.hdf', '/simulation/plasma/tau_sobolevs')
            #Iterate over shells to get mass densities.
            for j in range(len(self.D[str(i) + '_vinner'])):
                self.D[str(i) + '_eldens'].append(numdens.loc[self.Z][j])
            self.D[str(i) + '_eldens'] = np.asarray(self.D[str(i) + '_eldens'])
            
            #Iterate over ionization state and shells to get ion and
            #ion at a given state densities.
            for m in self.ionization_list:
                idx = str(i) + str(m)
                for j in range(len(self.D[str(i) + '_vinner'])):

                    #For masses, use ions which are that the lower level of the
                    #required transition.
                    self.D[idx + '_iondens'].append(iondens.loc[self.Z,m][j])
                    self.D[idx + '_lvldens'].append(
                      lvldens.loc[self.Z,m,self.transitions[m][0]][j])                
                    self.D[idx + '_taus'].append(
                      taus.loc[self.Z,m,self.transitions[m][0],
                      self.transitions[m][1]][j].values[0])
                    #Test that sumation of ions density equals el density.
           
        #Convert lists to arrays.
        for i, syn in enumerate(self.syn_list):
            for m in self.ionization_list:
                idx = str(i) + str(m)
                self.D[idx + '_iondens'] = np.asarray(self.D[idx + '_iondens'])
                self.D[idx + '_lvldens'] = np.asarray(self.D[idx + '_lvldens'])
                self.D[idx + '_taus'] = np.asarray(self.D[idx + '_taus'])                
       
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
                self.D['m_' + qtty + str(i) + '_u'] =\
                make_bin(A['vel'], A[qtty], time, fb, cb)                    
            
            #12 below is only valid for carbon.
            for m in self.ionization_list:
                idx = str(i) + str(m)
                
                A['iondens'] = (self.D[idx + '_iondens']
                                * u.g * u.u.to(u.g) * 12. / u.cm**3.)
                A['lvldens'] = (self.D[idx + '_lvldens']
                                * u.g * u.u.to(u.g) * 12. / u.cm**3.) 
                A['taus'] = self.D[idx + '_taus'] * u.m / u.m #dimensionless
                            
                #Get masses that require the ionization state.
                for qtty in ['iondens', 'lvldens']:
                    vel_cb,\
                    self.D['m_' + qtty + idx + '_cb'],\
                    self.D['m_' + qtty + idx + '_i'],\
                    self.D['m_' + qtty + idx + '_o'],\
                    self.D['m_' + qtty + idx + '_u'] =\
                    make_bin(A['vel'], A[qtty], time, fb, cb)

                #Get max taus.
                self.D['max_tau' + idx + '_i'],\
                self.D['max_tau' + idx + '_o'],\
                self.D['max_tau' + idx + '_u'] =\
                get_binned_maxima(A['vel'], A['taus'], fb, cb)

    def print_table(self):
        directory = './../OUTPUT_FILES/TABLES/'
        with open(directory + 'tb_quantities.txt', 'w') as out:

            #Header part.
            out.write('\\begin{deluxetable*}{llllll}\n')
            out.write('\\tablecaption{Relevant quantities. \\label{tb:quantities}}\n')
            out.write('\\tablecolumns{6}\n')
            out.write('\\tablehead{\\colhead{}& \\multicolumn{5}{c}{\\dotfill Epoch\\tablenotemark{b} (d)\\dotfill}\\\\\n')
            out.write('\\colhead{Quantity\\tablenotemark{a}} & \\colhead{5.9} & \\colhead{9} & \\colhead{12.1} & \\colhead{16.1} & \\colhead{19.1}}\n')
            out.write('\\startdata\n')
            
            
            #Write global properties.
            out.write('\\sidehead{Global properties}\n')
            
            out.write('$\\log_{10} L/L_\\odot$\\dotfill ')
            for L in L_list:
                out.write('& ' + L)
            out.write('\\\\\n')

            out.write('$v_{\\rm inner}~({\\rm km\,s^{-1}})$\\dotfill')
            for v in v_list:
                out.write('& ' + v)
            out.write('\\\\\n')            
            
            #Write info of the three regions.
            
            regions = [
              '\\sidehead{Inner part ($v_{\\rm inner} < v \leq 13300{\\rm\\,km\\,s^{-1}}$)}\n',
              '\\sidehead{Outer part ($13300 < v \\leq 16000{\\rm\\,km\\,s^{-1}}$)}\n',
              '\\sidehead{Outskirts ($v > 16000{\\rm\\,km\\,s^{-1}}$)}\n']
            for j, region in enumerate(['_i', '_o', '_u']):                
            
                out.write(regions[j])
                
                line1 = '$m_{\\rm tot}~(M_\\odot)$\\dotfill'
                line2 = '$m({\\rm C})~(M_\\odot)$\dotfill'
                line3 = '$m(\\mbox{\\ion{C}{1}})/m({\\rm C})$\\dotfill'
                line4 = '$m(\\mbox{\\ion{C}{2}})/m({\\rm C})$\\dotfill'
                line5 = 'max $\\tau(\\mbox{\\ion{C}{1}}~\\lambda10693)$\\ldots'
                line6 = 'max $\\tau(\\mbox{\\ion{C}{2}}~\\lambda6580)$\\dotfill'
                
                for i in range(len(self.syn_list)):

                    #Retrieve relevant quantities.
                    C_I_fraction = (
                      self.D['m_iondens' + str(i) + '0'  + region].value /
                      self.D['m_eldens' + str(i) + region].value)
                    C_II_fraction = (
                      self.D['m_iondens' + str(i) + '1'  + region].value /
                      self.D['m_eldens' + str(i) + region].value)    

                    #Write in output file.
                    line1 += (' & $' + format(self.D['m_dens' + str(i) + region]
                             .value, '.4f') + '$')
                    line2 += (' & $' + format(self.D['m_eldens' + str(i) + region]
                             .value, '.4f') + '$')
                    line3 += (' & $' + format(C_I_fraction, '.6f') + '$')
                    line4 += (' & $' + format(C_II_fraction, '.4f') + '$')
                    line5 += (' & $' + format(self.D['max_tau' + str(i) + '0'  + region]
                             .value, '.5f') + '$')
                    line6 += (' & $' + format(self.D['max_tau' + str(i) + '1'  + region]
                             .value, '.5f') + '$')
            
                out.write(line1 + ' \\\\\n')
                out.write(line2 + ' \\\\\n')
                out.write(line3 + ' \\\\\n')
                out.write(line4 + ' \\\\\n')
                out.write(line5 + ' \\\\\n')
                out.write(line6 + ' \\\\\n')
                            
            #Wrap up.
            out.write('\\enddata\n')
            out.write('\\tablenotetext{a}{All quantities for carbon are for a fiducial model where $(X(\\rm{C})_{\\rm{i}},X(\\rm{C})_{\\rm{o}})=(0.002,0.01)$.}\n')
            out.write('\\tablenotetext{b}{Relative to time of explosion, with maximum light at 19.1\\,d.}\n')
            out.write('\\end{deluxetable*}')
            
    def write_mass_range(self):
        X_C_i_l = 0.
        X_C_i_u = 5.e-3
        X_C_o_l = 5.e-3
        X_C_o_u = 5.e-2
        
        #Input mass fractions passed were percentages.
        X_C_i = float(X_i) * 0.01 
        X_C_o = float(X_o) * 0.01
        
        #factor inner (outer) lower (upper)
        f_i_l = X_C_i_l / X_C_i 
        f_i_u = X_C_i_u / X_C_i 
        f_o_l = X_C_o_l / X_C_o 
        f_o_u = X_C_o_u / X_C_o 
                
        directory = './../OUTPUT_FILES/OTHER/'
        with open(directory + 'mass_range.dat', 'w') as out:
            out.write('Description:\n')
            out.write('Given the mass fractions derived in the carbon paper '
                      + 'that can reproduce the data,\ncompute the respective '
                      + 'mass range in each zone.\n-----------\n')
            out.write('#vel_range[km/s],lower_m(C)[M_sun],upper_m(C)[M_sun]')
            
            #use last epoch so that the photosphere is at the lowest v (7850km/s)
            i = 4
            m_i = float(format(self.D['m_eldens4_i'].value, '.3f'))     
            m_o = float(format(self.D['m_eldens4_o'].value, '.3f'))
                        
            out.write('\n7850-13300,' + str(m_i * f_i_l) + ',' + str(m_i * f_i_u))
            out.write('\n13300-16000,' + str(m_o * f_o_l) + ',' + str(m_o * f_o_u))
            
    def run_make_table(self):
        self.retrieve_number_dens()
        self.get_C_mass()
        if self.save_table:
            self.print_table()
            self.write_mass_range()
        
if __name__ == '__main__': 
    
    fname = 'line_interaction-downbranch_excitation-dilute-lte_'\
            + 'C-F2-' + X_o + '_C-F1-' + X_i
    syn_list = [path_tardis_output + '11fe_' + t + 'd_C-best/' + fname
                     + '/' + fname for t in t_label]

    Make_Slab(syn_list, Z=6, ionization_list=[0,1],
              transitions=[[11,19], [10,12]], save_table=True)

