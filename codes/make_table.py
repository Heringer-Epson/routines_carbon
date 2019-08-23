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
X_m = '1.00'

L_list = ['7.903', '8.505', '9.041', '9.362', '9.505', '9.544']
t_list = ['3.7', '5.9', '9.0', '12.1', '16.1', '19.1']
t_label = ['4', '6', '9', '12', '16', '19']
v_list = ['13300', '12400', '11300', '10700', '9000', '7850']

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

    Notes:
    ------
    CI has multiple lines in TARDIS in the wavelength region of study.
    According to priv. comm. with Stuart: (Jul 30, 2018 - header Carbon paper -
    new draft.)
    "When quoting the optical depth, I would likely either just quote one
    component of the main multiplet (probably the strongest single line) or
    else sum the five strong lines of the multiplet. - I wouldn't likely
    further complicate things with lines of other multiplets unless there is
    really a need to - i.e. if you count the five from your list that have
    lower levels 9, 10 and 11 you will likely have got almost all of what
    matters.

    After correction of the CARSUS database, these lower levels
    are now 6, 7 and 8. The optical depth is the summation of these lines.
    
    Outputs:
    --------
    ./../OUTPUT_FILES/TABLES/tb_quantities.txt 
    ./../OUTPUT_FILES/OTHER/mass_range.dat
    """
    
    def __init__(
      self, syn_list, Z=6, ionization_list=[0,1], transitions_CI=[[10,18],
      [9,17], [11,19], [16,33], [10,17], [42,185], [43,188], [44,187],
      [11,18]], transitions_CII=[[10,11], [10,12]], save_table=False):
        
        self.syn_list = syn_list
        self.Z = Z
        self.ionization_list = ionization_list
        self.transitions = [transitions_CI, transitions_CII] 
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
                      lvldens.loc[self.Z,m,self.transitions[m][0][0]][j])                
                    _opacity = 0.
                    for lvls in self.transitions[m]:
                        _opacity += taus.loc[self.Z,m,lvls[0],lvls[1]][j].values[0]
                    self.D[idx + '_taus'].append(_opacity)
           
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
                self.D['m_' + qtty + str(i) + '_m'],\
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
                    self.D['m_' + qtty + idx + '_m'],\
                    self.D['m_' + qtty + idx + '_o'],\
                    self.D['m_' + qtty + idx + '_u'] =\
                    make_bin(A['vel'], A[qtty], time, fb, cb)

                #Get max taus.
                self.D['max_tau' + idx + '_i'],\
                self.D['max_tau' + idx + '_m'],\
                self.D['max_tau' + idx + '_o'],\
                self.D['max_tau' + idx + '_u'] =\
                get_binned_maxima(A['vel'], A['taus'], fb, cb)        

    def print_table(self):
        directory = './../OUTPUT_FILES/TABLES/'
        with open(directory + 'tb_quantities.txt', 'w') as out:

            #Header part.
            out.write('\\begin{deluxetable*}{lllllll}\n')
            out.write('\\tablecaption{Relevant quantities. \\label{tb:quantities}}\n')
            out.write('\\tablecolumns{7}\n')
            out.write('\\tablehead{\\colhead{}& \\multicolumn{6}{c}{\\dotfill Epoch\\tablenotemark{b} (d)\\dotfill}\\\\\n')
            out.write('\\colhead{Quantity\\tablenotemark{a}} & \\colhead{3.7} & \\colhead{5.9} & \\colhead{9} & \\colhead{12.1} & \\colhead{16.1} & \\colhead{19.1}}\n')
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
              '\\sidehead{Inner part ($v_{\\rm inner} < v \leq 13500{\\rm\\,km\\,s^{-1}}$)}\n',
              '\\sidehead{Middle part ($13500 < v \\leq 16000{\\rm\\,km\\,s^{-1}}$)}\n',
              '\\sidehead{Outer part ($16000 < v \\leq 19000{\\rm\\,km\\,s^{-1}}$)}\n',
              '\\sidehead{Outskirts ($v > 19000{\\rm\\,km\\,s^{-1}}$)}\n']
            for j, region in enumerate(['_i', '_m', '_o', '_u']):                
            
                out.write(regions[j])
                
                line1 = '$M_{\\rm tot}~(M_\\odot)$\\dotfill'
                line2 = '$M_{\\rm C}~(M_\\odot)$\dotfill'
                line3 = '$M_{\\rm C\, I}/M_{\\rm C}$\\dotfill'
                line4 = '$M_{\\rm C\, II}/M_{\\rm C}$\\dotfill'
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
            out.write('\\tablenotetext{a}{All quantities for carbon are for a fiducial model where $(\\xci , \\xcm)=(0.002,0.01)$.}\n')
            out.write('\\tablenotetext{b}{Relative to time of explosion, with maximum light at 19.1\\,d.}\n')
            out.write('\\end{deluxetable*}')
            
    def write_mass_range(self):
        X_C_i_l = 0.
        X_C_i_u = 5.e-3
        X_C_m_l = 1.e-3
        X_C_m_u = 5.e-2
        #X_C_u_l = 0.
        #X_C_u_u = 0.4
                
        #Input mass fractions passed were percentages.
        X_C_i = float(X_i) * 0.01 
        X_C_m = float(X_m) * 0.01
        #X_C_o = float(X_m) * 0.01
        
        #factor inner (outer) lower (upper)
        f_i_l = X_C_i_l / X_C_i 
        f_i_u = X_C_i_u / X_C_i 
        f_m_l = X_C_m_l / X_C_m 
        f_m_u = X_C_m_u / X_C_m 
                
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
            m_m = float(format(self.D['m_eldens4_m'].value, '.3f'))
            #m_o = float(format(self.D['m_eldens4_o'].value, '.3f'))
                        
            out.write('\n7850-13500,' + str(m_i * f_i_l) + ',' + str(m_i * f_i_u))
            out.write('\n13500-16000,' + str(m_m * f_m_l) + ',' + str(m_m * f_m_u))
            #out.write('\n16000-19000,' + str(m_o * f_o_l) + ',' + str(m_o * f_o_u))

    def print_highMG_opacity(self):

        fname = 'line_interaction-downbranch_C-F2-1.00_C-F1-0.2'
        syn = (path_tardis_output + '11fe_9d_C-best_Mg-test/' + fname + '/'
               + fname + '.hdf')


        self.D['Mg_vinner'] = (pd.read_hdf(syn, '/simulation/model/v_inner').values
                               * u.cm / u.s).to(u.km / u.s)
        taus = pd.read_hdf(syn, '/simulation/plasma/tau_sobolevs')
        
        for m in self.ionization_list:
            idx = str(m)
            self.D[idx + '_Mg_taus'] = []
            for j in range(len(self.D['Mg_vinner'])):               
                _opacity = 0.
                for lvls in self.transitions[m]:
                    _opacity += taus.loc[self.Z,m,lvls[0],lvls[1]][j].values[0]
                self.D[idx + '_Mg_taus'].append(_opacity)
    
           
        #Convert lists to arrays.
        for m in self.ionization_list:
            idx = str(m)
            self.D[idx + '_Mg_taus'] = np.asarray(self.D[idx + '_Mg_taus'])                
   
        #Print highest opacity in carbon rich zone.
        cond = (self.D['Mg_vinner'] > 19478. * u.km / u.s)
        print max(self.D['0_Mg_taus'][cond])
            
    def run_make_table(self):
        #self.retrieve_number_dens()
        #self.get_C_mass()
        #if self.save_table:
        #    self.print_table()
        #    self.write_mass_range()
        self.print_highMG_opacity()
        
if __name__ == '__main__': 
    
    fname = 'line_interaction-downbranch_' + 'C-F2-' + X_m + '_C-F1-' + X_i
    syn_list = [path_tardis_output + '11fe_' + t + 'd_C-best/' + fname
                     + '/' + fname for t in t_label]
    
    Make_Slab(syn_list, Z=6, ionization_list=[0,1], transitions_CI=[[7,15],
              [6,14], [8,16], [7,14], [8,15]], transitions_CII=[[10,11],
              [10,12]], save_table=True)

