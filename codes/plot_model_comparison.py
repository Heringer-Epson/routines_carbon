#!/usr/bin/env python
                                                      
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import constants as const
from astropy import units as u
from binning import make_bin

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'  
mpl.rcParams['hatch.color'] = '#e41a1c'

M_sun = const.M_sun.to('g').value
fs = 26.
lw = 2.

model_directories = [
  'Seitenzahl_2013_ddt/', 'Seitenzahl_2016_gcd/', 'Parkmor_2010_merger/']
model_isotope_files = [
  'ddt_2013_n100_isotopes.dat', 'gcd_2016_gcd200_isotopes.dat',
  'merger_2012_11_09_isotopes.dat']
model_time = [100.22 * u.s, 100.22 * u.s, 100.07 * u.s, 100.06 * u.s]
labels = ['DDT N100', 'GCD200', 'Violent Merger']
colors = ['#a6cee3','#1f78b4','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
ls = ['-', '-', '-', '-', '-', '-']

def read_hesma(fpath):
    v, dens, C = np.loadtxt(fpath, skiprows=1, usecols=(0, 1, 19), unpack=True)
    return v * u.km / u.s, dens * u.g / u.cm**3, C
    
def get_mass(v, dens, X, time):
    r = v.to(u.cm / u.s) * time    
    vol = 4. / 3. * np.pi * r**3.
    vol_step = np.diff(vol)
    mass_step = np.multiply(vol_step, dens[1:]) / const.M_sun.to('g')
    
    #Add zero value to array so that the length is preserved.
    mass_step = np.array([0] + list(mass_step))
    
    mass_cord = np.cumsum(mass_step)
    m_X = mass_step * X
    return mass_cord, m_X

class Plot_Models(object):   
    """
    Description:
    ------------
    Makes figure 4 in the carbon paper. Plots the carbon mass fraction
    in the top panel and optinally the carbon mass in the bottom panel as a
    function of the ejecta's velocity for a suit of SN Ia models.
    The mass panel serves as a side plot to figure 4 in the paper, showing that
    either a comparison by mass fraction or mass leads to similar conclusions. 

    Parameters:
    -----------
    mass_subp : ~bool
        If True, then make a bottom panel showing the mass distribution of C.
    
    Notes:
    ------
    Implementation of constrained/unconstrained regions is slightly different
    than in 'plot_densities.py'.
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_model.pdf
    ./../OUTPUT_FILES/FIGURES/Fig_model_with-mass.pdf
    """

    def __init__(self, mass_subp=False, show_fig=True, save_fig=True):

        self.mass_subp = mass_subp
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        if self.mass_subp:
            self.fig_model = plt.figure(figsize=(14.,16.))
            self.ax_XC = self.fig_model.add_subplot(211) 
            self.ax_mass = self.fig_model.add_subplot(212, sharex=self.ax_XC) 
        else:
            self.fig_model = plt.figure(figsize=(14.,8.))
            self.ax_XC = self.fig_model.add_subplot(111) 
        
        self.fb = 1. * u.km / u.s
        self.cb = 200. * u.km / u.s
        self.vel_cb_center = None
        
        self.top_dir = './../INPUT_FILES/model_predictions/'

        self.run_make()

    def set_fig_frame(self):
        
        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label_XC = r'$X_{\rm C}$'
        y_label_mass = r'$m_{\rm C}\ \ \rm{[M_\odot]}$'

        self.ax_XC.set_ylabel(y_label_XC, fontsize=fs)
        self.ax_XC.set_yscale('log')
        self.ax_XC.set_xlim(0., 30000.)
        self.ax_XC.set_ylim(1.e-4, 1.)
        self.ax_XC.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_XC.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_XC.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax_XC.tick_params(
          'both', length=4, width=1, which='minor', direction='in')
        self.ax_XC.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax_XC.xaxis.set_major_locator(MultipleLocator(5000.))  
                  
        if self.mass_subp:
            self.ax_XC.tick_params(labelbottom='off')
            self.ax_mass.set_xlabel(x_label, fontsize=fs)
            self.ax_mass.set_ylabel(y_label_mass, fontsize=fs)
            self.ax_mass.set_yscale('log')
            self.ax_mass.set_xlim(0., 30000.)
            self.ax_mass.set_ylim(1.e-7, 1.e-2)
            self.ax_mass.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
            self.ax_mass.tick_params(axis='x', which='major', labelsize=fs, pad=8)
            self.ax_mass.tick_params(
              'both', length=8, width=1, which='major', direction='in')
            self.ax_mass.tick_params(
              'both', length=4, width=1, which='minor', direction='in')
        else:
            self.ax_XC.set_xlabel(x_label, fontsize=fs)
            

    def add_analysis_regions(self):

        #Get SN 2011fe total mass profile.
        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Mazzali_2011fe/ejecta_layers.dat')                   
        time = 100. * u.s
        v, m, dens, C = np.loadtxt(fpath, skiprows=2, usecols=(1, 2, 3, 9),
                                   unpack=True)
        v = v * u.km / u.s
        dens = 10.**dens * u.g / u.cm**3.
        
        #Allowed regions
        #Note. In the simulations, the abundance in a given layer will remain
        #the same until it changes in the velocity of the next layer. For example
        #If there are two layers at 15808 and 16064 km/s and the abundance is
        #set to change at 16000 km/s, it will remain the same up to 16064 km/s.
        def allowed_C(v_array):
            X_l_array, X_u_array = [], []
            for v in v_array.to(u.km / u.s).value:
                if v < 7850:
                    X_l, X_u = float('NaN'), float('NaN')
                elif v >= 7850 and v < 11465.:
                    X_l, X_u = 0., 5.e-3
                elif v >= 11465. and v < 13680.:
                    X_l, X_u = 0., 1.e-2
                elif v >= 12424. and v < 13680.:
                    X_l, X_u = 0., 5.e-2
                elif v >= 13680. and v < 16064.:
                    X_l, X_u = 1.e-3, 5.e-2
                elif v >= 16064 and v < 19168.:
                    X_l, X_u = 0., 5.e-1
                elif v > 19168.:
                    X_l, X_u = float('NaN'), float('NaN')  
                X_l_array.append(X_l)
                X_u_array.append(X_u)
            return np.asarray(X_l_array), np.asarray(X_u_array)

        allowed_C_l, allowed_C_u = allowed_C(v)
        allowed_Cdens_at100s_l = np.multiply(dens, allowed_C_l) 
        allowed_Cdens_at100s_u = np.multiply(dens, allowed_C_u) 

        zeros_XC = np.zeros(len(v))
        
        #Allowed region
        self.ax_XC.fill_between(
          v.value, allowed_C_l, allowed_C_u, color='limegreen', edgecolor='None')
        #Constrained region
        self.ax_XC.fill_between(
          v.value, zeros_XC, allowed_C_l,
          edgecolor='#e41a1c', facecolor='None', hatch='//', lw=lw)     

        #The 11fe model stop at ~24000km/s, so one needs to make sure the
        #function below does not plot past this velocity.
        cond_XC = (v < 20000. * u.km / u.s)
        self.ax_XC.fill_between(
          v.value[cond_XC], allowed_C_u[cond_XC],
          (zeros_XC + 1.)[cond_XC], edgecolor='#e41a1c', facecolor='None',
          hatch='//', lw=lw)        

        #Plot unconstrained region.
        self.ax_XC.fill_between([0., 7850.], [0., 0.], [1., 1.],
          color='#bababa', edgecolor='None', alpha=0.5)              

        self.ax_XC.fill_between([19168., 35000.], [0., 0.], [1., 1.],
          color='#bababa', edgecolor='None', alpha=0.5) 

        if self.mass_subp:
        
            vel_cb, allowed_mass_l_cb, mass_l_i, mass_l_m, mass_l_o, mass_l_u =\
              make_bin(v, allowed_Cdens_at100s_l, time, self.fb, self.cb)         
            vel_cb, allowed_mass_u_cb, mass_u_i, mass_u_m, mass_u_o, mass_u_u =\
              make_bin(v, allowed_Cdens_at100s_u, time, self.fb, self.cb)   
            #print zip(v, allowed_C_u)
            #print zip(vel_cb, allowed_mass_l_cb, allowed_mass_u_cb)
            #print len(vel_cb), len(allowed_mass_u_cb)
            
            self.vel_cb_center = (vel_cb.value[0:-1] + vel_cb.value[1:]) / 2.
            
            zeros_mass = np.zeros(len(self.vel_cb_center))

            #Allowed region
            self.ax_mass.fill_between(
              self.vel_cb_center, allowed_mass_l_cb.value,
              allowed_mass_u_cb.value, color='limegreen', edgecolor='None')

            #Constrained region
            self.ax_mass.fill_between(
              self.vel_cb_center, zeros_mass, allowed_mass_l_cb.value,
              edgecolor='#e41a1c', facecolor='None', hatch='//', lw=lw)
            
            cond_mass = (self.vel_cb_center < 20000.)

            #To plot the upper constrained region, one has to replace the zeros in
            #the upper limit array with NaN, then add a large number to the non-NaN
            #numbers.
            up_lim = allowed_mass_u_cb.value
            up_lim[up_lim == 0.] = float('NaN')
            
            self.ax_mass.fill_between(
              self.vel_cb_center[cond_mass], allowed_mass_u_cb.value[cond_mass],
              (up_lim + 1.)[cond_mass], edgecolor='#e41a1c', facecolor='None',
              hatch='//', lw=lw)
       
            self.ax_mass.fill_between([0., 7850.], [0., 0.], [1., 1.],
              color='#bababa', edgecolor='None', alpha=0.5)        

            self.ax_mass.fill_between([19168., 35000.], [0., 0.], [1., 1.],
              color='#bababa', edgecolor='None', alpha=0.5)   
        
    def add_tomography_models(self):             
        
        #SN 2011fe.
        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Mazzali_2011fe/ejecta_layers.dat')                   
        time = 100. * u.s
        v, m, dens, C = np.loadtxt(fpath, skiprows=2, usecols=(1, 2, 3, 9),
                                   unpack=True)
        v = v * u.km / u.s
        dens_at100s = 10.**dens * u.g / u.cm**3.
        Cdens_at100s = np.multiply(dens_at100s, C) 

        #Plot original work.
        self.ax_XC.plot(v, C, ls='--', color='k', lw=4.,
                        drawstyle='steps', label=r'SN 2011fe (M14)')
                        
        #Bottom plot
        vel_cb, mass_cb, mass_i, mass_m, mass_o, mass_u =\
          make_bin(v, Cdens_at100s, time, self.fb, self.cb) 
        
        #All quantities should have the same coarse binning, so any works
        #for plotting.
        self.vel_cb_center = (vel_cb.value[0:-1] + vel_cb.value[1:]) / 2.                        
        if self.mass_subp:
            self.ax_mass.plot(
              self.vel_cb_center, mass_cb, ls='--', color='k', lw=4.,
              drawstyle='steps', zorder=4., label=r'SN 2011fe')

    def plot_W7_models(self):
        
        time = (0.000231481 * u.d).to(u.s)
        fpath = self.top_dir + 'W7/W7_model.dat'
        v, dens = np.loadtxt(fpath, skiprows=2, usecols=(1, 2), unpack=True)
        v = v * u.km / u.s
        dens = 10.**dens * u.g / u.cm**3
        dens_at100s = dens * (time / (100. * u.s))**3.

        fpath = self.top_dir + 'W7/W7_abundances.dat'
        C = np.loadtxt(fpath, skiprows=0, usecols=(6,), unpack=True)
        m, m_C = get_mass(v, dens, C, time)
        m_C = np.cumsum(m_C)

        Cdens_at100s = np.multiply(dens_at100s, C) 

        self.ax_XC.step(v, C, ls=ls[-1], color=colors[-1], lw=4., where='post',
                        label='W7')

        #Bottom plot
        vel_cb, mass_cb, mass_i, mass_m, mass_o, mass_u =\
          make_bin(v, Cdens_at100s, 100. * u.s, self.fb, self.cb) 
        if self.mass_subp:
            self.ax_mass.step(self.vel_cb_center, mass_cb, ls=ls[-1],
                              color=colors[-1], lw=4., where='post', label='W7')

    def add_hesma_models(self):
        for i, model_dir in enumerate(model_directories):
            #Note that mass does not need to be compute at 100s as it is conserved.,
            fpath = self.top_dir + model_dir + model_isotope_files[i]
            v, dens, C = read_hesma(fpath)
            dens_at100s = dens * (model_time[i] / (100. *u.s))**3.
            Cdens_at100s = np.multiply(dens_at100s, C) 

            self.ax_XC.step(v, C, ls=ls[i], color=colors[i], lw=4., where='post',
                            label=labels[i])  
                            
            #Bottom plot
            vel_cb, mass_cb, mass_i, mass_m, mass_o, mass_u =\
              make_bin(v, Cdens_at100s, 100. * u.s, self.fb, self.cb) 
            if self.mass_subp:
                self.ax_mass.step(self.vel_cb_center, mass_cb, ls=ls[i], lw=4.,
                                  color=colors[i], where='mid', label=labels[i])    

    def load_Fink_2010_doubledet(self):
        
        #Get time and velocity from model file.
        fpath = self.top_dir + 'Fink_2010_ddet/model_1d.txt'
        with open(fpath, 'r') as inp:
            num_cels = inp.readline() #not used.
            time = float(inp.readline()) * u.d
        v, dens = np.loadtxt(fpath, skiprows=2, usecols=(1,2), unpack=True)
        v = v * u.km / u.s
        dens = 10.**dens * u.g / u.cm**3        
        dens_at100s = dens * (time.to(u.s) / (100. *u.s))**3.

        #Get carbon mass fraction from abundance file.
        fpath = self.top_dir + 'Fink_2010_ddet/abund_1d.txt'
        C = np.loadtxt(fpath, skiprows=0, usecols=(6,), unpack=True)

        Cdens_at100s = np.multiply(dens_at100s, C) 

        #Perform relevant calculation to get mass.
        r_at100s = v.to(u.cm / u.s) * 100. * u.s
        volume = 3. / 4. * np.pi * (r_at100s.to(u.cm))**3.
        volume_at100s = 3. / 4. * np.pi * (r_at100s.to(u.cm))**3.
        volume_at100s_step = np.diff(volume_at100s)

        m, m_C = get_mass(v, dens, C, time)
        m_C = np.cumsum(m_C)
        
        self.ax_XC.step(v, C, ls=ls[-1], color=colors[-3], lw=4., where='mid',
                        label='Double Det.')

        #Bottom plot
        vel_cb, mass_cb, mass_i, mass_m, mass_o, mass_u =\
          make_bin(v, Cdens_at100s, 100. * u.s, self.fb, self.cb) 
        if self.mass_subp:
            self.ax_mass.step(self.vel_cb_center, mass_cb, ls=ls[-1],
                              color=colors[-3], lw=4., where='mid',
                              label='Double Det.')

    def load_Shen_2017_ddet_models(self):
                
        fpath = self.top_dir + 'Shen_2017_det/1.00_5050.dat'
        time = 10. * u.s #From email.
        m, v, C = np.loadtxt(fpath, skiprows=2, usecols=(0, 1, 17),
                             unpack=True)
        v = (v * u.cm / u.s).to(u.km / u.s)
        m = (m * u.solMass).to(u.g)
        m_step = np.diff(m)
        r_at100s = v.to(u.cm / u.s) * 100. * u.s
        #Ken Shen suggests not simply scaling the density because the explosion
        #is not quite homologous at 10s. Also, the data is already zone centered.
        volume_at100s = 3. / 4. * np.pi * (r_at100s.to(u.cm))**3.
        volume_at100s_step = np.diff(volume_at100s)
        dens_at100s = m_step.to(u.g) / volume_at100s_step
        v_avg = (v[0:-1] + v[1:]) / 2.

        Cdens_at100s = np.multiply(dens_at100s, C[1::]) 

        self.ax_XC.step(v, C, ls=ls[-2], color=colors[-2], lw=4., where='mid',
                        label=r'WD Det.')

        #Bottom plot
        vel_cb, mass_cb, mass_i, mass_m, mass_o, mass_u =\
          make_bin(v_avg, Cdens_at100s, 100. * u.s, self.fb, self.cb) 
        if self.mass_subp:
            self.ax_mass.step(self.vel_cb_center, mass_cb, ls=ls[-2],
                              color=colors[-2], lw=4., where='mid',
                              label=r'WD Det.')

    def add_legend(self):
        self.ax_XC.legend(
          frameon=True, fontsize=fs, numpoints=1, ncol=1, labelspacing=0.1,
          handlelength=1.5, handletextpad=0.5, loc=2, fancybox=True,
          shadow=True, framealpha=0.5, facecolor='ghostwhite', edgecolor='k') 
        plt.tight_layout()    

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            fname = 'Fig_model'
            if self.mass_subp:
                fname += '_with-mass'

            plt.tight_layout()   
            plt.savefig(directory + fname + '.pdf', format='pdf',
                        bbox_inches='tight', dpi=360)
    
    def run_make(self):
        self.set_fig_frame()
        self.add_analysis_regions()
        self.add_tomography_models()
        self.plot_W7_models()
        self.add_hesma_models()
        self.load_Fink_2010_doubledet()
        self.load_Shen_2017_ddet_models()
        self.add_legend()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
if __name__ == '__main__':
    #Plot_Models(mass_subp=False, show_fig=True, save_fig=False)
    Plot_Models(mass_subp=False, show_fig=False, save_fig=True)
    Plot_Models(mass_subp=True, show_fig=False, save_fig=True)

