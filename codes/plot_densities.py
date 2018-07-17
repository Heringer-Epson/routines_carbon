#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import constants as const
from astropy import units as u

mpl.rcParams['hatch.color'] = '#e41a1c'

M_sun = const.M_sun.to('g').value
fs = 26.

model_directories = ['Seitenzahl_2013_ddt/', 'Seitenzahl_2016_gcd/',
                     'Parkmor_2010_merger/', 'Sim_2012_doubledet/']
model_isotope_files = [
  'ddt_2013_n100_isotopes.dat', 'gcd_2016_gcd200_isotopes.dat',
  'merger_2012_11_09_isotopes.dat', 'doubledet_2012_csdd-s_isotopes.dat']
model_time = [100.22 * u.s, 100.22 * u.s, 100.07 * u.s, 100.06 * u.s]
labels = ['DDT N100', 'GCD200', 'Merger', 'Double det.']
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
    Makes three figures, including figure 5 in the carbon paper.
    These figures show how the mass fraction of carbon, mass of carbon, or
    total (mass) density of multiple literature models vary across the ejecta.
    The chosen x-coordiante is the velocity. 

    Notes:
    ------
    Figure Fig_model_implII.pdf is really just another attempt to make this
    plot and does not provide any different insigths than the figure produced
    by plot_model_comparison.py
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_model_implII.pdf
    ./../OUTPUT_FILES/FIGURES/Fig_density.pdf
    ./../OUTPUT_FILES/FIGURES/Fig_C_density.pdf
    """
    
    def __init__(self, show_fig=True, save_fig=True):
        
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        self.fig_model = plt.figure(1, figsize=(14.,8.))
        self.ax_model = self.fig_model.add_subplot(111) 
        self.fig_dens = plt.figure(2, figsize=(10.,10.))
        self.ax_dens = self.fig_dens.add_subplot(111) 
        self.fig_Cdens = plt.figure(3, figsize=(10.,10.))
        self.ax_Cdens = self.fig_Cdens.add_subplot(111) 
        
        self.top_dir = './../INPUT_FILES/model_predictions/'

        self.run_make()

    def set_fig_frame(self):
        
        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = r'$X(\rm{C})$'

        self.ax_model.set_xlabel(x_label, fontsize=fs)
        self.ax_model.set_ylabel(y_label, fontsize=fs)
        self.ax_model.set_yscale('log')
        self.ax_model.set_xlim(0., 30000.)
        self.ax_model.set_ylim(1.e-4, 2.)
        self.ax_model.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_model.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_model.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax_model.tick_params(
          'both', length=4, width=1, which='minor', direction='in')
        self.ax_model.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax_model.xaxis.set_major_locator(MultipleLocator(5000.))  

        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = r'$\rho\ \ \rm{[g\ \ cm^{-3}]}$'

        self.ax_dens.set_xlabel(x_label, fontsize=fs + 4)
        self.ax_dens.set_ylabel(y_label, fontsize=fs + 4)
        self.ax_dens.set_yscale('log')
        self.ax_dens.set_xlim(5000., 30000.)
        self.ax_dens.set_ylim(1.e-4, 1.e1)
        self.ax_dens.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_dens.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_dens.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax_dens.tick_params(
          'both', length=4, width=1, which='minor', direction='in')
        self.ax_dens.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax_dens.xaxis.set_major_locator(MultipleLocator(5000.))  

        x_label = r'$v\ \ \rm{[km\ \ s^{-1}]}$'
        y_label = r'$\rho\ \ \rm{[g\ \ cm^{-3}]}$'

        self.ax_Cdens.set_xlabel(x_label, fontsize=fs + 4)
        self.ax_Cdens.set_ylabel(y_label, fontsize=fs + 4)
        self.ax_Cdens.set_yscale('log')
        self.ax_Cdens.set_xlim(5000., 30000.)
        self.ax_Cdens.set_ylim(1.e-6, 1.e0)
        self.ax_Cdens.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax_Cdens.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax_Cdens.tick_params(
          'both', length=8, width=1, which='major', direction='in')
        self.ax_Cdens.tick_params(
          'both', length=4, width=1, which='minor', direction='in')
        self.ax_Cdens.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax_Cdens.xaxis.set_major_locator(MultipleLocator(5000.))  

    def add_analysis_regions(self):

        #Unconstrained regions
        color = '#bababa'
        alpha = 0.5
        hatch = ''
        fill = True
        lw=None
      
        self.ax_model.add_patch(
          mpl.patches.Rectangle((16000., 1.e-5), 40000. - 16000., 2.0 - 1.e-5,
          hatch=hatch, fill=fill, lw=lw, color=color, alpha=alpha, zorder=1))        

        self.ax_model.add_patch(
          mpl.patches.Rectangle((0., 1.e-5), 7850. - 0., 0.005 - 1.e-5,
          hatch=hatch, fill=fill, lw=lw, color=color, alpha=alpha, zorder=1)) 

        #Constrained regions.
        color = '#e41a1c'
        alpha = 1.0
        hatch = '//'
        fill = False
        lw=0.
        
        self.ax_model.add_patch(
          mpl.patches.Rectangle((13300., 1.e-5), 16000. - 13300., 0.005 - 1.e-5,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha))

        self.ax_model.add_patch(
          mpl.patches.Rectangle((13300., 0.05), 16000. - 13300., 2.0 - 0.05,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha))       

        self.ax_model.add_patch(
          mpl.patches.Rectangle((7850., 0.05), 13300. - 7850., 2.0 - 0.05,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha)) 

        self.ax_model.add_patch(
          mpl.patches.Rectangle((0., 0.005), 7850. - 0., 2.0 - 0.005,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha)) 

        self.ax_model.add_patch(
          mpl.patches.Rectangle((7850., 0.005), 9000. - 7850, 0.05 - 0.005,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha)) 

        self.ax_model.add_patch(
          mpl.patches.Rectangle((9000., 0.005), 10700. - 9000., 0.05 - 0.005,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha)) 
                
        self.ax_model.add_patch(
          mpl.patches.Rectangle((10700., 0.005), 11300. - 10700., 0.05 - 0.005,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha)) 
                
        self.ax_model.add_patch(
          mpl.patches.Rectangle((11300., 0.01), 12400. - 11300., 0.05 - 0.01,
          hatch=hatch, fill=fill, lw=lw, color=color, zorder=1, alpha=alpha)) 

        color = 'limegreen'
        alpha = 0.5
        hatch = ''
        fill = True      
        lw=0.

        self.ax_model.add_patch(
          mpl.patches.Rectangle((7850., 1.e-5), 13300. - 7850., 0.005 - 1.e-5,
          hatch=hatch, fill=fill, lw=lw, color=color, alpha=alpha, zorder=1))        

        self.ax_model.add_patch(
          mpl.patches.Rectangle((11300., 0.005), 12400. - 11300., 0.01 - 0.005,
          hatch=hatch, fill=fill, lw=lw, color=color, alpha=alpha, zorder=1)) 

        self.ax_model.add_patch(
          mpl.patches.Rectangle((12400., 0.005), 16000. - 12400., 0.05 - 0.005,
          hatch=hatch, fill=fill, lw=lw, color=color, alpha=alpha, zorder=1))    

    def add_tomography_models(self):             
        
        #SN 2011fe.
        fpath = ('/home/heringer/Research/TARDIS-bundle/INPUT_FILES/'
                 + 'Mazzali_2011fe/ejecta_layers.dat')                   
        time = 100. * u.s
        v, m, dens, C = np.loadtxt(fpath, skiprows=2, usecols=(1, 2, 3, 9),
                                   unpack=True)
        v = v * u.km / u.s
        dens = 10.**dens
        m_step = np.array([0.] + list(np.diff(m)))
        m_C = np.cumsum(np.multiply(m_step, C))
                
        Cdens_100 = np.multiply(dens, C) 
    
        #Plot original work.
        self.ax_model.step(v, C, ls='-', color='k', lw=4., where='post',
                           zorder=4., label=r'SN 2011fe')

        self.ax_dens.step(v, dens, ls='-', color='k', lw=4., where='post',
                          zorder=4., label=r'SN 2011fe')
        self.ax_Cdens.step(v, dens, ls='-', color='k', lw=4., where='post',
                           zorder=4., label=r'SN 2011fe')

    def add_hesma_models(self):
        for i, model_dir in enumerate(model_directories):

            fpath = self.top_dir + model_dir + model_isotope_files[i]
            v, dens, C = read_hesma(fpath)
            dens_100 = dens * (model_time[i] / (100. *u.s))**3.
            m, m_C = get_mass(v, dens, C, model_time[i])
            m_C = np.cumsum(m_C)

            Cdens_100 = np.multiply(dens_100, C) 
            
            self.ax_model.step(v, C, ls=ls[i], color=colors[i], lw=4., where='post',
                             label=labels[i])    
            self.ax_dens.step(v, dens_100, ls=ls[i], color=colors[i], lw=4.,
                              where='post', label=labels[i])  
            self.ax_Cdens.step(v, Cdens_100, ls=ls[i], color=colors[i], lw=4.,
                               where='post', label=labels[i])  
            
    def load_Shen_2017_ddet_models(self):
                
        fpath = self.top_dir + 'Shen_2017_det/1.00_5050.dat'
        time = 10. * u.s #From email.
        m, v, C = np.loadtxt(fpath, skiprows=2, usecols=(0, 1, 17),
                             unpack=True)
        v = (v * u.cm / u.s).to(u.km / u.s)
        m = (m * u.solMass).to(u.g)
        m_step = np.diff(m)
        r_100 = v.to(u.cm / u.s) * 100. * u.s
        #Ken Shen suggests not simply scaling the density because the explosion
        #is not quite homologous at 10s. Also, the data is already zone centered.
        volume_100 = 3. / 4. * np.pi * (r_100.to(u.cm))**3.
        volume_100_step = np.diff(volume_100)
        dens_100 = m_step.to(u.g) / volume_100_step
        v_avg = (v[0:-1] + v[1:]) / 2.

        Cdens_100 = np.multiply(dens_100, C[1::]) 
                
        self.ax_model.step(v, C, ls=ls[-2], color=colors[-2], lw=4., where='mid',
                         label=r'WD det.')
        self.ax_dens.step(v_avg, dens_100, ls=ls[-2], color=colors[-2], lw=4.,
                          where='mid', label=r'WD det.')  
        self.ax_Cdens.step(v_avg, Cdens_100, ls=ls[-2], color=colors[-2], lw=4.,
                           where='mid', label=r'WD det.')  

    def plot_W7_models(self):
        
        time = (0.000231481 * u.d).to(u.s)
        fpath = self.top_dir + 'W7/W7_model.dat'
        v, dens = np.loadtxt(fpath, skiprows=2, usecols=(1, 2), unpack=True)
        v = v * u.km / u.s
        dens = 10.**dens * u.g / u.cm**3
        dens_100 = dens * (time / (100. * u.s))**3.

        fpath = self.top_dir + 'W7/W7_abundances.dat'
        C = np.loadtxt(fpath, skiprows=0, usecols=(6,), unpack=True)
        m, m_C = get_mass(v, dens, C, time)
        m_C = np.cumsum(m_C)

        Cdens_100 = np.multiply(dens_100, C) 

        self.ax_model.step(v, C, ls=ls[-1], color=colors[-1], lw=4., where='post',
                         label='W7')
        self.ax_dens.step(v, dens_100, ls=ls[-1], color=colors[-1], lw=4.,
                          where='post', label='W7') 
        self.ax_Cdens.step(v, Cdens_100, ls=ls[-1], color=colors[-1], lw=4.,
                           where='post', label='W7') 
                          
    def add_legend(self):
        self.ax_model.legend(frameon=True, fontsize=fs, numpoints=1, ncol=1,
                       labelspacing=0.05, loc=2) 
        self.ax_dens.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1,
                       labelspacing=0.05, loc=1) 
        self.ax_Cdens.legend(frameon=False, fontsize=fs, numpoints=1, ncol=1,
                       labelspacing=0.05, loc=1) 
        plt.tight_layout()    

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            
            plt.figure(1)
            plt.tight_layout()   
            plt.savefig(directory + 'Fig_model_implII.pdf', format='pdf',
                bbox_inches='tight', dpi=360)
            
            plt.figure(2)
            plt.tight_layout()   
            plt.savefig(directory + 'Fig_density.pdf', format='pdf',
                bbox_inches='tight', dpi=360)

            plt.figure(3)
            plt.tight_layout()   
            plt.savefig(directory + 'Fig_C_density.pdf', format='pdf',
                bbox_inches='tight', dpi=360)
    
    def run_make(self):
        self.set_fig_frame()
        self.add_analysis_regions()
        self.add_tomography_models()
        self.add_hesma_models()
        self.load_Shen_2017_ddet_models()
        self.plot_W7_models()
        self.add_legend()
        self.save_figure()
        if self.show_fig:
            plt.show()
        
if __name__ == '__main__':
    Plot_Models(show_fig=True, save_fig=True)
    
