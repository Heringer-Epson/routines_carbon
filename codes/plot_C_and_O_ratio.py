#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy import constants as const
from astropy import units as u

fs = 26.

k_B = const.k_B.to(u.eV / u.K)
CII_pot = 11.26030 * u.eV #From Marion+ 2006
OII_pot = 13.61806 * u.eV #From Marion+ 2006

#Degeneracy factors **for the ground state**.
#Can be retrieved from a TARDIS sim via
#pd.read_hdf(fpath, '/simulation/plasma/g').loc[Z,ion,0]
g_CI, g_CII = 1., 2.
g_OI, g_OII = 5., 4.

def Boltz_factor(_T):
    return ((g_OI / g_OII) * (g_CII / g_CI)
            * np.exp((OII_pot - CII_pot) / (k_B * _T)))
    
class Plot_Boltz(object):
    '''
    Description:
    ------------
    Computes the relative fraction of singly ionized ions of oxygen to carbon.
    Relative is in the sense that these fraction are computed with respect to
    the number density of neutral atoms for each element.  
    
    The calculation is done by using the Saha equation. The 'g'-factor to
    account for degeneracy states was not included.      
    
    Outputs:
    --------
    ./../OUTPUT_FILES/FIGURES/Fig_Boltz_factor.pdf       
    '''
    
    def __init__(self, show_fig=True, save_fig=True):
        
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        self.fig = plt.figure(1, figsize=(10.,10.))
        self.ax = self.fig.add_subplot(111) 
        
        self.make_plot()

    def set_fig_frame(self):
        x_label = r'$T\ \mathrm{[K]}$'
        y_label = (
          r'$\frac{\frac{n(\mathrm{O_{I}})}{n(\mathrm{O_{II}})}}'\
          r'{\frac{n(\mathrm{C_{I}})}{n(\mathrm{C_{II}})}} \approx '\
          r'e^{\frac{(\epsilon_{\mathrm{OII}} - \epsilon_{\mathrm{CII}})}{KT}}$')

        self.ax.set_xlabel(x_label, fontsize=fs)
        self.ax.set_ylabel(y_label, fontsize=fs)
        self.ax.set_yscale('log')
        self.ax.set_xlim(5000., 20000.)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=8)       
        self.ax.tick_params(axis='x', which='major', labelsize=fs, pad=8)
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(5000.))  

    def plot_Boltz_factor(self):
        T_array = np.arange(4500., 20500., 10.) * u.K
        factor = Boltz_factor(T_array)
        
        self.ax.plot(T_array, factor, ls='-', lw=3., marker='None', color='b')

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            fpath = './../OUTPUT_FILES/FIGURES/Fig_Boltz_factor.pdf'
            plt.savefig(fpath , format='pdf')
        if self.show_fig:
            plt.show() 
        plt.close()

    def make_plot(self):
        self.set_fig_frame()
        self.plot_Boltz_factor()
        self.manage_output()
        
if __name__ == '__main__':
    Plot_Boltz(show_fig=True, save_fig=True)

