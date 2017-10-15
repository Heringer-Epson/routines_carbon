#!/usr/bin/env python

import os                                                               
import sys
import time

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors

import colormaps as cmaps
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

class Plot_L_Fe(object):
    """Makes a figure that contains a grid where carbon is plotted color-mapped
    in a luminosity and Fe grid.
    """
    
    def __init__(self, date='early', lm='downbranch', L_range=None, Fes_range=None,
                 T_range=None, colormap_for='L', show_fig=True, save_fig=False):

        self.date = date
        self.lm = lm
        self.L_range = L_range
        self.Fes_range = Fes_range
        self.T_range = T_range
        self.colormap_for = colormap_for
        self.show_fig = show_fig
        self.save_fig = save_fig 

        if self.date == '6d':
            L_max = 0.32e9
        elif self.date == '12d':
            L_max = 2.3e9  
        elif self.date == '19d':
            L_max = 3.5e9  
            
        self.L_scal = np.arange(0.2, 1.61, 0.1)
        self.L = np.array([lum2loglum(L_max * l) for l in self.L_scal]) 
        self.Fes = np.array(['0.00', '0.20', '0.50', '1.00', '2.00', '5.00',
                             '10.00', '20.00', '50.00', '100.00'])               
        
        self._norm = None
        
        self.fig, self.ax = plt.subplots(figsize=(14,8))
        self.title_suffix = ''
        self.name_suffix = ''

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
    
        self.make_plot()     
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Relative \ f}_{\lambda}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs_label)
        self.ax.set_ylabel(y_label,fontsize=self.fs_label)
        self.ax.set_xlim(1500.,10000.)
        self.ax.set_ylim(-.5,4.5)      
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(2.))        
        self.ax.tick_params(labelleft='off')  

    def make_loop_varaibles(self):
        
        if self.L_range is not None:
            L_min, L_max = self.L_range[0], self.L_range[1]            
            aux_L = self.L_scal.astype(float)
            self.loop_L = self.L[(aux_L >= L_min) & (aux_L <= L_max)]
            if L_min == L_max:
                self.title_suffix = (': $L / L_{\\rm{11fe}}'\
                                     + '=' + str(L_min) + '$')
                self.name_suffix = '_L-ratio-' + str(L_min)                    
            else:
                self.title_suffix = (': $' + str(L_min) + '\\leq L'\
                                     + '/ L_{\\rm{11fe}} \\leq '\
                                     + str(L_max) + '$')
                self.name_suffix = '_L-ratio-' + str(L_min) + '-' + str(L_max)                    
        else:
             self.loop_L = self.L   

        if self.Fes_range is not None:
            Fes_min, Fes_max = self.Fes_range[0], self.Fes_range[1]
            aux_Fes = self.Fes.astype('float')
            self.loop_Fes = self.Fes[(aux_Fes >= Fes_min) & (aux_Fes <= Fes_max)]
            self.title_suffix = '_Fe-range'
            if Fes_min == Fes_max:
                self.title_suffix = (': $X(\\rm{Fe}) / X(\\rm{Fe})_{\\rm{11fe}}'\
                                     + '=' + str(Fes_min) + '$')
                self.name_suffix = '_Fe-ratio-' + str(Fes_min)                    
            else:
                self.title_suffix = (': $' + str(Fes_min) + '\\leq X(\\rm{Fe})'\
                                     + '/ X(\\rm{Fe})_{\\rm{11fe}} \\leq '\
                                     + str(Fes_max) + '$')
                self.name_suffix = '_Fe-ratio-' + str(Fes_min) + '-' + str(Fes_max)                    
        else:
             self.loop_Fes = self.Fes 

        if self.T_range is not None:
            T_min, T_max = self.T_range[0], self.T_range[1]
            if T_min == T_max:
                self.title_suffix = (': $T =' + str(T_min) + '$')
                self.name_suffix = '_T-ratio-' + str(T_min)                    
            else:
                self.title_suffix = (': $' + str(T_min) + '\\leq T'\
                                     + '\\leq ' + str(T_max) + '$')
                self.name_suffix = '_T-' + str(T_min) + '-' + str(T_max)                    
            
    def get_colorbar_pars(self):
        
        if self.colormap_for == 'L':
            self.cmap = cmaps.viridis
            self.vmin = float(self.L_scal[0])
            self.vmax = float(self.L_scal[-1])
            self._norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)                       
            self.ticks = self.L_scal
            self.tick_labels = self.L_scal
            self.label = r'$L\ /\ L_{\mathrm{11fe}}$'

        elif self.colormap_for == 'Fe':
            self.cmap = cmaps.plasma
            self.vmin = 1
            self.vmax = len(self.Fes)
            self._norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)                       
            self.ticks = np.arange(self.vmin, self.vmax + 0.1, 1)
            self.tick_labels = self.Fes
            self.label = r'$X(\rm{Fe})\ /\ X(\rm{Fe})_{\mathrm{11fe}}$'

        elif self.colormap_for == 'T':
            self.cmap = cmaps.inferno
            self.vmin = 9500.
            self.vmax = 14500.
            self._norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)                       
            self.ticks = np.arange(self.vmin, self.vmax + 0.1, 1000)
            self.tick_labels = self.ticks.astype(int)
            self.label = r'$T \ \rm{[K]}$'
                                  
    def load_and_plot_observational_spectrum(self):
        
        #Load observed spectrum. 
        fullpath = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'\
                    + 'observational_spectra/2011fe/2011_08_28.pkl')
        with open(fullpath, 'r') as inp:
            pkl = cPickle.load(inp)
            
            self.ax.plot(pkl['wavelength_corr'], pkl['flux_smoothed'], color='k',
                         ls='-', lw=3., zorder=2., label='11fe obs.')
                    
    def load_data(self):
        
        case_folder = path_tardis_output + self.date + '-carbon-grid_v19590_UP/'
        def get_fname(lm, L, Fe): 
            fname = 'Fe0-' + Fe + '_loglum-' + L + '_line_interaction-' + lm 
            fname = case_folder + fname + '/' + fname + '.pkl'
            return fname     
            
        self.w_list, self.f_list, self.T_list, self.c_list = [], [], [], [] 

        for L, Ls in zip(self.loop_L, self.L_scal):
            for j, Fes in enumerate(self.loop_Fes):
                with open(get_fname(self.lm, L, Fes), 'r') as inp:
                    pkl = cPickle.load(inp)
                    to_append = True

                    #Get variables for plotting.
                    T = pkl['t_inner'].value
                    w = pkl['wavelength_corr']
                    f = pkl['flux_smoothed']
                  
                    #Collect variable to ve used for the colormap.
                    if self.colormap_for == 'L':
                        var2color = float(Ls)                  
                    elif self.colormap_for == 'Fe':
                        var2color = j                     
                    elif self.colormap_for == 'T':
                        var2color = T                    

                    c = self.cmap(self._norm(var2color))                       
                    
                    #Check if temperature range is requested and, if so,
                    #if the temperature falls within that range.
                    if self.T_range is not None:
                        T_min, T_max = self.T_range[0], self.T_range[1]
                        if T < T_min or T > T_max:
                            to_append = False
                    
                    #if to_append flag is True, then append variables to
                    #be plotted.
                    if to_append:        
                        self.T_list.append(T)                    
                        self.w_list.append(w)
                        self.f_list.append(f)
                        self.c_list.append(c)

    def plotting(self):
        for w, f, c in zip(self.w_list, self.f_list, self.c_list):
            self.ax.plot(w, f, color=c, ls='-', lw=1., zorder=1.)
        
    def add_colorbar(self):
        aux_mappable = mpl.cm.ScalarMappable(cmap=self.cmap, norm=self._norm)
        aux_mappable.set_array([])
        cbar = plt.colorbar(aux_mappable)
        cbar.set_ticks(self.ticks)
        cbar.ax.tick_params(width=1, labelsize=self.fs_label)
        cbar.set_label(self.label, fontsize=self.fs_label)     
        cbar.set_ticklabels(self.tick_labels)

    def add_title_and_legend(self):
        title = r'11fe_' + self.date + self.title_suffix
        self.ax.set_title(title, fontsize=self.fs_label)
        self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=1)
        plt.tight_layout()                         

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_' + self.date + '-spectra' +
                        self.name_suffix + '_' + self.lm + '_v19590_UP.png',
                        format='png', dpi=360)

    def make_plot(self):
        self.set_fig_frame()
        self.make_loop_varaibles()
        self.get_colorbar_pars()
        self.load_and_plot_observational_spectrum()
        self.load_data()
        self.plotting()
        self.add_colorbar()
        self.add_title_and_legend()
        self.save_figure()
        if self.show_fig:
            plt.show()

Plot_L_Fe(date='19d', lm='macroatom', L_range=[0.39, 0.41], Fes_range=None,
          T_range=None, colormap_for='Fe', show_fig=True, save_fig=True)



#Plot_L_Fe(date='early', lm='macroatom', L_range=[0.99, 1.01], Fes_range=None,
#          T_range=None, colormap_for='Fe', show_fig=True, save_fig=False)
          

                          
#Plot_L_Fe(lm='macroatom', L_range=None, Fes_range=None, T_range=None, 
#          colormap_for='L', show_fig=True, save_fig=False)

#Plot_L_Fe(lm='macroatom', L_range=None, Fes_range=None, T_range=None, 
#          colormap_for='Fe', show_fig=True, save_fig=False)

#Plot_L_Fe(lm='macroatom', L_range=None, Fes_range=None, T_range=None, 
#          colormap_for='T', show_fig=True, save_fig=False)

#Plot_L_Fe(lm='macroatom', L_range=[0.99, 1.01], Fes_range=None, T_range=None, 
#          colormap_for='Fe', show_fig=True, save_fig=True)

#Plot_L_Fe(lm='macroatom', L_range=None, Fes_range=[1.0, 1.0], T_range=None, 
#          colormap_for='L', show_fig=True, save_fig=False)

#Plot_L_Fe(lm='macroatom', L_range=[0.99, 1.01], Fes_range=None, T_range=None, 
#          colormap_for='Fe', show_fig=True, save_fig=True)

#Plot_L_Fe(lm='macroatom', L_range=None, Fes_range=[10.0, 10.0], T_range=None, 
#          colormap_for='T', show_fig=True, save_fig=True)

#Plot_L_Fe(lm='macroatom', L_range=None, Fes_range=None, T_range=[11000., 11500.], 
#          colormap_for='Fe', show_fig=True, save_fig=False)
          
          
          
