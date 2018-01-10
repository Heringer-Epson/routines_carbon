#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import numpy as np
import tardis   
import matplotlib.pyplot as plt
import matplotlib as mpl
import cPickle
import new_colormaps as cmaps
from matplotlib.ticker import MultipleLocator
from astropy import units as u
from matplotlib import cm
from matplotlib import colors

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

date_list = np.array(['5.9', '9.0', '12.1', '16.1', '19.1'])
scales = ['0.00', '0.05', '0.1', '0.2', '0.5', '1.00', '2.00', '5.00', '10.00']
labels = ['0', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10']
mass_fractions = np.asarray(scales).astype(float) * 0.01
Ns = len(scales)
Nd = len(date_list)
tick_pos = np.arange(1.5, Ns + 0.51, 1.)

pEW_min, pEW_max = 0., 5.

cmap_pEW = plt.cm.get_cmap('Blues')

fs = 20.

def lum2loglum(lum):
    return str(format(np.log10(lum), '.3f'))

texp2date = {'3.7': '2011_08_25', '5.9': '2011_08_28', '9.0': '2011_08_31',
             '12.1': '2011_09_03', '16.1': '2011_09_07', '19.1': '2011_09_10',
             '22.4': '2011_09_13', '28.3': '2011_09_19'}

texp2v = {'3.7': '13300', '5.9': '12400', '9.0': '11300',
          '12.1': '10700', '16.1': '9000', '19.1': '7850',
          '22.4': '6700', '28.3': '4550'}

texp2L = {'3.7': 0.08e9, '5.9': 0.32e9, '9.0': 1.1e9,
          '12.1': 2.3e9, '16.1': 3.2e9, '19.1': 3.5e9,
          '22.4': 3.2e9, '28.3': 2.3e9}

class Plateaus_Plot(object):
    
    def __init__(self, show_fig=True, save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig   
        
        self.im_pEW = None
        self.im_vel = None
            
        self.F = {}
        self.F['fig'] = plt.figure(figsize=(8,22))
        for i in range(Nd + 1):
            self.F['axl' + str(i)] = plt.subplot(6, 1, i + 1)
        
        self.F['fig'].subplots_adjust(left=0.16, right=0.80) 
        self.F['pEW_bar_ax'] = self.F['fig'].add_axes([0.84, 0.32, 0.03, 0.35])    
        
        plt.subplots_adjust(hspace=0.05)
        
        self.run_make()

    def set_frame(self):

        #Make labels.
        y_label = (r'$X(\rm{C}) \ \mathrm{[\%]}$ at 13300 $\leq\ v \ \leq$'\
                   r' 16000$ \ \mathrm{[km\ s^{-1}]}$')
        x_label = (r'$X(\rm{C}) \ \mathrm{[\%]}$ at 7850 $\leq\ v \ \leq$'\
                   r' 13300$ \ \mathrm{[km\ s^{-1}]}$')
        self.F['fig'].text(0.04, 0.5, y_label, va='center',
                           rotation='vertical', fontsize=fs)        
        self.F['fig'].text(0.22, 0.04, x_label, va='center',
                           rotation='horizontal', fontsize=fs)  
        
        #Frame Settings.
        for i in range(Nd + 1):
            idx = str(i)
            
            self.F['axl' + idx].tick_params(
              axis='y', which='major',labelsize=fs, pad=8)       
            self.F['axl' + idx].tick_params(
              axis='x', which='major', labelsize=fs, pad=8)
            self.F['axl' + idx].minorticks_off()  

            self.F['axl' + idx].set_xticks(tick_pos)
            self.F['axl' + idx].set_yticks(tick_pos)
            self.F['axl' + idx].set_yticklabels(labels)

            if i != Nd:
                self.F['axl' + idx].tick_params(labelbottom='off')
            else:
                self.F['axl' + idx].set_xticklabels(
                  labels, rotation='vertical')

    def load_observational_data(self):
        directory = ('/home/heringer/Research/routines_11fe-05bl/INPUT_FILES/'
                     + 'observational_spectra/2011fe/')
        
        for date in date_list:
            with open(directory + texp2date[date] + '.pkl', 'r') as inp:
                pkl = cPickle.load(inp)

                if np.isnan(pkl['pEW_fC']):
                    pkl['pEW_fC'], pkl['pEW_unc_fC'] = 0., 0.

                self.F[date + '_pEW_fC'] = pkl['pEW_fC']
                self.F[date + '_pEW_unc_fC'] = pkl['pEW_unc_fC']

    def retrieve_data(self, date):

        pEW_array, pEW_unc_array = [], []
        t = str(int(round(float(date))))

        case_folder = '11fe_' + t + 'd_C-plateaus_scaling/'
        def get_fname(s1, s2): 
            fname = 'C-F2-' + s2 + '_C-F1-' + s1
            fname = case_folder + fname + '/' + fname + '.pkl'        
            return path_tardis_output + fname     

        for s2 in scales:
            for s1 in scales:
                if float(s2) >= float(s1):
                    fpath = get_fname(s1, s2)
                    with open(fpath, 'r') as inp:
                        pkl = cPickle.load(inp)

                        pEW = pkl['pEW_fC']
                        #pEW_unc = pkl['pEW_unc_fC']
                        pEW_unc = 0.

                        if pEW <= pEW_min:
                            pEW = pEW_min
                        if pEW >= pEW_max:
                            pEW = pEW_max

                else:
                    pEW = pEW_min
                
                pEW_array.append(pEW)                    
                pEW_unc_array.append(pEW_unc)                    

        pEW_array = np.nan_to_num(np.array(pEW_array))
        pEW_unc_array = np.nan_to_num(np.array(pEW_unc_array))

        return pEW_array, pEW_unc_array
                    

    def loop_dates(self):
        
        #Canvas for plotting pEW contours.
        x = np.arange(1.5, Ns + 0.6, 1.)
        y = np.arange(1.5, Ns + 0.6, 1.)
        X, Y = np.meshgrid(x, y)

        for i, date in enumerate(date_list):
            
            #Collect data to be plotted.
            pEW_values, pEW_unc_values = self.retrieve_data(date)

            #Plot data.
            ##~=~=~ pEW_2D[s2,s1]
            pEW_2D = np.reshape(pEW_values, (Ns, Ns))
            pEW_unc_2D = np.reshape(pEW_unc_values, (Ns, Ns))

            #imshow
            self.im_pEW = self.F['axl' + str(i)].imshow(
              pEW_2D, interpolation='none', aspect='auto',
              extent=[1., Ns + 1., 1., Ns + 1.], origin='lower',
              cmap=cmap_pEW, vmin=pEW_min, vmax=pEW_max)

            #Mark pixels whose quantity value match the observed value.

            self.F[date + '_pEW-match'] = []
            self.F[date + '_vel-match'] = []


            for l, s2 in enumerate(scales):
                for f, s1 in enumerate(scales):
                    
                    if float(s2) >= float(s1):
                        
                        #pEW
                        #pEW_quad_unc = np.sqrt(self.F[date + '_pEW_unc_fC']**2.
                        #                       + pEW_unc_2D[l, f]**2.)

                        pEW_quad_unc = self.F[date + '_pEW_unc_fC']

                        if (abs(self.F[date + '_pEW_fC'] - pEW_2D[l, f])
                            <= 5. * pEW_quad_unc):
                            
                            self.F['axl' + str(i)].add_patch(
                              mpl.patches.Rectangle((f + 1, l + 1), 1., 1.,
                              fill=False, snap=False,
                              color='gold', zorder=2.))            
            
                            self.F[date + '_pEW-match'].append([l, f])

   
    def make_bottom_plots(self):
        
        #Mask non-physical region
        blank_2D = np.zeros((Ns, Ns))
        blank_2D.fill(np.nan)

        im = self.F['axl' + str(Nd)].imshow(
          blank_2D, interpolation='none', aspect='auto',
          extent=[1., Ns + 1., 1., Ns + 1.], origin='lower',
          cmap=cmap_pEW, vmin=pEW_min, vmax=pEW_max)           

        #Mark region of agreement.
        for l, s2 in enumerate(scales):
            for f, s1 in enumerate(scales):        
                flag_pEW = True
                
                for i, date in enumerate(date_list):
                    if [l, f] not in self.F[date + '_pEW-match']:
                        flag_pEW = False
                
                if flag_pEW:
                    self.F['axl' + str(Nd)].add_patch(
                      mpl.patches.Rectangle((f + 1, l + 1), 1., 1.,
                      fill=False, snap=False,
                      color='gold', zorder=2.))                      

    def make_colorbars(self):

        ########################Plot the pEW colorbar.########################
        ticks = np.arange(pEW_min, pEW_max + 0.1, 1)
        tick_labels = np.copy(ticks).astype(int).astype(str)
        tick_labels[-1] = r'$\geq$ ' + str(int(pEW_max))

        cbar_pEW = self.F['fig'].colorbar(
          self.im_pEW, cax=self.F['pEW_bar_ax'], orientation='vertical',
          ticks=ticks)
        
        cbar_pEW.ax.tick_params('y', length=8, width=1, which='major',
                                labelsize=fs)
        cbar_pEW.ax.set_yticklabels(tick_labels)

        #Set label.
        cbar_pEW_label = (r'pEW$\ \mathrm{[\AA]}$ of $\rm{C}\,\mathrm{II}$'\
                          + r'$ \ \lambda$6580')
        cbar_pEW.set_label(cbar_pEW_label, fontsize=fs, labelpad=-10.)

    def plot_hatched_region(self):

        for i in range(Nd + 1):
            for l, s2 in enumerate(scales):
                for f, s1 in enumerate(scales):
                    if float(s2) < float(s1):
                        self.F['axl' + str(i)].add_patch(
                          mpl.patches.Rectangle((f + 1, l + 1), 1., 1.,
                          hatch='//', fill=True, snap=False, color='grey'))

    def save_figure(self):        
        if self.save_fig:
            directory = './../OUTPUT_FILES/FIGURES/'
            plt.savefig(directory + 'Fig_11fe_C-plateaus.pdf',
                        format='pdf', dpi=360)
            plt.savefig(directory + 'Fig_11fe_C-plateaus.png',
                        format='png', dpi=360)

    def run_make(self):
        self.set_frame()
        self.load_observational_data()
        self.loop_dates()
        self.make_bottom_plots()
        self.make_colorbars()
        self.plot_hatched_region()
        self.save_figure()
        if self.show_fig:
            plt.show()
        plt.close()        

if __name__ == '__main__':
    Plateaus_Plot(show_fig=True, save_fig=True)

    
