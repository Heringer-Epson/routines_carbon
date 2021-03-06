import numpy as np
from astropy import constants as const
from astropy import units as u

def make_bin(vel_read, dens_read, time, fb, cb):
    """
    Description:
    ------------
    Compute quantities in extremely fine bins of 1 km/s which can then
    be integrate to obtain the mass in larger bins. This prevents
    cases where the density between two ejecta layers changes in
    between two coarser bins, leading to a wrong mass integration.
    
    The density in each layer is constant and stored at the upper
    velocity of that layer. Therefore the first v_inner (photosphere)
    is not relevant.
    
    *_read represents the data read from the original source and passed with
    the correct units.
    *_fb represents the fine bins.
    *_cb represents the coarser bins (for plotting purposes.) 
    """
    #=-=-=-=-=-=-=-=-=-=-=-=- get binned mass -=-=-=-=-=-=-=-=-=-=-=-=
    vu = vel_read.unit
    
    #Make coarse bins
    cb_step = cb.to(vu).value    
    vel_cb = np.arange(7100, 31101, cb_step) * vu
    vel_cb_center = (vel_cb.value[0:-1] + vel_cb.value[1:]) / 2.

    #Make finer bins.
    fb_step = fb.to(vu).value
    vel_fb = np.arange(vel_read.value[0], vel_read.value[-1], fb_step) * vu
    dens_fb = np.zeros(len(vel_fb)) * dens_read.unit

    #The binning procedure is tricky: the velocities passed are the velocties
    #at the inner boundary of a layer. The actual density, opacity, abundances,
    #etc, are those of at the upper boundary of a layer (such that the values
    #at the photosphere can be neglected. This is taken into account below.
    for i,vb in enumerate(vel_fb):
        for j,vr in enumerate(vel_read):
            if vr >= vb:
                db = dens_read[j-1]
                break        
        dens_fb[i] = db
    
    #Compute binned mass.
    r_fb = vel_fb * time
    vol = 4. / 3. * np.pi * r_fb ** 3.
    vol_fb = np.diff(vol)
    #Density at photosphere not used.
    mass_fb = np.multiply(vol_fb, dens_fb[1::]).to(u.solMass) 

    #Make coarses bins.
    mass_cb = []
    for (v_inner, v_outer) in zip(vel_cb[0:-1], vel_cb[1:]):
        #Because of the smaller ineq. and array sizes, don't include last vel_fb.
        cb_condition = ((vel_fb[0:-1] >= v_inner) & (vel_fb[0:-1] < v_outer))
        integrated_mass = sum(np.nan_to_num(mass_fb[cb_condition].value))
        #print v_inner, v_outer, integrated_mass
        mass_cb.append(integrated_mass)

    mass_cb = np.asarray(mass_cb) * mass_fb.unit
    
    #Compute integrate quantities in large zones (to printed for tables).
    condition_i = ((vel_fb[1:] > 7850. * u.km / u.s)
                   & (vel_fb[1:] <= 13680. * u.km / u.s))
    condition_m = ((vel_fb[1:] > 13680. * u.km / u.s)
                   & (vel_fb[1:] <= 16064. * u.km / u.s))   
    condition_o = ((vel_fb[1:] > 16064. * u.km / u.s)
                   & (vel_fb[1:] <= 19478. * u.km / u.s))
    condition_u = (vel_fb[1:] > 19478. * u.km / u.s)

    mass_i = sum(mass_fb[condition_i])
    mass_m = sum(mass_fb[condition_m])
    mass_o = sum(mass_fb[condition_o])
    mass_u = sum(mass_fb[condition_u])
    
    return vel_cb, mass_cb, mass_i, mass_m, mass_o, mass_u


def get_binned_maxima(vel_read, qtty_read, fb, cb):
    """
    Compute quantities in extremely fine bins of 1 km/s and then retrieve
    the maxima in given regions.) 
    """
    #=-=-=-=-=-=-=-=-=-=-=-=- get binned mass -=-=-=-=-=-=-=-=-=-=-=-=
    vu = vel_read.unit

    #Make finer bins.
    fb_step = fb.to(vu).value
    vel_fb = np.arange(vel_read.value[0], vel_read.value[-1], fb_step) * vu
    qtty_fb = np.zeros(len(vel_fb)) * qtty_read.unit
                
    for i,vb in enumerate(vel_fb):
        for j,vr in enumerate(vel_read):
            if vr >= vb:
                qr = qtty_read[j-1]
                #print '  gotcha', vb, vr, qr
                break        
        qtty_fb[i] = qr

    #Get qtty in regions.
    condition_i = ((vel_fb[0:-1] > 7850. * u.km / u.s)
                   & (vel_fb[0:-1] <= 13680. * u.km / u.s))
    condition_m = ((vel_fb[0:-1] >= 13681. * u.km / u.s)
                   & (vel_fb[0:-1] <= 16064. * u.km / u.s))   
    condition_o = ((vel_fb[0:-1] >= 16065. * u.km / u.s)
                   & (vel_fb[0:-1] <= 19168. * u.km / u.s))
    condition_u = (vel_fb[0:-1] >= 19169. * u.km / u.s)

    qtty_i = qtty_fb[condition_i]
    qtty_m = qtty_fb[condition_m]
    qtty_o = qtty_fb[condition_o]
    qtty_u = qtty_fb[condition_u]
    
    #print zip(vel_fb[condition_o],qtty_fb[condition_o])
    #print zip(vel_fb[condition_u],qtty_fb[condition_u])
    
    return max(qtty_i), max(qtty_m), max(qtty_o), max(qtty_u)

##Test##
if __name__ == '__main__': 

    density = np.arange(1., 30., 1.) * u.g / u.cm**3.
    velocity = np.arange(1000., 30000., 1000.) * u.km / u.s
    time = 100. * u.s
    make_bin(velocity, density, time, 10. * u.km / u.s, 200. * u.km / u.s)

