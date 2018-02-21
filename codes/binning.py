import numpy as np
from astropy import constants as const
from astropy import units as u

def make_bin(vel_read, dens_read, time, fb, cb):
    """
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
    vel_cb = np.arange(7100, 23101, cb_step) * vu
    vel_cb_center = (vel_cb.value[0:-1] + vel_cb.value[1:]) / 2.

    #Make finer bins.
    fb_step = fb.to(vu).value
    vel_fb = np.arange(vel_read.value[0], vel_read.value[-1], fb_step) * vu
    dens_fb = np.zeros(len(vel_fb)) * dens_read.unit
                
    for i,vb in enumerate(vel_fb):
        idx = next(j for j,vr in enumerate(vel_read) if vr >= vb)
        #print idx, vb
        dens_fb[i] = dens_read[idx]
    
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
        mass_cb.append(sum(mass_fb[cb_condition].value))

    mass_cb = np.asarray(mass_cb) * mass_fb.unit
    
    #Compute integrate quantities in large zones (to printed for tables).
    condition_i = ((vel_fb[1:] > 7850. * u.km / u.s)
                   & (vel_fb[1:] < 13300. * u.km / u.s))
    condition_o = ((vel_fb[1:] > 13300. * u.km / u.s)
                   & (vel_fb[1:] < 16000. * u.km / u.s))   
    condition_t = (vel_fb[1:] > 7850. * u.km / u.s)

    mass_i = sum(mass_fb[condition_i])
    mass_o = sum(mass_fb[condition_o])
    mass_t = sum(mass_fb[condition_t])
    
    return vel_cb, mass_cb, mass_i, mass_o, mass_t
    
##Test##
if __name__ == '__main__': 

    density = np.arange(1., 30., 1.) * u.g / u.cm**3.
    velocity = np.arange(1000., 30000., 1000.) * u.km / u.s
    time = 100. * u.s
    make_bin(velocity, density, time, 10. * u.km / u.s, 200. * u.km / u.s)

