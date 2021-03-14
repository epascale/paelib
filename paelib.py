import numpy as np
from scipy.optimize import  brentq
import matplotlib.pyplot as plt
import astropy.units as u

###########################################################################
# Module name: Planets and Exoplanets
# This is a collection of numerical functions in support of lab activities
###########################################################################

def anomaly(t, periastron_time, eccentricity, period, full=False):
    '''
    Estimates true anomaly from time and orbital parameters
    
    Parameters
    ----------
    t : scalar or array like
        time in commensurable units of time
    periastron_time : float
        time of periastron, same units as t
    eccentricity : float
        orbital eccentricity
    period : float
        orbital period. Same units as t
    full : bool
        when False (defult), returns true anomaly.
        when True, returns the tuple (true anomaly, eccentric anomaly, mean anomaly)
    
    Returns
    -------
    out : scalar or array like  
    '''  
    
    kepler_eq = lambda E, M, e : E - e*np.sin(E) - M
    method = brentq
    mean_anomaly = (2.0*np.pi*(t-periastron_time)/period) % (2.0*np.pi)
    
    try:
        iterator = iter(t)
        eccentric_anomaly = [method(kepler_eq, 0.0, 2.0*np.pi, args=(ma, eccentricity) ) for ma in mean_anomaly]
        eccentric_anomaly = np.array(eccentric_anomaly)
    except TypeError:
        eccentric_anomaly = method(kepler_eq, 0.0, 2.0*np.pi, args=(mean_anomaly, eccentricity) )
        
    cs = np.sqrt(1.0-eccentricity)*np.cos(0.5*eccentric_anomaly)
    si = np.sqrt(1.0+eccentricity)*np.sin(0.5*eccentric_anomaly)
    true_anomaly = 2.0*np.arctan2(si, cs)
    
    if full:
        return true_anomaly, eccentric_anomaly, mean_anomaly
    
    return true_anomaly

def radial_velocity(tt, periastron_time, K, period, eccentricity, argument_pericenter):
    ''''
    Calculate radial velocity curve
    
    Parameters
    ----------
    
    t : scalar or array like
        time in commensurable units of time
    periastron_time : float
        time of periastron, same unit as t
    K : scalar
        radial velocity semi-amplitude
    period : float
        orbital period. Same unit as t
    eccentricity : float
        orbital eccentricity
    argument_pericenter: scalar
       argument of the pericenter in radians
    
    Returns
    -------
    out: scalar or array like
       mean anomaly (rad), radial velocity (K.unit)
    '''
    
    true_anomaly, ea, ma = anomaly(tt, periastron_time, eccentricity, period, full=True)
    
    vr = K * (np.cos(argument_pericenter + true_anomaly) + eccentricity*np.cos(argument_pericenter))
    
    return ma, vr


def plot_confidence_level(cov, pos, ax, cl='1s', **kwargs):
  '''
  Plot 2D confidence levels in given axes
  
  Parameters
  ----------
  
  cov : array 
        covariance matrix. The dimension has to be 2x2
  pos : array 
        (x, y) posion of the centre of the confidence level ellipse 
  ax  : maptlotlib axis
        axis where to plot the ellispse
  cl  : confidence level 
        use '1s' for 68% CL, '2s' for the 95% CL, '3s' for the 99.73%, ... , '6s'
  '''
  
  from scipy.optimize import brenth
  from scipy.special import gammainc as gammap
  from matplotlib.patches import Ellipse
  
  clevs = {'1s': 0.682689492137, '2s': 0.954499736104, '3s': 0.997300203937, 
           '4s': 0.999936657516, '5s': 0.999999426697, '6s': 0.999999998027}

  # Find delta_chi2 corresponding to desired C.L.
  gp = lambda x: gammap(cov.shape[0]/2.0, x/2.0)-clevs[cl] 
  delta_chi2 = brenth(gp, 0.5, 100.0)
  
  # Estimate Fisher matrix
  FM = np.linalg.inv(cov)
  
  # Find eigenvalues and eigenvectors of FM, and order them in descending order
  vals, vecs = np.linalg.eigh(FM)
  idx = np.argsort(vals)[::-1]
  vals = vals[idx]; vecs = vecs[idx]
  
  theta = np.rad2deg(np.arctan2(vecs[0,0], vecs[1,0]) )
  width, height = 2 *  np.sqrt(delta_chi2/vals)
  ellip = Ellipse(xy=pos, width=width, height=height, angle=theta-90, **kwargs)

  ax.add_artist(ellip)
    
  return ellip, vals


if __name__ == "__main__":
    # Code below for testing purposes only. Please ignore.
    planet = {
        'e': 0.25 * u.dimensionless_unscaled,
        'a': 0.031 * u.au,      
        'P': 2.2185733 * u.day,  
        'tp': 0.5 * u.day,       
        'arg_pericenter': 0.0*u.deg 
        }
    
    t = np.linspace(0.0,   2*planet['P'], 1024)
    K = 15.0 * u.m/u.s
    
    for omega, cc in zip([0.0*u.deg, 90.0*u.deg, 270.0*u.deg], ['r', 'b', 'g']):
        ph, vr = radial_velocity(t, planet['tp'], K, planet['P'], planet['e'], np.deg2rad(omega))
        plt.plot(t, vr, '.'+cc, label='$\omega = $ {:.0f} deg'.format(omega)); 
    
    plt.grid()
    plt.legend()
    plt.xlabel('time [days]')
    plt.ylabel('Rad. Vel. [m/s]')
    plt.ion();plt.show()
