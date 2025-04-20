import pyIGRF.calculate as calculate
import numpy as np

from .geo import subsol
from mosgim.utils.time_util import sec_of_day
# GEOMAGNETIC AND MODIP COORDINATES SECTION

# North magnetic pole coordinates, for 2017
# Taken from here: http://wdc.kugi.kyoto-u.ac.jp/poles/polesexp.html
POLE_THETA = np.pi/2 - np.radians(80.5)
POLE_PHI = np.radians(-72.6)

# Geodetic to Geomagnetic transform: http://www.nerc-bas.ac.uk/uasd/instrums/magnet/gmrot.html
GEOGRAPHIC_TRANSFORM = np.array([
    [np.cos(POLE_THETA)*np.cos(POLE_PHI), np.cos(POLE_THETA)*np.sin(POLE_PHI), -np.sin(POLE_THETA)],
    [-np.sin(POLE_PHI), np.cos(POLE_PHI), 0],
    [np.sin(POLE_THETA)*np.cos(POLE_PHI), np.sin(POLE_THETA)*np.sin(POLE_PHI), np.cos(POLE_THETA)]
])


@np.vectorize
def geo2mag(theta: float, phi: float, date) -> tuple[float, float]:

    ut = sec_of_day(date)
    doy = date.timetuple().tm_yday
    year = date.year

    # Calculate subsolar point position
    phi_sbs, theta_sbs = subsol(year, doy, ut)
    r_sbs = np.array([np.sin(theta_sbs) * np.cos(phi_sbs), 
                      np.sin(theta_sbs) * np.sin(phi_sbs), 
                      np.cos(theta_sbs)])

    # Transform subsolar point to magnetic coordinates
    r_sbs_mag = GEOGRAPHIC_TRANSFORM.dot(r_sbs)
    theta_sbs_m = np.arccos(r_sbs_mag[2])
    phi_sbs_m = np.arctan2(r_sbs_mag[1], r_sbs_mag[0])
    if phi_sbs_m < 0.:
        phi_sbs_m = phi_sbs_m + 2. * np.pi

    # Transform input point to magnetic coordinates
    r = np.array([np.sin(theta) * np.cos(phi), 
                  np.sin(theta) * np.sin(phi), 
                  np.cos(theta)])
    r_mag = GEOGRAPHIC_TRANSFORM.dot(r)
    theta_m = np.arccos(r_mag[2])
    phi_m = np.arctan2(r_mag[1], r_mag[0])
    if phi_m < 0.:
        phi_m = phi_m + 2. * np.pi

    # Calculate magnetic local time
    mlt = phi_m - phi_sbs_m + np.pi
    if mlt < 0.:
        mlt = mlt + 2. * np.pi
    if mlt > 2. * np.pi:
        mlt = mlt - 2. * np.pi

    return theta_m, mlt


def inclination(lat, lon, alt=300., year=2005.):
    """
    :return
         I is inclination (+ve down)
    """
    if lon < 0:
        lon = lon + 360.      
    FACT = 180./np.pi
    REm = 6371.2  # Earth's mean radius in km
    
    # Calculate magnetic field components using IGRF
    x, y, z, f = calculate.igrf12syn(year, 2, REm + alt, lat, lon)  # 2 for geocentric coordinates
    
    # Calculate horizontal component and inclination
    h = np.sqrt(x * x + y * y)
    i = FACT * np.arctan2(z, h)
    return i

@np.vectorize
def geo2modip(theta: float, phi: float, date) -> tuple[float, float]:
    
    year = date.year
    
    # Calculate magnetic inclination and modified dip latitude
    I = inclination(lat=np.rad2deg(np.pi/2 - theta), 
                   lon=np.rad2deg(phi), 
                   alt=300., 
                   year=year)  # alt=300 for modip300
    theta_m = np.pi/2 - np.arctan2(np.deg2rad(I), np.sqrt(np.cos(np.pi/2 - theta)))
    
    # Calculate magnetic local time
    ut = sec_of_day(date)
    phi_sbs = np.deg2rad(180. - ut*15./3600)  # Subsolar longitude
    
    # Normalize angles to [0, 2π]
    if phi_sbs < 0.:
        phi_sbs = phi_sbs + 2. * np.pi
    if phi < 0.:
        phi = phi + 2. * np.pi
        
    # Calculate MLT
    mlt = phi - phi_sbs + np.pi
    if mlt < 0.:
        mlt = mlt + 2. * np.pi
    if mlt > 2. * np.pi:
        mlt = mlt - 2. * np.pi
        
    return theta_m, mlt
