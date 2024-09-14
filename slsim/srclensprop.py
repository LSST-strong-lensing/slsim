#!/usr/bin/env python
from math import *
from pylab import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np


"""
This script provides a random source position which lies within the caustic
of a singular isothermal ellipsoid. 
The definition conventions are based upon Keeton Mao and Witt 2000. 
We set the external shear to be equal to zero in their formulae.

It takes the Einstein radius, the projected axis ratio, and the deflector 
position angle as the inputs and returns the x and y positions of the sources 
where the major axis is aligned with the x axis

  b_I: This is the einstein radius of the SIE
    b_I = b_SIS eps_3/sin^{-1}(eps_3), where eps_3 is the eccentricity of the mass distribution

  q: Projected axis ratio
    q=sqrt(q3^2sin^2i+cos^2i) where q3 is the 3d acis ratio of the ellipsoid and i is inclination angle
    eps_3=(1-q3^2)^{1/2}


Note that we also provide a wrapper (srcpos_bsisq) which uses a value for b_SIS and q to define the 
projected surface mass density. Comparing Keeton, Mao and Witt 2000 equations with Kormann et al. 1994 yields:
  b_I = sqrt(q)*b_{SIS}

Although it seems that magically the inclination angle dependence vanishes out from kappa. This equation 
is on average ok given the oblateness and prolateness of the halos (Oguri, private communication, 
see also paper by Chae 2003).


References:
.. [1] Keeton Mao and Witt (2004), astro-ph/0002401, doi: 10.1086/309087
.. [2] Chae (2003), astro-ph/0211244, doi: 10.1111/j.1365-2966.2003.07092.x
.. [3] Kormann et al. (1994), A&A, 284, 285. 
"""


# define a fiducial value of Einstein radius used in scaling cross-sectional calculations, do not change this value
fid_b_I = 10.0 

#set initial cross-section flag = 0
init_cs_spl = 0

#set initial cross-section cubic spline interpolation flag = 0
cross_sect_spl = 0

def getcaustics(b_I,q):
  """
  Calculate the caustics for a Singular Isothermal Ellipsoid (SIE) lens model.

  Parameters:
  - b_I: Einstein radius of the SIE
  - q: Projected axis ratio

  Returns:
  - c1: Spline function for the tangential caustic
  - c2: Spline function for the radial caustic
  - maxr: Maximum radial distance between the caustics and the origin
  """


  ## Do not change the next line or face consequences
  t = np.arange(0.0, pi / 2.0 + pi / 100.0, pi / 200.0)

  den = (1 + q ** 2) - (1 - q**2) * np.cos(2 * t)

  # Calculate the radial distance for the caustics  
  r = sqrt(2.0) * b_I / np.sqrt(den)

  # Compute the coordinates for the caustics
  x = r * np.cos(t)
  y = r * np.sin(t)

  ## Generate the parametric function for the tangential caustic
  xi = np.sqrt(2.0 * (1 - q ** 2) / den)
  u = np.cos(t) * r
  v = np.sin(t) * r
  if(q != 1.0):
      u = u - b_I * np.arctan( xi * np.cos(t) ) / sqrt(1 - q ** 2)
      v = v - b_I * np.arctanh( xi * np.sin(t) ) / sqrt(1 - q ** 2)

  ## Generate the parametric function for the radial caustic
  up = b_I * 0.0
  vp = b_I * 0.0
  if(q != 1.0):
      up =- b_I * np.arctan(xi * np.cos(t)) / sqrt(1 - q ** 2)
      vp =- b_I * np.arctanh(xi * np.sin(t)) / sqrt(1 - q ** 2)

  # Set up splines for the tangential and radial caustic
  r1 = np.sqrt( u ** 2 + v ** 2)
  r2 = np.sqrt( up ** 2 + vp ** 2)
  
  # Calculate the angles for the tangential and radial caustics
  t1 = [0.] * u.size; 
  t2 = [0.] * up.size; 
  for ii in range(u.size):
      if(u[ii] == 0.):
          t1[ii] = np.pi/2.
      else:
          t1[ii] = np.arctan(np.abs(v[ii] / u[ii]))
      if(t[ii] > pi / 2.0):
          t1[ii] = pi - t1[ii]
  
  for ii in range(up.size):
      if(up[ii] == 0.):
          t2[ii] = np.pi / 2.0
      else:
          t2[ii] = np.arctan(np.abs(vp[ii] / up[ii]))
      if(t[ii] > pi / 2.0):
          t2[ii] = pi - t2[ii]

  # Create cubic splines for the tangential and radial caustics
  c1=interp1d(t1,r1,kind='cubic')
  c2=interp1d(t2,r2,kind='cubic')

  # determine the maximum radial distance between the caustics and the origin
  maxr=np.max([r1,r2])

  return c1,c2,maxr

## Given the SIE caustics, get the source position
def srcpos_sie(b_I,q,pa):
  """
  Get the source position within the SIE caustics.

  Parameters:
  - b_I: Einstein radius of the SIE
  - q: Projected axis ratio
  - pa: Position angle in degrees

  Returns:
  - xnew: x-coordinate of the source position
  - ynew: y-coordinate of the source position
  """
  ## Generate parametric theta values and the caustics at that theta

  # Get caustics and maximum radial distance
  c1,c2,maxr=getcaustics(b_I,q)

  ## Generate random points within a circle with radius maxr until you find one
  ## inside the caustic
  while(1):
    rgen = sqrt(np.random.rand(1)) * maxr
    tgen = np.random.rand(1) * (pi / 2.0)

    if(rgen < c1(tgen) or rgen < c2(tgen) ):
      break

  ## Generate the positions in the first quadrant
  xret = rgen * cos(tgen)
  yret = rgen * sin(tgen)

  ## randomly assign a quadrant
  quadr = np.random.rand(1)
  if(quadr < 0.5):
    xret =- xret
  quadr = np.random.rand(1)
  if(quadr < 0.5):
    yret =- yret

  # Rotate the position by the given position angle
  rad = pa * pi / 180.0
  xnew = xret * cos(rad) - yret * sin(rad)
  ynew = xret * sin(rad) + yret * cos(rad)

  return xnew, ynew

def srcpos_bsisq(bsis,q,pa):
  """
  Get the source position within the SIS caustics.

  Parameters:
  - bsis: Einstein radius for a Singular Isothermal Sphere (SIS)
  - q: Projected axis ratio
  - pa: Position angle in degrees

  Returns:
  - xnew: x-coordinate of the source position
  - ynew: y-coordinate of the source position
  """

  # Get the source position using the SIE model
  xret,yret=srcpos_sie(bsis*sqrt(q),q,pa)

  return xret,yret

## Gives cross-section in units of bsis
def getcrosssect(bsis,q):
  """
  Calculates the angular cross-section in the units of the Singular Isothermal Sphere (SIS) Einstein radius.

  Parameters:
  - bsis: Einstein radius of the Singular Isothermal Sphere (SIS)
  - q: Projected axis ratio of the lens

  Returns:
  - result: Angular cross-sectional area in units of the SIS Einstein radius
  """

  # Convert SIS Einstein radius to SIE Einstein radius
  b_I = bsis * sqrt(q)

  ## Generate parametric theta values and the critical curve
  t = np.arange(0.0, pi / 2.0 + pi / 200.0, pi / 200.0)
  c1, c2, maxr = getcaustics(b_I, q)

  # Evaluate the radial distances
  r1 = c1(t)
  r2 = c2(t)

  # Ensure that r1 contains the maximum of r1 and r2
  for i in range(r1.size):
    r1[i] = max(r1[i],r2[i])

  # Create a new interpolated function for the cross-sectional area
  # We want to integrate \int_0^{2pi} 1/2 r^2(t) dt= 4 \int_0^{pi/2} 1/2 r^2(t) dt
  cnew = interp1d(t, 4 * 0.5 * r1 * r1)
  
  # Integrate under the curve to find the total cross-sectional area
  result, err = quad(lambda x: cnew(x), 0.0, pi/2.0, epsrel=1.0E-3)

  #This is the angular area is in units^2 where units is the unit of b. 
  return result

## Einstein radius in radians
def getreinst(zlens,zsrc,sigma,cosmo,constants):
    """
    Computes the Einstein radius of a lens in radians.

    Parameters:
    - zlens: Redshift of the lens
    - zsrc: Redshift of the background source
    - sigma: Velocity dispersion of the lens (in km/s)
    - cosmo: Cosmology used
    - constants: Dictionary containing physical constants used in calculations

    Returns:
    - reinst: Einstein radius in radians
    """

    if(zsrc<zlens):
       return 0

    else:
      # Get the angular diameter distances
      Ds= (cosmo.angular_diameter_distance(zsrc)*cosmo.H(0)/100).value
      Dds=(cosmo.angular_diameter_distance_z1z2(zlens,zsrc)*cosmo.H(0)/100).value

      # Compute the Einstein radius in radians
      reinst=4*pi*(sigma/constants.get('light_speed'))**2.*(Dds/Ds)
      
      return reinst

## First note that the cross-section in sie just depends upon q where as the
## dependence on b_I can be easily predicted. As source redshift changes, only
## b_I changes. This implies we can initialize a cross-section spline which is a
## function of only q for a fiducial b_I.

def init_crosssect():
  """
  Initializes the cross-section spline interpolation from the Crosssect.dat file.
  """

  global cross_sect_spl, init_cs_spl

  # Print a message indicating the initialization process
  print("Initializing Cross-section")

  # Load data from the file 'Crosssect.dat'
  qi,csecti=np.loadtxt("Crosssect.dat",unpack=1)

  # Create a cubic spline interpolation function for the cross-section
  cross_sect_spl=interp1d(qi,csecti,kind='cubic')

  # Set the initialization flag to indicate that the spline is ready
  init_cs_spl=1


def getcrosssect_num(bsis, q):
  """
  Get the cross-section in steradian using Einstein radius of SIS and projected axis ratio

    Parameters:
  - bsis (float): The Einstein radius of the SIS (in radians).
  - q (float): The projected axis ratio of the lens.

    Returns:
  - float: The cross-section in steradian
  """
  global cross_sect_spl,init_cs_spl
  
  if(q > 0.999):
      q = 0.999
  if(q < 0.1):
      q = 0.1
  if(init_cs_spl == 0):
    init_crosssect()

  b_I = bsis * sqrt(q)

  # Compute the cross-section using the spline interpolation and return the result
  return cross_sect_spl(q) * (b_I/ fid_b_I) **2

