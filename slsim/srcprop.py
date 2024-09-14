#!/usr/bin/env python
#from math import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import srclensprop as slp
import scipy.integrate as sci


"""
This script uses the source redshift distributions and luminosity functions 
to calculate the source properties. It calculates the source number density 
and lens cross-section to derive the number of sources per lens along with 
the source redshift and i-band magnitude.

References:
.. [1] Fan et al., (2001), astro-ph/0008123, doi: 10.1086/318033
.. [2] Richards et al., (2005), astro-ph/0504300, doi: 10.1111/j.1365-2966.2005.09096.x
.. [3] Richards et al., (2006), astro-ph/0601434, doi: 10.1086/503559
.. [4] Oguri and Marshal (2010), astro-ph/1001.2037, doi: 10.1111/j.1365-2966.2010.16639.x
.. [5] More et al., (2016), astro-ph/1504.05587, doi: 10.1093/mnras/stv1965
.. [6] Faure et a;., (2009), astro-ph/0810.4838, doi: 10.1088/0004-637X/695/2/1233

"""

fp1=0
fp2=0
def setlogfile(lgfile):#,outfile):
    global fp1,fp2
    fp1=open(lgfile,"w")
    #fp2=open(outfile,"w")
    

init_Phi_qso=0
Phi_qso_spl=0

init_P_gal=0
P_gal_spl=0

init_nsmlim_gal=0
nsmlim_gal_spl=0


#######################
## PART 1- For quasars
#######################

def dPhibydM_qso(redshift, mag, cosmo):
    """
    Compute the quasar luminosity function, dPhi/dM, for a given redshift and apparent magnitude.

    Here, we adopt the standard double power law for quasar luminosity function with
    the parametric form given in Oguri and Marshal 2010, eq 10.
    
    Parameters:
    - redshift: The redshift of the quasar.
      type: float
    - mag: The i-band apparent magnitude of the quasar.
      type: float
    - cosmo: the cosmology used.
      type: astropy.cosmology

    Returns:
    - The luminosity function dPhi/dM in units of h70^3 Mpc^{-3}/mag.
    """    

    # define the continuum slope for quasar spectrum  
    alp = -0.5 
    
    #find the K-correction factor to z=0, as shown in Richards et al. (2006)
    kcorr = -2.5 * (1 + alp) * np.log10(1+redshift)
    # NOTE:K_{em} check
    
    # get the luminosity distance in Mpc from the redshift
    Dlum = cosmo.luminosity_distance(redshift).value
    DM = 5 * np.log10(Dlum)

    # compute the absolute magnitude of the quasar
    # The term '25.0' accounts for the distance modulus adjustment in Mpc
    Mabs = mag - kcorr - DM - 25.0

    
    ## define the bright end slope alpha = -3.31 for z < 3 from Richards et al. (2005)
    ## However, we use the modified value of alpha = -2.58 for z > 3 from Fan et al 2001
    if(redshift < 3):
        alpha = -3.31
    else:
        alpha = -2.58
    
    # Define the faint end slope beta from Richards et al. (2005)
    beta = -1.45

    # Model the redshift evolution of the luminosity function as pure luminosity evolution
    # Parameters zeta, xi, and zstar are fitted values from Oguri and Marshall (2010)
    zeta = 2.98
    xi = 4.05
    zstar = 1.60            # characterisitic redshift for evolution of quasar luminosity function

    # compute the normalization factor for the luminosity function from Richards et al. (2005)
    phistar = 5.34E-6 * pow(cosmo.H0.value/100, 3)

    # hence, the redshift dependent factor f(z), eq 12, Oguri and Marshall 2010.
    fofz = (np.exp(zeta * redshift) * (1 + np.exp(xi * zstar)))/ pow(np.sqrt(np.exp(xi * redshift)) + np.sqrt(np.exp(xi * zstar)), 2)

    # compute the the characteristic absolute magnitude of quasars using eq 11, Oguri and Marshall 2010.
    Mstar = -20.90 + 5 * np.log10(cosmo.H0.value/100) - 2.5 * np.log10(fofz)

    # Compute and return the luminosity function dPhi/dM
    return phistar/(pow(10,(0.4*(alpha+1)*(Mabs-Mstar)))+pow(10,(0.4*(beta+1)*(Mabs-Mstar))))


def initPhiqso(kwargs_source, cosmo):
  """
  Initializes the quasar luminosity function, Phi(z), as a function of redshift.
  The resulting Phi(z) spline can be used to interpolate the number density of 
  quasars for a given redshift within the specified range.
  Units of Phi(z) will be h70^3 Mpc^{-3}

  Parameters:
  - kwargs_source: A dictionary containing the source parameters

  - cosmo: cosmology defined
    type: astropy.cosmology
  
  """

  global Phi_qso_spl, init_Phi_qso

  # Log the initialization process
  fp1.write("Initializing Phi(z)\n")

  #define the redshift range
  zi = np.arange(kwargs_source.get('min_z'), kwargs_source.get('max_z')+0.01, 0.01)
  
  #Initialize an array
  Phii = zi*0.0

  for ii in range(zi.size):
    # Integrate the quasar luminosity function over the magnitude range
    Phii[ii] = sci.quad(
       lambda mag: dPhibydM_qso(zi[ii], mag, cosmo), 
       kwargs_source.get('min_mag'), 
       kwargs_source.get('max_mag'))[0]

  # Create a cubic spline interpolation of Phi(z) based on the computed values  
  Phi_qso_spl = interp1d(zi, Phii, kind='cubic')
  
  # Set the initialization flag = 1
  init_Phi_qso = 1


def dNbydz_qso(zz,vdisp,zred,q,kwargs_source,cosmo,constants):
  """
  Computes the number density of quasars per unit redshift.

  Parameters:
    - zz: Redshift at which to evaluate the function.
      type: float
    - vdisp: Velocity dispersion of the lensing galaxy (in km/s).
      type: float
    - zred: Redshift of the lensing galaxy.
      type: float
    - q: Projected axis ratio of the lens.
      type: float
    - cosmo: Cosmology used.
      type: astropy.cosmology
    - constants: Dictionary of physical constants used in the calculations.

  Returns:
    - The number density of quasars per unit redshift, considering the lensing cross-section.
  """

  ## \int dz dV/dz \int dM dPhi/dM(z) \sigma(zl,zs,q)
  global Phi_qso_spl,init_Phi_qso

  # Initialize the quasar luminosity function, Phi(z), if not already done
  if(init_Phi_qso==0):
    initPhiqso(kwargs_source, cosmo)
  
  # Get the quasar luminosity function at the given redshift zz
  Phi=Phi_qso_spl(zz)

  # Calculate the Einstein radius of the lensing galaxy in radians
  bsis=slp.getreinst(zred,zz,vdisp,cosmo,constants)

   # Calculate the lensing cross-section in steradians
  csect=slp.getcrosssect_num(bsis,q)

  ## Multiply by volume factor and return the number density
  return Phi*1./(cosmo.H(zz)/cosmo.H(0)).value*(constants.get('light_speed')/(cosmo.H(0).value))*(cosmo.angular_diameter_distance(zz).value)**2.*csect*(1+zz)**2


def findzqso(ztry,vdisp,zred,q,Ntarget,kwargs_source,cosmo,constants):
  """
  Find the redshift at which the integrated quasar number density equals Ntarget.

  Parameters:
  - ztry: The trial redshift at which to evaluate the condition. #NOTE: confirm
    type: float
  - vdisp: The velocity dispersion of the lensing galaxy (in km/s).
      type: float
  - zred: The redshift of the foreground deflector.
      type: float
  - q: the axis ratio of the foreground deflector.
      type: float
  - Ntarget: The target number density of galaxies.
      type: float
  - kwargs_source: A dictionary for the source parameters.
  - cosmo: cosmology used.
    type: astropy.cosmology
  - constants: A dictionary of physical constants needed for calculations.

  Returns:
  - The difference between the target number density and the integrated number density up to ztry. #NOTE: confirm
  """
  return Ntarget-sci.quad(lambda zz: dNbydz_qso(zz,vdisp,zred,q,kwargs_source,cosmo,constants),kwargs_source.get('min_z'),ztry)[0]


def findmagqso(magtry, zsrc, Phitarget, kwargs_source, cosmo):
  """
  NOTE: confirm these comments
  Compute the difference between the target quasar number density and the modeled number density 
  for a given trial magnitude.

  Parameters:
  - magtry: The trial magnitude of the quasar.
    type: float
  - zsrc: The redshift of the quasar.
    type: float
  - Phitarget: The target number density of quasars.
    type: float
  
  Returns:
  - The difference between the target number density and the modeled number density.
  """
  result=Phitarget-sci.quad(lambda mag: dPhibydM_qso(zsrc, mag, cosmo),kwargs_source.get('min_mag'),magtry)[0]

  return result


def Nsrc_qso(magg,magr,magi,zred,q,vdisp,myseed,kwargs_source,cosmo,constants):
  """
  Calculate the number of background quasars that could be lensed by a galaxy and determine their properties.

  Parameters:
    - magg: g-mag of the foreground
      type: float
    - magr: r-mag of the foreground
      type: float
    - magi: i-mag of the foreground
      type: float
    - zred: Redshift of the foreground galaxy
      type: float
    - q: Projected axis ratio of the foreground
      type: float
    - myseed: Seed for the random number generator
      type: int
    - kwargs_source: Dictionary containing source parameters (e.g., min_z, max_z, min_mag, max_mag).
    - cosmo: Cosmology model used for calculations
      type: from astropy.cosmology.
    - constants: Dictionary of physical constants used in the calculations.

    Returns:
    - Nreal: The number of quasars that could be lensed by this galaxy.
    - listmag: List of magnitudes of these quasars.
    - listz: List of redshifts of these quasars.
    - rands: List of random numbers used in the process.
    - vdisp: Velocity dispersion of the lensing galaxy.
  
  """
  global boost_csect_qso

  #bsist=slp.getreinst(zred,zmax_qso,vdisp)*206264.8;
  ## to allow generating small image sep. lenses

  # Calculate the mean number of quasars behind this object within its lensing cross-section 
  # This is given by the following integral
  # \int dz dV/dz \int dM dPhi/dM(z) \sigma(zl,zs,q)
  # For quasars, let us integrate from zmin=1.0 to 5.0
  Nsrcmean = sci.quad(lambda zz: dNbydz_qso(zz, vdisp, zred, q, kwargs_source,
                        cosmo, constants),kwargs_source.get('min_z'), 
                        kwargs_source.get('max_z')
                      )[0]
  Nsrcmean_boost = Nsrcmean * kwargs_source.get('boost_csect')

  ## Return a Poisson deviate
  np.random.seed(myseed)
  Nreal = np.random.poisson(Nsrcmean_boost)

  fp1.write("This lens has %f quasars behind it on average and Poisson deviate is %d\n"%(Nsrcmean_boost,Nreal))

  # Initialize lists to store magnitudes, redshifts, and random numbers
  listmag=[]
  listz=[]
  rands=[]

  for ii in range(Nreal):
    qsozsrc=0.0

    # Generate redshift for each quasar until it's greater than the redshift of the lens
    while(qsozsrc < zred):
        np.random.seed(myseed)
        rr = np.random.random()
        Ntarg = rr * Nsrcmean

        # Use root finding to determine the redshift at which the cumulative number of quasars equals Ntarg
        qsozsrc=brentq(findzqso, kwargs_source.get('min_z'), kwargs_source.get('max_z'),
                       args=(vdisp, zred, q, Ntarg, kwargs_source, cosmo, constants), 
                       xtol=1.e-3)

    rands = np.append(rands, rr)
    listz = np.append(listz, qsozsrc)
    done=False

    # Generate magnitude for each quasar based on the calculated redshift
    while(not done):
        try:
            np.random.seed(myseed)
            rr = np.random.random()
            Phitarg = rr * Phi_qso_spl(qsozsrc)
            rands = np.append(rands,rr)
            qsomag = brentq(findmagqso, kwargs_source.get('min_mag'), kwargs_source.get('max_mag'), 
                            args=(qsozsrc, Phitarg, kwargs_source, cosmo), 
                            xtol=1.e-3)
            done = True
        except ValueError:
            print("checking", findmagqso(kwargs_source.get('min_mag'), qsozsrc, Phitarg), 
                  findmagqso(kwargs_source.get('max_mag'), qsozsrc,Phitarg), kwargs_source.get('min_mag'), 
                  kwargs_source.get('max_mag'), Phitarg)
            print("Done is:",done)

    listmag = np.append(listmag, qsomag)

  return Nreal, listmag, listz, rands#, vdisp


## Calculate the total number of qsos in a survey
def Nqso(kwargs_source,constants,cosmo):
  """
  Calculate the total number of quasars (QSOs) in a survey based on the quasar luminosity function.

  Parameters:
    - kwargs_source: Dictionary containing source parameters:
    - constants: Dictionary of physical constants used in the calculations.
    - cosmo: Cosmology used.
      type: astropy.cosmology

  Returns:
    - totNqso: The total number of quasars in the survey, integrated over redshift.
  """


  global Phi_qso_spl,init_Phi_qso

  # Initialize the quasar luminosity function, Phi(z), if not already done
  if(init_Phi_qso==0):
    initPhiqso(kwargs_source, cosmo)
  
  def dNqso(zz,kwargs_source,constants,cosmo=cosmo):
    """
    Calculate the number of quasars per unit redshift.

    Parameters:
    - zz: Redshift at which to evaluate the function.
    - kwargs_source: Dictionary containing source parameters.
    - constants: Dictionary of physical constants used in the calculations.
    - cosmo: Cosmology used
      type: astropy cosmology

    Returns:
    - The number of quasars per unit redshift.
    """

    # Get the quasar luminosity function at the given redshift zz
    Phi=Phi_qso_spl(zz)

    #return the number of quasars per unit redshift
    return Phi*1./(cosmo.H(zz)/cosmo.H(0)).value*(constants.get('light_speed')/(cosmo.H(0).value))*(cosmo.angular_diameter_distance(zz).value)**2.*(1+zz)**2

  # integrate over the redshift range
  # Units are in steradian^(-1) 
  totNqso=sci.quad(lambda zz:dNqso(zz),kwargs_source.get('min_z'),kwargs_source.get('max_z'))[0]

  return totNqso





#******************************************************************************************************************#
#******************************************************************************************************************#
########################
## PART 2- For galaxies
########################
#******************************************************************************************************************#
#******************************************************************************************************************#

def initPhigal(mlim, kwargs_source):
  """
  Initialize the galaxy luminosity function per unit comoving volume.

  Parameters:
  - mlim: Magnitude limit used to compute the characteristic redshift.
  - kwargs_source: Dictionary containing the source parameters.

  This function computes the galaxy number density function Phi(z) and gives an interpolated
  function.
  This gives a dimensionless P(zs,mlim)
  """
  global P_gal_spl,init_P_gal
  
  # log the inititalization step
  fp1.write("Initializing Phi(z)\n")

  # define the redshift range and parameters
  zi = np.arange(kwargs_source.get('min_z'), kwargs_source.get('max_z')+0.01, 0.01)
  
  #find the characteristic redshift for magnitude limit from eq 2, Faure et al 2009
  z0 = 0.13 * mlim - 2.2 

  beta = 1.5

  ## Compute the galaxy luminosity function per unit comoving volume (Faure et al 2009) 
  Phii = beta * (zi / z0) ** 2 * np.exp(-(zi / z0) ** beta) / z0

  ## interpolate Phi(z) wihtin the redshift limits
  P_gal_spl=interp1d(zi,Phii,kind='cubic')

  # Set the initialization flag
  init_P_gal=1


def dnsbydm(mag):
  """
  calculates the galaxy number density per unit magnitude
  
  Parameters:
  - mag: Magnitude value for which the number density is computed.
    type: float

  Return:
  - Number density of galaxies per unit magnitude in radians squared.
  """

  # define the parameters (n0, m1, a, b) in the source redshift distribution from eq 3, Faure et al. 2009
  ## convert from degree squared to degree radian
  n0 = 3e3*(180./np.pi)**2.0
  
  m1 = 20.0
  a = 0.30
  b = 0.56

  # return the number density per unit magnitude
  return n0 / np.sqrt(10.0**(2 * a * (m1-mag)) +10.0**(2 * b * (m1-mag)) )



def initnsmlim(kwargs_source):
  """
  Initialize the number density of galaxies as a function of magnitude limit.

  Parameters:
  - kwargs_source: Dictionary containing the source parameters.

  This function calculates the number density of galaxies up to a given magnitude
  limit and returns an interpolated function.
  """

  global nsmlim_gal_spl,init_nsmlim_gal

  #log the initialization step
  fp1.write("Initializing ns(mlim)\n")

  # Define the magnitude range
  magi=np.arange(kwargs_source.get('min_mag'), kwargs_source.get('max_mag') + 0.01, 0.01)
  
  #Compute number densities
  nsi = np.array([sci.quad(lambda xmag: dnsbydm(xmag), kwargs_source.get('min_mag'), mag)[0] for mag in magi])
  
  # Interpolate the number density function
  nsmlim_gal_spl = interp1d(magi, nsi, kind='cubic')

  #Set the initialization flag
  init_nsmlim_gal=1



def dNbydz_gal(zz,vdisp,zred,q,kwargs_source,cosmo,constants):
  """
  Calculates the number density of galaxies per redshift interval

    Parameters:
  - zz: Redshift value at which to evaluate the galaxy number density.
      type: float
  - vdisp: Velocity dispersion of the the lensing galaxy (in km/s).
      type: float
  - zred: redshift of the foreground deflector
      type: float
  - q:    axis ratio of the foreground deflector
      type: float
  - kwargs_source: Dictionary containing source parameters.
  - cosmo: Cosmology used.
      type: astropy.cosmology
  - constants: Physical constants needed for calculations.

  """
  ## \int dz dV/dz p(zs,mlim) \sigma(zl,zs,q)
  global P_gal_spl, init_P_gal

  # Initialize the galaxy luminosity function if not already done
  if(init_P_gal==0):
    initPhigal(kwargs_source.get('max_mag'), kwargs_source)

  ## Get the galaxy luminosity function (Phi) for the given redshift
  Phi=P_gal_spl(zz)

  ## Get cross-section in steradian
  bsis=slp.getreinst(zred,zz,vdisp,cosmo,constants)
  csect=slp.getcrosssect_num(bsis,q)

  # Return the galaxy luminosity function*cross-section
  return Phi*csect


def findzgal(ztry, vdisp, zred, q, Ntarget, kwargs_source, cosmo, constants):
  """
  Find the redshift at which the integrated galaxy number density equals Ntarget.

  Parameters:
  - ztry: The trial redshift at which to evaluate the condition. #NOTE: confirm
      type: float
  - vdisp: The velocity dispersion of the galaxies.
      type: float
  - zred: The redshift of the foreground deflector.
      type: float
  - q: the axis ratio of the foreground deflector.
      type: float
  - Ntarget: The target number density of galaxies.
      type: int
  - kwargs_source: A dictionary for the source parameters.
  - cosmo: cosmology used.
      type: astropy.cosmology
  - constants: A dictionary of physical constants needed for calculations.

  Returns:
  - The difference between the target number density and the integrated number density up to ztry. #NOTE: confirm
  """

  return Ntarget-sci.quad(lambda zz: dNbydz_gal(zz, vdisp, zred, q, 
                                                kwargs_source, cosmo,constants),
                                                kwargs_source.get('min_z'),ztry)[0]

def findmaggal(magtry, zsrc, Phitarget):
  """
  NOTE: confirm these comments
  Compute the difference between the target galaxy number density and the modeled number density 
  for a given trial magnitude.

  Parameters:
  - magtry: The trial magnitude of the galaxy.
      type: float
  - zsrc: The redshift of the galaxy.
      type: float
  - Phitarget: The target number density of galaxies.

  Returns:
  - The difference between the target number density and the modeled number density.
  """

  # Compute the characteristic redshift z0 based on magtry
  z0 = 0.13 * magtry - 2.2

  beta = 1.5  #taken from Faure et al 2009

  # compute the source number density Pmagzs based on the trial magnitude and redshift
  Pmagzs = beta * (zsrc/z0) ** 2.0 * np.exp(-(zsrc/z0)**beta)/z0

  # Compute the number density using nsmlim_gal_spl and the trial magnitude
  # and calculate the difference from the target number density
  return Phitarget-nsmlim_gal_spl(magtry)*Pmagzs


def Nsrc_gal(magg,magr,magi,zred,q,vdisp,myseed,kwargs_source,cosmo,constants):
  '''
  Calculate the number of background galaxies that could be lensed by a foreground galaxy and determine their properties.

  Parameters:
  - magg: The galaxy's g-band magnitude.
      type: float
  - magr: The galaxy's r-band magnitude.
      type: float
    - magi: i-mag of the foreground
      type: float
  - zred: The redshift of the foreground galaxy.
      type: float
  - q: the axis ratio of the foreground.
      type: float
  - myseed: seed for random number generation.
      type: int
  - kwargs_source: Dictionary containing source parameters.
  - cosmo: cosmology used.
      type: astropy.cosmology
  - constants: A dictionary of physical constants needed for calculations.

  Returns:
  - Nreal: The number of background galaxies.
  - listmag: List of magnitudes for the background galaxies.
  - listz: List of redshifts for the background galaxies.
  - rands: List of random numbers used in the process.
  - vdisp: The velocity dispersion of the foreground galaxy.
  '''

  global init_nsmlim_gal

  listmag = []
  listz = []
  rands = []

  # Ensure the redshift of the foreground galaxy is above the threshold
  if(zred > kwargs_source.get('max_z')):
      return 0, listmag, listz, rands

  ## Initialize ns(mlim) if not done before
  if(init_nsmlim_gal == 0):
      initnsmlim(kwargs_source)


  Nsrcmean = sci.quad(lambda zz: dNbydz_gal(zz, vdisp, zred, q, kwargs_source,
                                             cosmo, constants), kwargs_source.get('min_z'), 
                                             kwargs_source.get('max_z'))[0]
  
  ## boost the number artificially by a given boosting factor
  Nsrcmean_boost = Nsrcmean * nsmlim_gal_spl(kwargs_source.get('max_mag')) * kwargs_source.get('boost_csect')

  ## Return a Poisson deviate
  np.random.seed(myseed)
  Nreal = np.random.poisson(Nsrcmean_boost)
  #print("This lens has %f galaxies behind it on average and Poisson deviate is %d\n"%(Nsrcmean_boost,Nreal))
  fp1.write("This lens has %f galaxies behind it on average and Poisson deviate is %d\n"%(Nsrcmean_boost, Nreal))

  # Generate properties for each galaxy
  for ii in range(Nreal):
    galzsrc = 0.0
    trials = 0
    while(galzsrc < zred):
        np.random.seed(myseed)
        rr = np.random.random()
        Ntarg = rr * Nsrcmean

        # Find the redshift of the background galaxy
        galzsrc = brentq(findzgal, kwargs_source.get('min_z'), kwargs_source.get('max_z'), 
                         args = (vdisp, zred, q, Ntarg, kwargs_source, cosmo, constants), 
                         xtol = 1.0e-3)

    rands = np.append(rands, rr)
    listz = np.append(listz, galzsrc)
    done = False

    while (not done):
        try:
            np.random.seed(myseed)
            rr = np.random.random()
            Phitarg = rr * P_gal_spl(galzsrc) * nsmlim_gal_spl(kwargs_source.get('max_mag'))
            rands = np.append(rands, rr)
            # Find the magnitude of the background galaxy
            galmag = brentq(findmaggal,kwargs_source.get('min_mag'), kwargs_source.get('max_mag'), 
                            args = (galzsrc, Phitarg), xtol=1.0e-3)
            done = True

        except ValueError:
            print("checking", findmaggal(kwargs_source.get('min_mag'), galzsrc, Phitarg), 
                  findmaggal(kwargs_source.get('max_mag'), galzsrc, Phitarg), 
                  kwargs_source.get('min_mag'), kwargs_source.get('max_mag'), Phitarg)
            print("Done is:", done)
            
    listmag = np.append(listmag, galmag)
      

  return Nreal,listmag,listz,rands#,vdisp
