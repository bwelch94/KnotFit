#check what you actually use, delete unused imports
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling.functional_models import Sersic2D, Gaussian2D
from reproject import reproject_interp
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import convolve, Gaussian2DKernel
import emcee
import scipy.special as sp
from scipy.optimize import minimize


def main():
    print("Let's begin...")
    print()
    indir = '/home/brian/Documents/JHU/lensing/whl0137'
    imfile = 'hlsp_relics_hst_wfc3ir-60mas_whl0137-08_f110w_drz.fits'
    imfile = os.path.join(indir,imfile)
    modeldir = '/home/brian/Documents/JHU/lensing/whl0137/lens_modelling/ModelC_re/best'
    dflx = os.path.join(modeldir, 'dx_z6.fits')
    dfly = os.path.join(modeldir, 'dy_z6.fits')

    print('Reading and reprojecting')
    ax, ay, imdataHST = initDeflection_Image(imfile, dflx, dfly, 
                                             zlens=0.566, zsource_in=6.0, zsource_out=6.2)
    
    print('Making Magnification Map')
    magnifinv = makeMagnificationMap(ax, ay)
    
    # Define coordinate grid
    yy, xx = np.indices(magnifinv.shape)
    # Source position grid
    xss = xx - ax
    yss = yy - ay
    
    xlo, xhi = 2700, 3100
    ylo, yhi = 1800, 2100
    
    imstamp = imdataHST[ylo:yhi, xlo:xhi]
    
    print('Generating Convolution Kernel')
    star = starGen(imdataHST, starLoc=(2232, 1622))
    
    knotpos = (2950, 1976) # lower knot
    #knotpos = (2981, 2006) # upper knot
    x, y = knotpos
    print()
    print('knot x:',x, 'knot y:', y)
    print()

    rmsfile = 'hlsp_relics_hst_wfc3ir-60mas_whl0137-08_f110w_rms.fits'
    rmsfile = os.path.join(indir, rmsfile)
    print('Generating ArgDict')
    argdict = initArgDict(rmsfile, imstamp, limits=(xlo,xhi,ylo,yhi), ax=ax, ay=ay,
                          knotpos=knotpos, sourcegrid=(xss,yss), star=star)
    
    theta = np.array([0.01, 1., 1.]) #, x, y]) # pixel radius
    #theta = np.array([0.1, 0.1]) #for Gaussian profile
    print('Initial Parameters:')
    print('Flux:', theta[0])
    print('Radius (parsecs):', theta[1])
    print('Sersic index:', theta[2])
    print()
    print('Beginning Maximum Likelihood Fit')
    print()
    sol = minimize(neg_lhood, theta, args=argdict, method='Nelder-Mead',
                    options={'maxiter':2500})
    print('Maximum Likelihood Estimate Complete!')
    print(sol)

    mcmcdict = '/home/brian/Documents/JHU/lensing/knotfit/chains'
    mcmc_outfile = 'lowknot_minimize_2k_z6twopt.h5'
    mcmc_outfile = os.path.join(mcmcdict,mcmc_outfile)
    print('MCMC Chain Output File: ', mcmc_outfile)
    print()
    print('Beginning MCMC')
    theta = sol.x
    nwalk = len(theta) * 2
    sampler = runMCMC(theta, argdict, nwalkers=nwalk, 
                      niter=2000, outfile=mcmc_outfile)
    
    return sampler


def initDeflection_Image(imagefile, deflectionFileX, deflectionFileY, 
                         zlens, zsource_in, zsource_out):
    
    ### Read in image file
    ### Read in deflection maps, 
    ### reproject to image file coordinates
    
    imhdu = fits.open(imagefile)
    imdata = imhdu[0].data
    
    hdudeflx = fits.open(deflectionFileX)
    hdudefly = fits.open(deflectionFileY)
    
    #reproject deflection fields to HST WCS pixels
    deflxHST, footprint = reproject_interp(hdudeflx[0], imhdu[0].header)
    deflyHST, footprint = reproject_interp(hdudefly[0], imhdu[0].header)
    ax = deflxHST / 0.06 # arcsec -> pixels
    ay = deflyHST / 0.06 # arcsec -> pixels
    
    # convert back to Dds / Ds = 1
    if zsource_in == 0:
        Dds_Ds_out = Dds_Ds(zlens, zsource_out)

        ax = ax * Dds_Ds_out
        ay = ay * Dds_Ds_out
    else:
        Dds_Ds_in  = Dds_Ds(zlens, zsource_in)
        Dds_Ds_out = Dds_Ds(zlens, zsource_out)

        ax = ax / Dds_Ds_in * Dds_Ds_out
        ay = ay / Dds_Ds_in * Dds_Ds_out
    
    return ax, ay, imdata


def Dds_Ds(zl, zs):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    Dds = cosmo.angular_diameter_distance_z1z2(zl, zs)
    Ds  = cosmo.angular_diameter_distance_z1z2(0 , zs)
    return Dds / Ds


def makeMagnificationMap(ax, ay, outfile=None):
    axx = ddx(ax)
    ayy = ddy(ay)
    
    axy = ddy(ax)
    ayx = ddx(ay)
    
    kappa  = 0.5 * (axx + ayy)
    gamma1 = 0.5 * (axx - ayy)
    gamma2 = axy
    gamma  = np.sqrt(gamma1**2 + gamma2**2)
    
    kappa = zeropad(kappa)
    gamma = zeropad(gamma)
    
    magnifinv = (1-kappa)**2 - gamma**2
    
    if outfile:
        magnif = 1. / magnifinv
        hdumag = fits.PrimaryHDU(magnif)
        hdumag.header = imhdu[0].header
        hdumag.writeto(outfile)
    
    return magnifinv


def ddy(A):
    """Numerical derivative: 2nd-order
    output array will have dimentions (N-2, N-2)"""
    dAdy1 = (A[2:] - A[:-2]) / 2.
    dAdy2 = (-A[4:] + 8*A[3:-1] - 8*A[1:-3] + A[:-4]) / 12.
    dAdy1[1:-1,:] = dAdy2
    dAdy1 = dAdy1[:,1:-1]
    return dAdy1

def ddx(A):
    dAdx = ddy(A.T).T
    return dAdx

# Results have dimensions (N-2, N-2)
# Add zero padding to restore to (N, N) HST image pixel grid
def zeropad(data):
    ny, nx = data.shape
    padded = np.zeros((ny+2, nx+2))
    padded[1:-1,1:-1] = data
    return padded

def starGen(imdata, starLoc, extent=15):
    xmin, xmax = (starLoc[0]-(extent/2), starLoc[0]+((extent/2)))
    xmin, xmax = int(xmin), int(xmax)
    ymin, ymax = (starLoc[1]-(extent/2), starLoc[1]+((extent/2)))
    ymin, ymax = int(ymin), int(ymax)
    star = imdata[ymin:ymax, xmin:xmax]
    return star


def initArgDict(rmsfile, imstamp, limits, ax, ay,
                knotpos, sourcegrid, star, delta=5):
    xlo, xhi, ylo, yhi = limits
    x, y = knotpos
    xs = x - ax[y, x]
    ys = y - ay[y, x]
    xss, yss = sourcegrid
    rmsfo = fits.open(rmsfile)
    rms = rmsfo[0].data
    rmscut = rms[ylo:yhi,xlo:xhi]
    
    datax, datay = x - xlo, y - ylo
    
    rms_cutout = rmscut[datay-delta:datay+delta, datax-delta:datax+delta]
    knotbounds = [datax-delta, datax+delta, datay-delta,datay+delta]
    knot = imstamp[datay-delta:datay+delta, datax-delta:datax+delta]

    #sigma = poisson(knot, t_expose=5123.501098) #whl0137
    sigma = poisson(knot, t_expose=4821.973725) #macs0308 
    sigma += rms_cutout

    # Include in args: xs, ys, xss, yss, star, data, sigma=RMS
    argdict = {
        "xs" : xs,
        "ys" : ys,
        "ax" : ax,
        "ay" : ay,
        "xss" : xss,
        "yss" : yss,
        "star" : star,
        "arcIm" : imstamp,
        "sigma" : sigma,
        "knotbounds" : knotbounds,
        "limits" : limits
    }
    return argdict


def runMCMC(theta_init, argdict, nwalkers=4, niter=500, outfile=None, **kwargs):
    ndim = len(theta_init)
    if min(theta_init) < 1e-5:
        frac = 0.1 * min(theta_init)
    else:
        frac = 1e-5
    pos = theta_init + frac * np.random.randn(nwalkers, ndim)
    
    if outfile:
        backend = emcee.backends.HDFBackend(outfile)
        backend.reset(nwalkers,ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, kwargs=argdict, backend=backend)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, kwargs=argdict)
    
    sampler.run_mcmc(pos, niter, progress=True)
    return sampler


def log_probability(theta, **kwargs):
    pri = log_prior(theta, **kwargs)
    if not np.isfinite(pri):
        return -np.inf
    return pri + log_likelihood(theta,**kwargs)
    

def log_prior(theta, **kwargs):
    amp1, reff1, amp2, reff2, n2, x2, y2 = theta #BDW
    #amp1, reff1, amp2, reff2, x2, y2 = theta
    xs = kwargs["xs"]
    ys = kwargs["ys"]
    xmin, xmax = xs-15, xs+15
    ymin, ymax = ys-15, ys+15
    # Don't forget to change reff limits when changing from pixel to physical constraints
    #if 0 < amp1 < 50 and 0 < reff1 < 1000 and 0.1 < n1 < 10 and 0 < amp2 < 50 and 0 < reff2 < 1000 and 0.1 < n2 < 10:
    if (
        0 < amp1 < 500 and 0 < reff1 < 500 #and 0.1 < n1 < 10
        and 0 < amp2 < 500 and 0 < reff2 < 500 and 0.1 < n2 < 10
        and xmin < x2 < xmax and ymin < y2 < ymax
        ):
        return 0
    return -np.inf
    

def log_likelihood(theta, **kwargs):
    chisq = chisquared(theta, **kwargs)
    sigma = kwargs["sigma"]
    log_lhood = -np.log(sigma) - (1./2.) * np.log(2*np.pi) - (chisq / 2.)
    result = np.sum(log_lhood)
    return result


def chisquared(theta, **kwargs):
    sigma = kwargs["sigma"]
    arcIm = kwargs["arcIm"]
    simIm = np.zeros_like(arcIm)
    conv = convolved(theta, **kwargs)
    simIm[:,:] += conv[:,:] # add sim image to data image
    # cut out each knot
    knotbounds = kwargs["knotbounds"]
    trueKnot = arcIm[knotbounds[2]:knotbounds[3], knotbounds[0]:knotbounds[1]]
    simKnot = simIm[knotbounds[2]:knotbounds[3], knotbounds[0]:knotbounds[1]]
    # chisquared calculation
    result = (1/sigma**2) * (trueKnot - simKnot)**2
    return result


def convolved(theta, **kwargs): 
    #amp1, reff1, n1 = theta #BDW
    amp1, reff1, amp2, reff2, n2, x2, y2 = theta
    #amp, reff, n = preconv(theta, **kwargs) #, xs, ys
    #print(amp,reff, xs, ys)
    xs, ys = kwargs["xs"], kwargs["ys"]
    #x2, y2 = xs-3, ys-3
    xss, yss = kwargs["xss"], kwargs["yss"]
    star = kwargs["star"]
    xlo, xhi, ylo, yhi = kwargs["limits"]
    #sersic1 = Sersic2D(amplitude=amp1, r_eff=reff1, n=n1, x_0=xs, y_0=ys)
    sersic2 = Sersic2D(amplitude=amp2, r_eff=reff2, n=n2, x_0=x2, y_0=y2)
    gauss1 = Gaussian2D(amp1, x_mean=xs, y_mean=ys, x_stddev=reff1, y_stddev=reff1)
    #gauss2 = Gaussian2D(amp2, x_mean=x2, y_mean=y2, x_stddev=reff2, y_stddev=reff2)
    xsscut = xss[ylo:yhi, xlo:xhi]
    ysscut = yss[ylo:yhi, xlo:xhi]
    S1 = gauss1(xsscut, ysscut).value
    S2 = sersic2(xsscut, ysscut).value
    S1S2 = S1 + S2
    #S1 = gauss1(xsscut, ysscut) + gauss2(xsscut, ysscut)
    #S1 = gauss(xsscut, ysscut).value
    S1conv = convolve(S1S2, star)
    return S1S2#conv


def preconv(theta, **kwargs):
    flux, r_eff, n = theta
    #n = 4
    #rpix = pc_to_pix(r_eff, 6.2)
    ellip = 0
    a = r_eff
    b = (1 - ellip) * r_eff
    b_n = sp.gammaincinv(2*n,0.5)
    Sersic_total = 2*np.pi*a*b*n*sp.gamma(2*n)*np.exp(b_n)/(b_n**(2*n))
    amp = flux / Sersic_total
    return amp, r_eff, n #, xs, ys


def poisson(flux_eps, t_expose, gain=1.):
    ix = tuple([flux_eps<0])
    flux_eps[ix] = 0. # remove zeros to avoid nans in sqrt
    flux_photon = flux_eps / gain
    Nphoton = flux_photon * t_expose
    result = np.sqrt(Nphoton) / t_expose
    return result 


def pc_to_pix(r_pc, z):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DA = cosmo.angular_diameter_distance_z1z2(0 , z).value * 10**6 #returns in pc
    theta = r_pc / DA # radian
    theta = theta * (180 / np.pi) * 3600 # arcsec
    pixSize = 0.06 #HST WFC3 pixel size
    rpix = theta / pixSize
    return rpix


def pix_to_pc(rpix, z):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DA = cosmo.angular_diameter_distance_z1z2(0 , z).value * 10**6 #returns in pc
    pixSize = 0.06 #HST WFC3 pixel size
    theta = rpix * pixSize #arcsec
    theta = theta / ((180. / np.pi) * 3600) #radian
    r_pc = theta * DA #phys
    return r_pc

