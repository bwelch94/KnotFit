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
from scipy.interpolate import interp2d
from multiprocessing import Pool
# added inconsequential edit to test github auth


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




def runMCMC(theta_init, argdict, nwalkers=4, niter=500, outfile=None, multiprocess=False, 
            threshold_converged=100, min_improvement=0.01, full_output=False, **kwargs):
    ndim = len(theta_init)
    if min(theta_init) < 1e-5:
        frac = 0.1 * min(theta_init)
    else:
        frac = 1e-5
    pos = theta_init + frac * np.random.randn(nwalkers, ndim)
    
    if outfile:
        backend = emcee.backends.HDFBackend(outfile)
        backend.reset(nwalkers,ndim)
    else:
        backend = None

    index = 0
    autocorr = np.empty(niter)
    old_tau = np.inf 

    if multiprocess == True:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, kwargs=argdict, backend=backend, pool=pool)
            sampler.run_mcmc(pos, niter, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, kwargs=argdict, backend=backend)    
        #sampler.run_mcmc(pos, niter, progress=True)
        for sample in sampler.sample(pos, iterations=niter, progress=True):
            if sampler.iteration % 100:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            converged = np.all(tau * threshold_converged < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < min_improvement)
            if converged:
                break
            old_tau = tau
    if full_output:
        return sampler, autocorr, index
    else:
        return sampler


def log_probability(theta, **kwargs):
    pri = log_prior(theta, **kwargs)
    if not np.isfinite(pri):
        return -np.inf
    return pri + log_likelihood(theta,**kwargs)
    

def log_prior(theta, **kwargs):
    amp1, reff1 = theta#, amp2, reff2 = theta#, amp3, reff3 = theta
    xs = kwargs["xs"]
    ys = kwargs["ys"]
    #xmin, xmax = x2-3, x2+3
    #ymin, ymax = y2-3, y2+3
    # Don't forget to change reff limits when changing from pixel to physical constraints
    #if 0 < amp1 < 50 and 0 < reff1 < 1000 and 0.1 < n1 < 10 and 0 < amp2 < 50 and 0 < reff2 < 1000 and 0.1 < n2 < 10:
    if (
        0 < amp1 < 5000000 and 0. < reff1 < 500 #and 0.1 < n1 < 10
        #and 0 < amp2 < 500 and 0 < reff2 < 500 #and 0.1 < n2 < 10
        #and 0 < amp3 < 500 and 0 < reff3 < 500 #and 0.1 < n3 < 10
        #and 0 < amp4 < 500 and 0 < reff4 < 500 #and 0.1 < n4 < 10
        #and xmin < x2 < xmax and ymin < y2 < ymax
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
    # make simulated knot
    conv = convolved(theta, **kwargs)
    simIm[:,:] += conv[:,:] # add sim image to data image
    # chisquared calculation
    result = (1/sigma**2) * (arcIm - simIm)**2
    return result


def convolved(theta, **kwargs): 
    amp1, reff1, amp2, reff2 = theta #, amp3, reff3, amp4, reff4 = theta #
    xs, ys = kwargs["xs"], kwargs["ys"]
    xs2, ys2 = kwargs["xs2"], kwargs["ys2"]
    #xs3, ys3 = kwargs["xs3"], kwargs["ys3"]
    #xs4, ys4 = kwargs["xs4"], kwargs["ys4"]
    arcIm = kwargs["arcIm"]
    #x2, y2 = 4013.63699909436, 4780.744281984086 # z10
    #x3, y3 = 4016.430463104877, 4783.919833222558 # z10
    xss, yss = kwargs["xss"], kwargs["yss"]
    star = kwargs["star"]
    #sersic1 = Sersic2D(amplitude=amp1, r_eff=reff1, n=n1, x_0=xs, y_0=ys)
    #sersic2 = Sersic2D(amplitude=amp2, r_eff=reff2, n=n2, x_0=xs2, y_0=ys2)
    #sersic3 = Sersic2D(amplitude=amp3, r_eff=reff3, n=n3, x_0=x3, y_0=y3)
    gauss1 = Gaussian2D(amp1, x_mean=xs, y_mean=ys, x_stddev=reff1, y_stddev=reff1)
    gauss2 = Gaussian2D(amp2, x_mean=xs2, y_mean=ys2, x_stddev=reff2, y_stddev=reff2)
    #gauss3 = Gaussian2D(amp3, x_mean=xs3, y_mean=ys3, x_stddev=reff3, y_stddev=reff3)
    #gauss4 = Gaussian2D(amp4, x_mean=xs4, y_mean=ys4, x_stddev=reff4, y_stddev=reff4)
    combined =  gauss1 + gauss2 #+ gauss3 #+ gauss4
    #S1S2_hires = gauss2d(xss, yss, amplitude=amp1, x_mean=xs, y_mean=ys, x_stddev=reff1, y_stddev=reff1, theta=0) 
    S1S2 = combined(xss, yss)
    #S1S2 = rebin(S1S2_hires, arcIm.shape)
    S1conv = convolve(S1S2, star)
    return S1conv


def convolved_TEST(theta, **kwargs): 
    # assume you have a list called "clumps" with instances of Clump class (I think this should work?)
    clumps = kwards["clumps"]
    ind = 0
    for i in range(len(clumps)):
        n_params = clumps[i].get_num_params()
        clumps[i].update(theta[ind:ind+n_params])
        ind += n_params
    combined = clumps[0].draw()
    if len(clumps) > 1:
        for i in range(1,len(clumps)):
            combined += clumps[i].draw()
    arcIm = kwargs["arcIm"]
    xss, yss = kwargs["xss"], kwargs["yss"]
    psf = kwargs["psf"]
    if kwargs["resolution"] == 'high':
        galaxy = combined(xss, yss)
        galaxy_rebin = rebin(galaxy, arcim.shape)
        return convolve(galaxy_rebin, psf)
    else:
        galaxy = combined(xss, yss)
        return convolve(galaxy, psf)


def gauss2d(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = x_stddev ** 2
    ystd2 = y_stddev ** 2
    xdiff = x - x_mean
    ydiff = y - y_mean
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)))


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

def rebin(a, shape):
    """
    Re-bin hi-resolution simulated image to lower (HST) resolution. 
    """
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)