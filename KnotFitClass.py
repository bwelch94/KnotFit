'''
Basically here I am going to recreate the KnotFit code but in a class structure
This should give greater flexibility and functionality without having to edit the code directly anymore
We'll see how this goes...


TO-DO:
- Use combination of all stars in the field to avoid weird lopsided thing you're seeing now
- Test this whole class structure thing

'''

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



class GalModel:
    """
    Docstring:

    To do: 
    Write docstring I guess
    maybe add some initialization functions? ala initDeflectionImage, makeMagnificationMap??
    """

    def __init__(self, init='Basic'):
        self.clumps = []
        # initialize variables, each of which will have to be set later by user
        # maybe there's a better way to do this???
        if init == 'Basic':
            self.sigma = 0
            self.xlo = 0
            self.ylo = 0
            self.xhi = 1000
            self.yhi = 1000
            self.xss = np.zeros((100,100))
            self.yss = np.zeros((100,100))
            self.arcim = np.zeros((100,100))
            self.hi_res = False
#       elif init == 'Advanced':
#           x, y, im = self.init_deflection_image()
#           self.arcim = im[self.ylo:self.yhi, self.xlo:self.xhi]
#
#           mag = self.make_magnification_map()
#
#           if rmsfile:
#               self.set_psf(style='star', star_imdata=im, star_pos=(2232,1622))



    def add_clump(self, clump):
        self.clumps.append(clump)


    def set_psf(self, style='star', star_pos=None, star_imdata=None, extent=15, pos_list=None):
        """
        pos_list should be a list of tuples with (x,y) positions for each star to be used on PSF calculation.
        """
        if style == 'star':
            self.psf = self.star_gen(star_imdata, star_pos, extent)
        elif style == 'list':
            stars = []
            for pos in pos_list:
                stars.append(self.star_gen(star_imdata, pos, extent))
            stars = np.array(stars)
            self.psf = np.mean(stars, axis=0)



    def star_gen(self, star_imdata, star_pos, extent=15):
        xmin, xmax = (star_pos[0]-(extent/2), star_pos[0]+((extent/2)))
        xmin, xmax = int(xmin), int(xmax)
        ymin, ymax = (star_pos[1]-(extent/2), star_pos[1]+((extent/2)))
        ymin, ymax = int(ymin), int(ymax)
        star = star_imdata[ymin:ymax, xmin:xmax]
        return star


    def set_sigma(self, rmsfile=None, exp_time=5000, sig=0):
        if rmsfile:
            rmsf = fits.open(rmsfile)
            rms = rmsf[0].data
            rmscut = rms[self.ylo:self.yhi, self.xlo:self.xhi]
            self.sigma = self.poisson(self.arcim, t_expose=exp_time) + rmscut
        else:
            self.sigma = sig


    def poisson(self, flux_eps, t_expose, gain=1.):
        ix = tuple([flux_eps<0])
        flux_eps[ix] = 0. # remove zeros to avoid nans in sqrt
        flux_photon = flux_eps / gain
        Nphoton = flux_photon * t_expose
        result = np.sqrt(Nphoton) / t_expose
        return result 


    def min_fit(self, param0, method='Nelder-Mead', maxiter=5000):
        sol = minimize(self.neg_lhood, param0, method=method,
                        options={'maxiter':maxiter})
        return sol


    def neg_lhood(self, theta):
        return -self.log_probability(theta)


    def log_probability(self, theta):
        pri = self.log_prior(theta)
        if not np.isfinite(pri):
            return -np.inf 
        return pri + self.log_likelihood(theta)


    def log_prior(self, theta):
        ind = 0
        for i in range(len(self.clumps)):
            n_params = self.clumps[i].get_num_params()
            self.clumps[i].update(theta[ind:ind+n_params])
            ind += n_params
        priors = sum([clump.prior() for clump in self.clumps])
        return priors


    def log_likelihood(self, theta):
        log_lhood = -np.log(self.sigma) - (1./2.) * np.log(2*np.pi) - (self.chisquared(theta) / 2.)
        result = np.sum(log_lhood)
        return result


    def chisquared(self, theta):
        conv = self.convolved(theta)
        sim_copy = np.copy(self.simim)
        sim_copy[:,:] += conv[:,:]
        result = (1. / self.sigma**2) * (self.arcim - sim_copy)**2
        return result


    def convolved(self, theta):
        ind = 0
        for i in range(len(self.clumps)):
            n_params = self.clumps[i].get_num_params()
            self.clumps[i].update(theta[ind:ind+n_params])
            ind += n_params
        combined = self.clumps[0].draw()
        if len(self.clumps) > 1:
            for i in range(1,len(self.clumps)):
                combined += self.clumps[i].draw()
        galaxy = combined(self.xss, self.yss)

        if self.resolution == 'high':
            galaxy_rebin = self.rebin(galaxy, self.arcim.shape)
            return convolve(galaxy_rebin, self.psf)

        elif self.resolution == 'mixed': # be sure to run init_mixed_resolution before using this method!
        # this gets suuuuper slow for now. I'll probably need to multiprocess to get this done in a reasonable time. Blarg!
            galaxy_hires = combined(self.xss_hires, self.yss_hires)
            galaxy_rebin = self.rebin(galaxy_hires, self.hires_im.shape)
            galaxy[self.yslice_hires, self.xslice_hires] = galaxy_rebin
            return convolve(galaxy, self.psf)

        else:
            return convolve(galaxy, self.psf)


    def rebin(self, a, shape):
        """
        Re-bin hi-resolution simulated image to lower (HST) resolution. 
        """
        sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)


    def init_mixed_resolution(self, res, xlo_hires, xhi_hires, ylo_hires, yhi_hires, xx, yy, ax, ay):
        self.xslice_hires = slice(xlo_hires - self.xlo, xhi_hires - self.xlo)
        self.yslice_hires = slice(ylo_hires - self.ylo, yhi_hires - self.ylo)
        xx_slice = xx[self.yslice_hires, self.xslice_hires]
        yy_slice = yy[self.yslice_hires, self.xslice_hires]
        ax_slice = ax[self.yslice_hires, self.xslice_hires]
        ay_slice = ay[self.yslice_hires, self.xslice_hires]

        ax_interp = interp2d(xx_slice, yy_slice, ax_slice)
        ay_interp = interp2d(xx_slice, yy_slice, ay_slice)
        x_hires = np.arange(xlo_hires, xhi_hires, res)
        y_hires = np.arange(ylo_hires, yhi_hires, res)
        xx_hires, yy_hires = np.meshgrid(x_hires, y_hires)

        ax_hires = ax_interp(x_hires, y_hires)
        ay_hires = ay_interp(x_hires, y_hires)
        self.xss_hires = xx_hires - ax_hires
        self.yss_hires = yy_hires - ay_hires



    def runMCMC(self, theta_init, nwalkers=4, niter=500, outfile=None, multiprocess=False, 
                threshold_converged=100, min_improvement=0.01, full_output=False,
                n_core=None, **kwargs):
        ndim = len(theta_init)
        if nwalkers < 2 * ndim:
            nwalkers = 2 * ndim
            print('Setting nwalkers = 2 * ndim')
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

        if multiprocess:
            if n_core == None:
                n_core = len(os.sched_getaffinity(0)) # returns number of CPU cores that current process can use
            with Pool(n_core) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, backend=backend, pool=pool)
                sampler.run_mcmc(pos, niter, progress=True)
                #for sample in sampler.sample(pos, iterations=niter, progress=True):
                #    if sampler.iteration % 100:
                #        continue
                #    tau = sampler.get_autocorr_time(tol=0)
                #    autocorr[index] = np.mean(tau)
                #    index += 1

                #    converged = np.all(tau * threshold_converged < sampler.iteration)
                #    converged &= np.all(np.abs(old_tau - tau) / tau < min_improvement)
                #    if converged:
                #        print("Converged!")
                #        break
                #    old_tau = tau

        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, backend=backend)    
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
                    print("Converged!")
                    break
                old_tau = tau
        if full_output:
            return sampler, autocorr, index
        else:
            return sampler




class Clump:

    def __init__(self, x0, y0, amp0=1., r0=1., profile="Gaussian"):
        self.x = x0
        self.y = y0
        self.amp = amp0
        self.r = r0
        if profile == "Gaussian":
            self.theta = np.array([amp0, r0])
        elif profile == "Delta":
            self.theta = np.array([x0, y0, amp0])
        self.fixed = False
        self.profile = profile

    def draw(self):
        if self.profile == 'Gaussian':
            return Gaussian2D(self.amp, x_mean=self.x, y_mean=self.y,
                                x_stddev=self.r, y_stddev=self.r)
        elif self.profile == 'Delta':
            return Delta2D(self.x, self.y, self.amp)


    def prior(self):
        if self.profile == "Gaussian":
            if (self.amin < self.amp < self.amax and self.rmin < self.r < self.rmax):
                return 0
            return -np.inf 
        elif self.profile == "Delta":
            if (self.amin < self.amp < self.amax and self.xmin < self.x < self.xmax and self.ymin < self.y < self.ymax):
                return 0
            return -np.inf


    def set_prior(self, amin=0, amax=1e10, rmin=0, rmax=1e10, xmin=0, xmax=1e10, ymin=0, ymax=1e10):
        self.amin = amin
        self.amax = amax
        self.rmin = rmin
        self.rmax = rmax
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def update(self, theta):
        if self.fixed:
            pass
        else:
            self.theta = theta
            if self.profile == "Gaussian":
                self.amp = theta[0]
                self.r = theta[1]
            elif self.profile == "Delta":
                self.x = theta[0]
                self.y = theta[1]
                self.amp = theta[2]

    def get_num_params(self):
        return len(self.theta)


    


class Delta2D:
    def __init__(self, x0=0, y0=0, amp=1):
        self.x0 = x0
        self.y0 = y0
        self.amp = amp

    @staticmethod
    def evaluate(xx, yy, x0, y0, amp):
        delta_x = 0.01#(max(xx[0]) - min(xx[0])) / len(xx[0])
        delta_y = 0.01#(max(yy[:][0]) - min(yy[:][0])) / len(yy[0])
        result = np.zeros(xx.shape)
        result[(xx>=x0-delta_x) & (xx<=x0+delta_x) & (yy>=y0-delta_y) & (yy<=y0+delta_y)] = amp
        return result
    
    def __call__(self, xx, yy):
        return self.evaluate(xx, yy, self.x0, self.y0, self.amp)