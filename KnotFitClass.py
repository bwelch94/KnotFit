'''
Basically here I am going to recreate the KnotFit code but in a class structure
This should give greater flexibility and functionality without having to edit the code directly anymore
We'll see how this goes...


TO-DO:
 - add source plane position optimization

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
from scipy.integrate import dblquad

# Define functions for fitting outside of class structure
# its a bit ugly, but this helps with parallelization of the MCMC

def log_probability(theta, **kwargs):
    pri = log_prior(theta, **kwargs)
    if not np.isfinite(pri):
        return -np.inf
    return pri + log_likelihood(theta,**kwargs)
    

def log_prior(theta, **kwargs):
    clumps = kwargs["clumps"]
    ind = 0
    for i in range(len(clumps)):
        n_params = clumps[i].get_num_params()
        clumps[i].update(theta[ind:ind+n_params])
        ind += n_params
    priors = sum([clump.prior() for clump in clumps])    
    return priors

def log_likelihood(theta, **kwargs):
    chisq = chisquared(theta, **kwargs)
    sigma = kwargs["sigma"]
    log_lhood = -np.log(sigma) - (1./2.) * np.log(2*np.pi) - (chisq / 2.)
    result = np.sum(log_lhood)
    return result


def chisquared(theta, **kwargs):
    sigma = kwargs["sigma"]
    arcim = kwargs["arcim"]
    simim = np.zeros_like(arcim)
    #simim -= 4.
    # make simulated knot
    conv = convolved(theta, **kwargs)
    simim[:,:] += conv[:,:] # add sim image to data image
    # chisquared calculation
    result = (1/sigma**2) * (arcim - simim)**2
    return result



def convolved(theta, **kwargs): 
    # assume you have a list called "clumps" with instances of Clump class
    clumps = kwargs["clumps"]
    ind = 0
    for i in range(len(clumps)):
        n_params = clumps[i].get_num_params()
        clumps[i].update(theta[ind:ind+n_params])
        ind += n_params
    combined = clumps[0].draw()
    if len(clumps) > 1:
        for i in range(1,len(clumps)):
            combined += clumps[i].draw()
    arcim = kwargs["arcim"]
    xss, yss = kwargs["xss"], kwargs["yss"]
    psf = kwargs["psf"]
    resol = kwargs["resolution"]
    if resol == 'high':
        galaxy = combined(xss, yss)
        galaxy_rebin = rebin(galaxy, arcim.shape)
        return convolve(galaxy_rebin, psf)

    elif resol == 'mixed': # be sure to run init_mixed_resolution before using this method!
    # this gets suuuuper slow for now. 
        xss_hires = kwargs["xss_hires"]
        yss_hires = kwargs["yss_hires"]
        xslice_hires = kwargs["xslice_hires"]
        yslice_hires = kwargs["yslice_hires"]
        galaxy = combined(xss, yss)
        galaxy_hires = combined(xss_hires, yss_hires)
        if kwargs["super_res"]:
            galaxy = rebin(galaxy, arcim.shape)
            galaxy_rebin = galaxy_hires.mean() 
        else:
            galaxy_rebin = rebin(galaxy_hires, hires_im.shape)
        galaxy[yslice_hires, xslice_hires] = galaxy_rebin
        return convolve(galaxy, psf)

    else:
        galaxy = combined(xss, yss)
        return convolve(galaxy, psf)


def rebin(a, shape):
    """
    Re-bin hi-resolution simulated image to lower (HST) resolution. 
    """
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)




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


    def set_sigma(self, rmsfile=None, unit='eps',exp_time=1, photfnu=1e-8, gain=1.):
        if rmsfile:
            rmsf = fits.open(rmsfile)
            rms = rmsf[0].data
            rmscut = rms[self.ylo:self.yhi, self.xlo:self.xhi]
            self.sigma = self.poisson(np.copy(self.arcim), 
                                      t_expose=exp_time, 
                                      unit=unit, 
                                      gain=gain, 
                                      photfnu=photfnu) + (rmscut*exp_time)
        else:
            self.sigma = 0.01 * self.poisson(np.copy(self.arcim), 
                                      t_expose=exp_time, 
                                      gain=gain, 
                                      photfnu=photfnu) + 0.01


    def poisson(self, flux, t_expose=1, gain=1., unit='eps', photfnu=1e-8):
        if unit == 'eps':
            ix = tuple([flux<0])
            flux[ix] = 0. # remove zeros to avoid nans in sqrt
            flux_photon = flux / gain
            Nphoton = flux_photon #* t_expose
            result = np.sqrt(Nphoton) #/ t_expose
        elif unit == 'nJy':
            ix = tuple([flux<0])
            flux[ix] = 0. # remove zeros to avoid nans in sqrt
            flux_electron = (flux * 1e-9) / photfnu # put in Jy, divide by Jy*sec/elec.
            Nphoton = flux_electron * gain
            result = np.sqrt(Nphoton)
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

        if self.resolution == 'high':
            galaxy = combined(self.xss, self.yss)
            galaxy_rebin = self.rebin(galaxy, self.arcim.shape)
            return convolve(galaxy_rebin, self.psf)

        elif self.resolution == 'mixed': # be sure to run init_mixed_resolution before using this method!
        # this gets suuuuper slow for now. 
            galaxy = combined(self.xss, self.yss)
            galaxy_hires = combined(self.xss_hires, self.yss_hires)
            if self.super_res:
                galaxy = self.rebin(galaxy, self.arcim.shape)
                #print(galaxy_hires.shape, galaxy_hires.sum(), galaxy_hires.mean())
                galaxy_rebin = galaxy_hires.mean() #/ galaxy_hires.shape[0]
            else:
                galaxy_rebin = self.rebin(galaxy_hires, self.hires_im.shape)
            galaxy[self.yslice_hires, self.xslice_hires] = galaxy_rebin
            return convolve(galaxy, self.psf)

#        elif self.resolution == 'integrate':
#            # this will be glacially slow. 
#            galaxy = self.discretize_integrate_2D(combined)
#            return convolve(galaxy, self.psf)

        else:
            galaxy = combined(self.xss, self.yss)
            return convolve(galaxy, self.psf)


    def rebin(self, a, shape):
        """
        Re-bin hi-resolution simulated image to lower (HST) resolution. 
        """
        sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)


    def discretize_integrate_2D(self, model):
        """
        Discretize model by integrating the model over the pixel.
        """
        # Set up grid
        #x = np.arange(x_range[0] - 0.5, x_range[1] + 0.5)
        #y = np.arange(y_range[0] - 0.5, y_range[1] + 0.5)
        x = self.xss
        y = self.yss
        values = np.empty((y.shape[0] - 1, x.shape[1] - 1))

     # Integrate over all pixels
        #print(values.shape, x.shape, y.shape)
        for i in range(x.shape[1] - 1):
            for j in range(y.shape[0] - 1):
                #print(x[i, j], x[i+1, j+1], y[i, j], y[i+1, j+1])
                values[j, i] = np.abs(dblquad(lambda y, x: model(x, y), x[i, j], x[i + 1, j+1],
                                    lambda x: y[i, j], lambda x: y[i+1, j + 1])[0])
        return values   


#    def init_integrate_grid(x_range, y_range, resolution):
#        x_hires = np.arange(xlo - (resolution/2.), xhi - (resolution/2.), resolution)
#        y_hires = np.arange(ylo - (resolution/2.), yhi - (resolution/2.), resolution)



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


    def init_mixed_v2(self, xlo_hires, xhi_hires, ylo_hires, yhi_hires, xss_hires, yss_hires):
        self.xslice_hires = slice(xlo_hires - self.xlo, xhi_hires - self.xlo)
        self.yslice_hires = slice(ylo_hires - self.ylo, yhi_hires - self.ylo)

        self.xss_hires = xss_hires
        self.yss_hires = yss_hires


    def build_kwarg_dict(self):
        argdict = {
                    "clumps"     : self.clumps,
                    "sigma"      : self.sigma, 
                    "arcim"      : self.arcim,
                    "psf"        : self.psf,
                    "resolution" : self.resolution,
                    "xss"        : self.xss,
                    "yss"        : self.yss
        }
        if self.resolution == 'mixed':
            argdict["xss_hires"]    = self.xss_hires
            argdict["yss_hires"]    = self.yss_hires
            argdict["xslice_hires"] = self.xslice_hires
            argdict["yslice_hires"] = self.yslice_hires

        return argdict



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
            argdict = self.build_kwarg_dict()
            with Pool(n_core) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, kwargs=argdict, backend=backend, pool=pool)
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
            self.theta = np.array([amp0, r0, x0, y0])
        elif profile == "Gaussian-asym":
            self.rx = r0
            self.ry = r0
            self.rot_angle = rot_angle = 0.
            self.theta = np.array([amp0, r0, r0, rot_angle, x0, y0])
        elif profile == "Delta":
            self.theta = np.array([x0, y0, amp0])
        elif profile == "Sersic":
            n0 = 1.
            self.n = n0
            self.ellip = ellip = 0.
            self.rot_angle = rot_angle = 0.
            self.theta = np.array([amp0, r0, n0, ellip, rot_angle, x0, y0])
        self.fixed = False
        self.profile = profile

    def draw(self):
        if self.profile == 'Gaussian':
            return Gaussian2D(self.amp, x_mean=self.x, y_mean=self.y,
                                x_stddev=self.r, y_stddev=self.r)
        elif self.profile == 'Gaussian-asym':
            return Gaussian2D(self.amp, x_mean=self.x, y_mean=self.y,
                                x_stddev=self.rx, y_stddev=self.ry, theta=self.rot_angle)
        elif self.profile == 'Delta':
            return Delta2D(self.x, self.y, self.amp)
        elif self.profile == 'Sersic':
            return Sersic2D(self.amp, r_eff=self.r, n=self.n, x_0=self.x, y_0=self.y, 
                            ellip=self.ellip, theta=self.rot_angle)


    def prior(self):
        if self.profile == "Gaussian":
            if (self.amin < self.amp < self.amax 
                and self.rmin < self.r < self.rmax
                and self.xmin < self.x < self.xmax 
                and self.ymin < self.y < self.ymax):
                return 0
            return -np.inf 
        elif self.profile == "Gaussian-asym":
            rads = np.array([rx, ry])
            ellip = np.sqrt(1 - (min(rads) / max(rads)))
            if (self.amin < self.amp < self.amax 
                and self.rmin < self.rx < self.rmax 
                and self.rmin < self.ry < self.rmax 
                and self.rot_min <= self.rot_angle <= self.rot_max
                and self.emin <= ellip <= self.emax
                and self.xmin < self.x < self.xmax 
                and self.ymin < self.y < self.ymax):
                return 0
            return -np.inf
        elif self.profile == "Delta":
            if (self.amin < self.amp < self.amax 
                and self.xmin < self.x < self.xmax 
                and self.ymin < self.y < self.ymax):
                return 0
            return -np.inf
        elif self.profile == "Sersic":
            if (self.amin < self.amp < self.amax 
                and self.rmin < self.r < self.rmax 
                and self.nmin < self.n < self.nmax 
                and self.emin < self.ellip < self.emax 
                and self.rot_min <= self.rot_angle <= self.rot_max
                and self.xmin < self.x < self.xmax 
                and self.ymin < self.y < self.ymax):
                return 0
            return -np.inf


    def set_prior(self, amin=0, amax=1e10, rmin=0, rmax=1e10, 
                xmin=0, xmax=1e10, ymin=0, ymax=1e10, 
                nmin=0, nmax=1e10, rot_min=0., rot_max=2*np.pi,
                emin=0., emax=1.):
        self.amin = amin
        self.amax = amax
        self.rmin = rmin
        self.rmax = rmax
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nmin = nmin
        self.nmax = nmax
        self.rot_min = rot_min
        self.rot_max = rot_max 
        self.emin = emin
        self.emax = emax

        
    def update(self, theta):
        if self.fixed:
            pass
        else:
            self.theta = theta
            if self.profile == "Gaussian":
                self.amp = theta[0]
                self.r = theta[1]
                self.x = theta[2]
                self.y = theta[3]
            elif self.profile == "Gaussian-asym":
                self.amp = theta[0]
                self.rx = theta[1]
                self.ry = theta[2]
                self.rot_angle = theta[3]
                self.x = theta[4]
                self.y = theta[5]
            elif self.profile == "Delta":
                self.x = theta[0]
                self.y = theta[1]
                self.amp = theta[2]
            elif self.profile == "Sersic":
                self.amp=theta[0]
                self.r = theta[1]
                self.n = theta[2]
                self.ellip = theta[3]
                self.rot_angle = theta[4]
                self.x = theta[5]
                self.y = theta[6]

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


