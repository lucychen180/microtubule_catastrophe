"""
Functions for modeling or computing using a proposed alternative distribution.
The alternative distribution models the waiting time for two Poisson 
consecutive processes to arrive, in which the first process has a rate `beta1`
and the second (which occurs after the first) has a rate `beta2`.
"""

import warnings

import numpy as np
import scipy
import scipy.stats as st

def draw(params, size, rg=None):
    """
    Randomly generate `size` values from the alternative distribution, 
    parameterized by `params`.
    
    Parameters
    ----------
    params : tuple (beta1, beta2) of type float
        Parameterizes the alternative distribution to draw from.
    size : int
        Number of samples to generate.
    rg : numpy.random.Generator object, default None
        The rng object generating the samples.
    
    Returns
    -------
    output : float or ndarray
        Drawn samples from the parameterized alternative distribution.
    """
    # Initialize generator if none specified
    if rg is None:
        rg = np.random.default_rng()

    beta1, beta2 = params
    
    # Each event is a different Poisson process
    t1 = rg.exponential(1/beta1, size=size)
    t2 = rg.exponential(1/beta2, size=size)

    # Event 2 can only happen after event 1, so wait times cumulate
    return t1 + t2

def pdf(params, t):
    """
    Compute the probability density function for the alternative 
    distribution parameterized by `params` given measurements `t`.
    
    Parameters
    ----------
    params : tuple (beta1, beta2) of type float
        Parameterizes the alternative distribution to compute from.
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : float or ndarray
        PDF of the parameterized alt distribution at values `t`.
    """
    beta1, beta2 = params
    const = (beta1 * beta2) / (beta2 - beta1)
    return const * (np.exp(-beta1 * t) - np.exp(-beta2 * t))


def cdf(params, t):
    """
    Compute the cumulative distribution function for the alternative
    distribution parameterized by `params` given measurements `t`.
    
    Parameters
    ----------
    params : tuple (beta1, beta2) of type float
        Parameterizes the alternative distribution to compute from.
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : float or ndarray
        CDF of the parameterized alt distribution at values `t`.
    """
    beta1, beta2 = params
    const = (beta1 * beta2) / (beta2 - beta1)
    vals = (1 - np.exp(-beta1 * t)) / beta1 - (1 - np.exp(-beta2 * t)) / beta2
    return const * vals

def log_like(params, t):
    """
    Compute the log likelihood of i.i.d. measurements `t` for the
    alternative distribution parameterized by `params`.
    
    Parameters
    ----------
    params : tuple (beta1, beta2) of type float
        Parameterizes the alternative distribution to compute from.
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : float or ndarray
        Log likelihood of the parameterized alt distribution at `t`.
    """
    beta1, beta2 = params

    # Punish invalid values
    if beta1 <= 0 or beta2 <= 0:
        return -np.inf
    
    # If beta1 \approx beta2, this is basically a Gamma distribution w/ alpha=2
    alpha = 2
    if np.isclose(beta1, beta2):
        return np.sum(scipy.stats.gamma.logpdf(t, alpha, loc=0, scale=1/beta1))
    
    else:
        return np.sum(np.log(pdf(params, t)))


def mle(t):
    """
    Compute the maximum likelihood estimates of the alternative 
    distribution parameters given i.i.d. measurements `t`.
    
    Parameters
    ----------
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : ndarray
        MLEs of the parameters (beta1, beta2) of the alt distribution.
    """
    # Because the alternative distribution models a 2-step process, 
    # for our initial guess we assume that each step occurs at the same rate. 
    # Thus, our estimates for beta1 and beta2 are the same, and we obtain them
    # by multiplying the average rate (1 / mean time) by 2.
   
    # Initial guess
    beta_guess = (1 / np.mean(t)) * 2
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like(params, t),
            x0=[beta_guess, beta_guess],
            args=(t,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)