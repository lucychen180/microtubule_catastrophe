"""
Functions for modeling or computing using the Gamma distribution.
The Gamma distribution models the waiting time for `alpha` consecutive 
arrivals of Poisson processes that each have an arrival rate of `beta`.
"""

import warnings

import numpy as np
import scipy
import scipy.stats as st

def draw(params, size, rg=None):
    """
    Randomly generate `size` values from the Gamma distribution,
    parameterized by `params`.
    
    Parameters
    ----------
    params : tuple (alpha, beta) of type float
        Parameterizes the Gamma distribution to draw from.
    size : int
        Number of samples to generate.
    rg : numpy.random.Generator object, default None
        The rng object generating the samples.
    
    Returns
    -------
    output : float or ndarray
        Drawn samples from the parameterized gamma distribution.
    """
    # Initialize generator if none specified
    if rg is None:
        rg = np.random.default_rng()
        
    alpha, beta = params
    return rg.gamma(alpha, 1/beta, size=size)


def pdf(params, t):
    """
    Compute the probability density function for the Gamma distribution
    parameterized by `params` given measurements `t`.
    
    Parameters
    ----------
    params : tuple (alpha, beta) of type float
        Parameterizes the Gamma distribution to compute from.
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : float or ndarray
        PDF of the parameterized Gamma distribution at values `t`.
    """
    alpha, beta = params
    return st.gamma.pdf(t, alpha, loc=0, scale=1/beta)


def cdf(params, t):
    """
    Compute the cumulative distribution function for the Gamma
    distribution parameterized by `params` given measurements `t`.
    
    Parameters
    ----------
    params : tuple (alpha, beta) of type float
        Parameterizes the Gamma distribution to compute from.
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : float or ndarray
        CDF of the parameterized Gamma distribution at values `t`.
    """
    alpha, beta = params
    return st.gamma.cdf(x, alpha, loc=0, scale=1/beta)


def log_like(params, t):
    """
    Compute the log likelihood of i.i.d. measurements `t` for a Gamma
    distribution parameterized by `params`.
    
    Parameters
    ----------
    params : tuple (alpha, beta) of type float
        Parameterizes the Gamma distribution to compute from.
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : float or ndarray
        Log likelihood of the parameterized Gamma distribution at `t`.
    """
    alpha, beta = params

    # Punish invalid values
    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(t, alpha, loc=0, scale=1/beta))


def mle(t):
    """
    Compute the maximum likelihood estimates of Gamma distribution
    parameters given i.i.d. measurements `t`.
    
    Parameters
    ----------
    t : array_like
        One-dimensional array of data.
    
    Returns
    -------
    output : ndarray
        MLEs of the parameters (alpha, beta) of the Gamma distribution.
    """
    # We obtain initial parameter estimates using the method of moments.
    # The mean of the Gamma distribution is alpha / beta and 
    # the variance is alpha / beta ** 2. 
    # Thus, if obtain the plug-in estimate for the mean and variance, 
    # the following are good guesses:
    
    # Initial guess
    t_bar = np.mean(t)
    beta_guess = t_bar / np.var(t)
    alpha_guess = t_bar * beta_guess
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like(params, t),
            x0=(alpha_guess, beta_guess),
            args=(t,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
