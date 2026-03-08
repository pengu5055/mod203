"""
Contains functions used throught the project.
"""
import numpy as np
from scipy.optimize import brentq


def numerov(x, y, k):
    """
    Numerov's method for solving second order ODEs of the form
    y'' = k(x) * y.
    """
    n = len(x)
    h = x[1] - x[0]
    y_new = np.zeros(n)
    y_new[0] = y[0]
    y_new[1] = y[1]

    for i in range(2, n):
        y_new[i] = (2 * (1 - (5 * h**2 * k[i-1]) / 12) * y_new[i-1] - (1 + (h**2 * k[i-2]) / 12) * y_new[i-2]) / (1 + (h**2 * k[i]) / 12)

    return y_new

def k_hydrogen(E, r, l):
    """
    The function k(r) for the hydrogen atom, given by:
        k(r) = -2 * (E + 1/r) + l*(l+1)/r^2
    used to solve:
        u''(r) = k(r) * u(r)
    where u(r) = r * R(r), and R(r) is the radial wavefunction.
    """
    return -2 * (E + 1/r) + l*(l+1)/r**2

def seed_hydrogen(r, E, l):
    """
    Provides the initial conditions for the hydrogen atom problem.
    For small r, the solution behaves like r^(l+1). With that 
    we can seed the first two points of the integration as:

    u(r) ~ r^(l+1) => u(0) = 0 and u'(0) = (l+1)*r^l => u(h) ~ h^(l+1)
    """

    return r[0]**(l+1), r[1]**(l+1)


def shoot(shoot_par, x, k_func, y_seed_func):
    """
    Single shot using integration of the ODE with given parameters.
    Integrate with value `shoot_par` and return the boundary value
    at the end of the integration interval.

    Parameters
    ----------
    shoot_par : float
        The parameter to shoot with, e.g. the energy E in the hydrogen atom case.
    x : np.ndarray
        The spatial grid to integrate over.
    k_func : callable
        The function k(x, shoot_par) that defines the ODE to solve.
    y_seed_func : callable
        A function y_seed(x, shoot_par) that provides the initial conditions 
        for the integration, given shoot_par.

    Returns
    -------
    float
        The value of the solution at the end of the integration interval.
    """
    k = k_func(x, shoot_par)
    y0, y1 = y_seed_func(x, shoot_par)
    y = numerov(x, np.array([y0, y1]), k)

    return y[-1]

def scan_eigenvalues(x, k_func, y_seed_func, shoot_par_range, n_scan=500):
    """
    Scan for eigenvalues by shooting over a range of parameters and 
    looking for sign changes in the boundary value at the end of the integration interval.

    Parameters
    ----------
    x : np.ndarray
        The spatial grid to integrate over.
    k_func : callable
        The function k(x, shoot_par) that defines the ODE to solve.
    y_seed_func : callable
        A function y_seed(x, shoot_par) that provides the initial conditions 
        for the integration, given shoot_par.
    shoot_par_range : tuple
        A tuple (shoot_par_min, shoot_par_max) defining the range of parameters to scan.
    n_scan : int, optional
        The number of parameter values to scan over, by default 500.

    Returns
    -------
    brackets : list of tuples
        A list of tuples (shoot_par_left, shoot_par_right) where a sign change was detected,
        indicating a potential eigenvalue between shoot_par_left and shoot_par_right.
    """
    par_vals = np.linspace(shoot_par_range[0], shoot_par_range[1], n_scan)
    residuals = np.array([shoot(par, x, k_func, y_seed_func) for par in par_vals])
    brackets = []
    for i in range(len(par_vals) - 1):
        if np.sign(residuals[i]) != np.sign(residuals[i+1]):
            brackets.append((par_vals[i], par_vals[i+1]))

    return brackets

def find_eigenvalues(x, k_func, y_seed_func, shoot_par_range, n_scan=500):
    """
    Find eigenvalues by first scanning for sign changes and then using root finding to 
    refine the eigenvalue estimates.

    Parameters
    ----------
    x : np.ndarray
        The spatial grid to integrate over.
    k_func : callable
        The function k(x, shoot_par) that defines the ODE to solve.
    y_seed_func : callable
        A function y_seed(x, shoot_par) that provides the initial conditions 
        for the integration, given shoot_par.
    shoot_par_range : tuple
        A tuple (shoot_par_min, shoot_par_max) defining the range of parameters to scan.
    n_scan : int, optional
        The number of parameter values to scan over for the initial scan, by default 500.

    Returns
    -------
    eigenvalues : list of floats
        A list of refined eigenvalue estimates found in the specified range.
    """
    brackets = scan_eigenvalues(x, k_func, y_seed_func, shoot_par_range, n_scan)
    eigenvalues = []
    for left, right in brackets:
        try:
            eigenvalue = brentq(shoot, left, right, args=(x, k_func, y_seed_func))
            eigenvalues.append(eigenvalue)
        except ValueError:
            # If brentq fails, we can skip this bracket
            print(f"Warning: brentq failed to find a root between {left} and {right}. Skipping this bracket.")
            continue
    
    return eigenvalues

def get_wavefunction(shoot_par, x, k_func, y_seed_func):
    """
    Get the wavefunction for a given parameter by integrating the ODE.
    Built in normalization of the wavefunction using the trapezoidal rule.

    Parameters
    ----------
    shoot_par : float
        The parameter to use for integration, e.g. the energy E in the hydrogen atom case.
    x : np.ndarray
        The spatial grid to integrate over.
    k_func : callable
        The function k(x, shoot_par) that defines the ODE to solve.
    y_seed_func : callable
        A function y_seed(x, shoot_par) that provides the initial conditions 
        for the integration, given shoot_par.

    Returns
    -------
    np.ndarray
        The wavefunction values at each point in x.
    """
    k = k_func(x, shoot_par)
    y0, y1 = y_seed_func(x, shoot_par)
    y = numerov(x, np.array([y0, y1]), k)
    norm = np.trapezoid(y**2, x)

    return y / np.sqrt(norm)

