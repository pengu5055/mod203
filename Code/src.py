"""
Contains functions used throught the project.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import factorial, genlaguerre


def numerov(x, y, k, renorm_every=None):
    """
    Numerov's method for solving second order ODEs of the form
    y'' = k(x) * y.
    
    Parameters
    ----------
    x : np.ndarray
        The spatial grid.
    y : np.ndarray
        Initial conditions [y0, y1].
    k : np.ndarray
        The function k(x) evaluated on the grid.
    renorm_every : int, optional
        If set, renormalize the solution every `renorm_every` steps to prevent
        overflow. Preserves zero structure but not amplitude. Default is None (no renorm).
    """
    n = len(x)
    h = x[1] - x[0]
    y_new = np.zeros(n)
    y_new[0] = y[0]
    y_new[1] = y[1]

    for i in range(2, n):
        y_new[i] = (2 * (1 - (5 * h**2 * k[i-1]) / 12) * y_new[i-1] - (1 + (h**2 * k[i-2]) / 12) * y_new[i-2]) / (1 + (h**2 * k[i]) / 12)

        if renorm_every is not None:
            if i % renorm_every == 0:
                norm = np.max(np.abs(y_new[:i+1]))
                if norm > 0:
                    y_new[:i+1] /= norm

    return y_new

def k_hydrogen(r, E, l, scale=1):
    """
    The function k(r) for the hydrogen atom, given by:
        k(r) = -2 * (E + 1/r) + l*(l+1)/r^2
    used to solve:
        u''(r) = k(r) * u(r)
    where u(r) = r * R(r), and R(r) is the radial wavefunction.
    Scale for numerical stability. Will be killed with normalization anyway,
    but might help avoid r=0 divergence.
    """
    return -2 * scale * (E + 1/r) + l*(l+1)/r**2
    # return -(2/r - l*(l+1)/r**2 + E)  # BORKED: for some reason???

def seed_hydrogen(r, E, l):
    """
    Provides the initial conditions for the hydrogen atom problem.
    For small r, the solution behaves like r^(l+1). With that 
    we can seed the first two points of the integration as:

    u(r) ~ r^(l+1) => u(0) = 0 and u'(0) = (l+1)*r^l => u(h) ~ h^(l+1)
    """

    return r[0]**(l+1), r[1]**(l+1)

def seed_hydrogen_inward(r, E, l, start_idx=None):
    if start_idx is None:
        start_idx = -1
        
    kappa = np.sqrt(-2 * E)
    # print(f"Seeding inward with kappa={kappa:.6f}")  # DEBUG
    return np.exp(-kappa * r[start_idx]), np.exp(-kappa * r[start_idx-1])

def hydrogen_analytic(r, n, l):
    rho = 2* r / n
    prefactor = np.sqrt((2/n)**3 * factorial(n-l-1) / (2*n * factorial(n+l)))
    laguerre = np.exp(-rho/2) * rho**l * genlaguerre(n-l-1, 2*l+1)(rho)
    return r * prefactor * laguerre

def shoot(shoot_par, x, k_func, y_seed_func, renorm_every=None):
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
    renorm_every : int, optional
        Passed to numerov function.
        If set, renormalize the solution every `renorm_every` steps to prevent
        overflow. Preserves zero structure but not amplitude. Default is None (no renorm).

    Returns
    -------
    float
        The value of the solution at the end of the integration interval.
    """
    k = k_func(x, shoot_par)
    y0, y1 = y_seed_func(x, shoot_par)
    y = numerov(x, np.array([y0, y1]), -k, renorm_every=renorm_every)

    return y[-1]

def shoot_midpoint(shoot_par, x, k_func, y_seed_func, match_idx=None, inward_buffer=5.0):
    """
    Bidirectional shooting with log-derivative matching at a midpoint.

        discontinuity = (y_out'(x_m) / y_out(x_m)) - (y_in'(x_m) / y_in(x_m))
    
    Parameters
    ----------
    shoot_par : float
        The parameter to shoot with, e.g. the energy E in the hydrogen atom case.
    x : np.ndarray
        The spatial grid to integrate over.
    k_func : callable
        The function k(x, shoot_par) that defines the ODE to solve.
    y_seed_func : tuple of callable
        A tuple of two functions that provide the initial conditions for the outward and inward integrations, respectively:
        y_seed_outward(x, shoot_par) -> (y0, y1) near x[0]
        y_seed_inward(x, shoot_par) -> (y[-1], y[-2]) near x[-1]
    match_idx : int, optional
        Index of matching point. Defaults to classical turning point.
    inward_buffer : float, optional
        Multiple of the turning point distance to use for inward integration window.
        e.g. 5.0 means integrate inward from 5x the turning point radius.
    
    Returns
    -------
    float
        Log-derivative discontinuity at match point.
    """
    y_seed_outward, y_seed_inward = y_seed_func
    k = k_func(x, shoot_par)

    # Determine match point (classical turning point)
    if match_idx is None:
        sign_changes = np.where(np.diff(np.sign(k)))[0]
        match_idx = sign_changes[0] if len(sign_changes) > 0 else len(x) // 2

    # Pin inward start index to buffer * match point
    x_match = x[match_idx]
    x_inward_start = inward_buffer * x_match
    inward_start_idx = np.searchsorted(x, x_inward_start)
    inward_start_idx = min(inward_start_idx, len(x) - 1)

    # Outward integration
    y0, y1 = y_seed_outward(x, shoot_par)
    y_out = numerov(x[:match_idx+2], np.array([y0, y1]), k[:match_idx+2])

    # Inward Integration only over buffered region
    x_in = x[match_idx-1:inward_start_idx+1][::-1]
    k_in = k[match_idx-1:inward_start_idx+1][::-1]
    y_n, y_n1 = y_seed_inward(x, shoot_par, inward_start_idx)
    y_in_flip = numerov(x_in, np.array([y_n, y_n1]), k_in)
    y_in = y_in_flip[::-1]

    # Normalize at match point
    y_out_m = y_out[-1]
    y_in_m = y_in[0]

    # Log-derivative discontinuity
    h = (x[match_idx+1] - x[match_idx])
    dy_out = (y_out[-1] - y_out[-2]) / h
    dy_in = (y_in[1] - y_in[0]) / h
    
    return (dy_out / y_out_m) - (dy_in / y_in_m)

def scan_eigenvalues(x, k_func, y_seed_func, shoot_par_range, n_scan=500, shoot_func=None, **kwargs):
    """
    Scan for eigenvalues by shooting over a range of parameters and 
    looking for sign changes in the boundary value at the end of the integration interval.

    Parameters
    ----------
    x : np.ndarray
        The spatial grid to integrate over.
    k_func : callable
        The function k(x, shoot_par) that defines the ODE to solve.
    y_seed_func : callable or tuple of callables
        A function y_seed(x, shoot_par) or a tuple of two functions that provide 
        the initial conditions for the integration, given shoot_par and the shooting method used.
    shoot_par_range : tuple
        A tuple (shoot_par_min, shoot_par_max) defining the range of parameters to scan.
    n_scan : int, optional
        The number of parameter values to scan over, by default 500.
    shoot_func : callable, optional
        The shooting function to use, by default None which uses the standard shoot function.

    Returns
    -------
    brackets : list of tuples
        A list of tuples (shoot_par_left, shoot_par_right) where a sign change was detected,
        indicating a potential eigenvalue between shoot_par_left and shoot_par_right.
    """
    if shoot_func is None:
        shoot_func = shoot
    print("Performing initial scan for eigenvalues...")
    par_vals = np.linspace(shoot_par_range[0], shoot_par_range[1], n_scan)
    # residuals = np.array([shoot_func(par, x, k_func, y_seed_func, **kwargs) for par in par_vals])
    # Change to explicit loop for progress tracking
    residuals = np.zeros(n_scan)
    for i, par in enumerate(par_vals):
        print(f"Scanning parameter {i+1}/{n_scan}...", end="\r")
        residuals[i] = shoot_func(par, x, k_func, y_seed_func, **kwargs)
    brackets = []
    for i in range(len(par_vals) - 1):
        print(f"Checking pair {i}/{len(par_vals)-1}...", end="\r")
        if np.sign(residuals[i]) != np.sign(residuals[i+1]):
            brackets.append((par_vals[i], par_vals[i+1]))

    print("")
    return brackets

def find_eigenvalues(x, k_func, y_seed_func, shoot_par_range, n_scan=500, shoot_func=None, **kwargs):
    """
    Find eigenvalues by first scanning for sign changes and then using root finding to 
    refine the eigenvalue estimates.

    Parameters
    ----------
    x : np.ndarray
        The spatial grid to integrate over.
    k_func : callable
        The function k(x, shoot_par) that defines the ODE to solve.
    y_seed_func : callable or tuple of callables
        A function y_seed(x, shoot_par) or a tuple of two functions that provide 
        the initial conditions for the integration, given shoot_par and the shooting method used.
    shoot_par_range : tuple
        A tuple (shoot_par_min, shoot_par_max) defining the range of parameters to scan.
    n_scan : int, optional
        The number of parameter values to scan over for the initial scan, by default 500.
    shoot_func : callable, optional
        The shooting function to use, by default None which uses the standard shoot function.

    Returns
    -------
    eigenvalues : list of floats
        A list of refined eigenvalue estimates found in the specified range.
    """
    if shoot_func is None:
        shoot_func = shoot
    brackets = scan_eigenvalues(x, k_func, y_seed_func, shoot_par_range, n_scan, shoot_func)
    eigenvalues = []
    for left, right in brackets:
        print(f"Refining root between {left:.6f} and {right:.6f}...", end="\r")
        try:
            _shoot_func = lambda par: shoot_func(par, x, k_func, y_seed_func, **kwargs)
            eigenvalue = brentq(_shoot_func, left, right)
            eigenvalues.append(eigenvalue)
        except ValueError:
            # If brentq fails, we can skip this bracket
            print(f"Warning: brentq failed to find a root between {left} and {right}. Skipping this bracket.")
            continue
    print("\nEigenvalue finding complete.")
    return eigenvalues

def get_wavefunction(shoot_par, x, k_func, y_seed_func, blowup_threshold=None):
    """
    Get the wavefunction for a given parameter by integrating the ODE.
    Built in blowup handling: if the solution exceeds `blowup_threshold` times its peak value, 
    it is set to zero beyond that point. Built in normalization of the wavefunction using the trapezoidal rule.

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
    blowup_threshold : float, optional
        The multiple of the peak value at which to consider the solution as 
        blowing up, by default None, which means no blowup handling.

    Returns
    -------
    np.ndarray
        The wavefunction values at each point in x.
    """
    k = k_func(x, shoot_par)
    y0, y1 = y_seed_func(x, shoot_par)
    y = numerov(x, np.array([y0, y1]), -k)
    
    # Blowup Prevention
    if blowup_threshold is not None:
        peak = np.max(np.abs(y))
        blowup_idx = np.where(np.abs(y) > blowup_threshold * peak)[0]
        if len(blowup_idx) > 0:
            y[blowup_idx[0]:] = 0
    
    norm = np.trapezoid(y**2, x)
    return y / np.sqrt(norm)
