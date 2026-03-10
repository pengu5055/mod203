"""
Contains functions used throught the project.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import factorial, genlaguerre, k0, k1
import ray

def rerr(exact, approx):
    """Relative error between exact and approximate solutions."""
    return np.abs((exact - approx) / exact)

def aerr(exact, approx, eps=1e-16):
    """Absolute error between exact and approximate solutions."""
    return np.abs(exact - approx) + eps

def init_ray():
    """
    Initialize Ray.
    """
    port = 8265
    context = ray.init(address="auto",
                       runtime_env={"working_dir": "./Code", "pip": ["numpy", "scipy"]},
                       dashboard_port=port,
                       )
    url = context.dashboard_url if context.dashboard_url else "localhost"
    print("Ray initialized")
    print(f"Dashboard: {url}")
    return context

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

def shoot(shoot_par, x, k_func, y_seed_func, renorm_every=None, negate_k=True):
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
    negate_k : bool, optional
        Whether to negate the k function when passing to numerov. This is because the standard form
        of the ODE for Numerov is y'' = k(x) * y, but in our case we often have it in the form y'' = -k(x) * y.

    Returns
    -------
    float
        The value of the solution at the end of the integration interval.
    """
    k = k_func(x, shoot_par)
    y0, y1 = y_seed_func(x, shoot_par)
    y = numerov(x, np.array([y0, y1]), -k if negate_k else k, renorm_every=renorm_every)

    return y[-1]

def shoot_midpoint(shoot_par, x, k_func, y_seed_func, match_idx=None, inward_buffer=5.0, negate_k=True):
    """
    Bidirectional shooting with RMS normalized Wronskian condition residual.

        residual = y_out(match_idx) * dy_in(match_idx) - y_in(match_idx) * dy_out(match_idx)
    
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
    negate_k : bool, optional
        Whether to negate the k function when passing to numerov. This is because the standard form
        of the ODE for Numerov is y'' = k(x) * y, but in our case we often have it in the form y'' = -k(x) * y.
    
    Returns
    -------
    float
        Log-derivative discontinuity at match point.
    """
    y_seed_outward, y_seed_inward = y_seed_func
    k = -k_func(x, shoot_par) if negate_k else k_func(x, shoot_par)

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

    if abs(y_out_m) < 1e-10 or abs(y_in_m) < 1e-10:
        return np.nan # Try kill match point divergence
    
    # return (dy_out / y_out_m) - (dy_in / y_in_m)
    # DEBUG:
    # Return Wronskian condition instead
    # Normalize at match point before Wronskian
    # y_out_norm = y_out / y_out_m
    # y_in_norm = y_in / y_in_m
    # dy_out_norm = dy_out / y_out_m
    # dy_in_norm = dy_in / y_in_m
    # 
    # return y_out_norm[-1] * dy_in_norm - y_in_norm[0] * dy_out_norm

    # Turns out RMS normalization works much much better
    y_out_rms = np.sqrt(np.mean(y_out**2))
    y_in_rms = np.sqrt(np.mean(y_in**2))

    y_out_n = y_out / y_out_rms
    y_in_n = y_in / y_in_rms

    dy_out_n = (y_out_n[-1] - y_out_n[-2]) / h
    dy_in_n = (y_in_n[1] - y_in_n[0]) / h

    return y_out_n[-1] * dy_in_n - y_in_n[0] * dy_out_n

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
        if np.isnan(residuals[i]) or np.isnan(residuals[i+1]):
            continue  # Skip pairs where either value is NaN
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

def get_hydrogen_wavefunctions(eigenvalues, l, idx=None, n_eval=10000, multiple=5, r_min=1e-3):
    """
    Get the hydrogen wavefunctions corresponding to the given eigenvalues by integrating the ODE with those eigenvalues as parameters.
    The integration range is determined by the formula r_max = multiple * n^2, where n is the index of the eigenvalue (starting from 1). 
    This is based on the fact that the radial extent of the hydrogenic wavefunctions scales roughly with n^2. The `multiple` parameter 
    allows for some extra space.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues for which to compute the wavefunctions.
    l : int
        The angular momentum quantum number to use in the k function and seed function.
    idx : list of int, optional
        The indices of the eigenvalues to compute wavefunctions for. If None, compute for all eigenvalues, by default None.
    n_eval : int, optional
        The number of spatial points to evaluate the wavefunction on, by default 10000.
    multiple : float or list of float, optional
        The multiple of n^2 to use for the maximum radius, by default 5. If a single float is provided, it will be used for all
        eigenvalues. If a list is provided, it should have the same length as the number of eigenvalues and will specify the multiple for each one.
    r_min : float, optional
        The minimum radius to start integration from, by default 1e-3.

    Returns
    -------
    x_ranges : list of np.ndarray
        The spatial grids used for each wavefunction.
    wavefunctions : list of np.ndarray
        The computed wavefunctions corresponding to the given eigenvalues.
    """
    if idx is None:
        idx = range(len(eigenvalues))

    if np.isscalar(multiple):
        multiples = [multiple] * len(eigenvalues)
    else:
        multiples = multiple
        if len(multiples) != len(idx):
            raise ValueError("Length of multiples list must match number of eigenvalues/idx.")

    x_ranges = []
    wavefunctions = []
    k_func = lambda r, E: k_hydrogen(r, E, l=l)
    seed_func_out = lambda r, E: seed_hydrogen(r, E, l=l)

    for i in idx:
        print(f"Getting wavefunction {i+1} with E={eigenvalues[i]:.6f}...")
        n = i + 1
        r_max = multiples[i] * n**2
        r_range = np.linspace(r_min, r_max, n_eval)
        
        wave = get_wf(eigenvalues[i], r_range, k_func, seed_func_out)

        x_ranges.append(r_range)
        wavefunctions.append(wave)

    return x_ranges, wavefunctions

def get_fiber_wavefunctions(eigenvalues, k, idx=None, n_eval=10000, multiple=5, x_min=1e-3):
    """
    BROKEN WITH get_wf_midpoint! Using get_wf does not really work as solution outside of core does NOT decay but oscillates.
    Get the fiber wavefunctions corresponding to the given eigenvalues by integrating the ODE with those eigenvalues as parameters.
    The integration range is determined by wavefunction decay, which should be roughly exp(-gamma * x) where gamma ~ sqrt(lambda^2 - k^2).
    We can estimate the decay length as 1/gamma, and set the maximum radius to be some multiple of that decay length to 
    ensure we capture the wavefunction adequately.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues for which to compute the wavefunctions.
    k : float
        The angular momentum quantum number to use in the k function and seed function.
    idx : list of int, optional
        The indices of the eigenvalues to compute wavefunctions for. If None, compute for all eigenvalues, by default None.
    n_eval : int, optional
        The number of spatial points to evaluate the wavefunction on, by default 10000.
    multiple : float or list of float, optional
        The multiple of the decay length to use for the maximum radius, by default 5. If a single float is provided, it will be used for all
        eigenvalues. If a list is provided, it should have the same length as the number of eigenvalues and will specify the multiple for each one.
    x_min : float, optional
        The minimum radius to start integration from, by default 1e-3.

    Returns
    -------
    x_ranges : list of np.ndarray
        The spatial grids used for each wavefunction.
    wavefunctions : list of np.ndarray
        The computed wavefunctions corresponding to the given eigenvalues.
    """
    if idx is None:
        idx = range(len(eigenvalues))

    if np.isscalar(multiple):
        multiples = [multiple] * len(eigenvalues)
    else:
        multiples = multiple
        if len(multiples) != len(idx):
            raise ValueError("Length of multiples list must match number of eigenvalues/idx.")

    x_ranges = []
    wavefunctions = []
    k_func = lambda r, lam: k_fiber(r, lam, k=k)
    seed_func_out = lambda r, lam: seed_fiber(r, lam, k=k)
    seed_func_in = lambda r, lam, start_idx: seed_fiber_inward(r, lam, k, start_idx)

    for i in idx:
        print(f"Getting wavefunction {i+1} with E={eigenvalues[i]:.6f}...")
        x_max = 1 + multiples[i] / np.sqrt(eigenvalues[i]**2 - k**2)
        x_range = np.linspace(x_min, x_max, n_eval)
        core_idx = np.searchsorted(x_range, 1.0)
        
        # wave = get_wf_midpoint(eigenvalues[i], x_range, k_func, (seed_func_out, seed_func_in), match_idx=core_idx, inward_buffer=5.0, negate_k=True)
        # wave = get_wf(eigenvalues[i], x_range, k_func, seed_func_out, blowup_threshold=None, negate_k=True)
        wave = get_wf_fiber(eigenvalues[i], x_range, k_func, seed_func_out, k_val=k, core_idx=core_idx)

        x_ranges.append(x_range)
        wavefunctions.append(wave)

    return x_ranges, wavefunctions

def get_wf(shoot_par, x, k_func, y_seed_func, blowup_threshold=None, negate_k=True):
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
    negate_k : bool, optional
        Whether to negate the k function when passing to numerov. This is because the standard form
        of the ODE for Numerov is y'' = k(x) * y, but in our case we often have it in the form y'' = -k(x) * y.

    Returns
    -------
    np.ndarray
        The wavefunction values at each point in x.
    """
    k = k_func(x, shoot_par)
    y0, y1 = y_seed_func(x, shoot_par)
    y = numerov(x, np.array([y0, y1]), -k if negate_k else k)
    
    # Blowup Prevention
    if blowup_threshold is not None:
        peak = np.max(np.abs(y))
        blowup_idx = np.where(np.abs(y) > blowup_threshold * peak)[0]
        if len(blowup_idx) > 0:
            y[blowup_idx[0]:] = 0
    
    norm = np.trapezoid(y**2, x)
    return y / np.sqrt(norm)

def get_wf_midpoint(shoot_par, x, k_func, y_seed_func, match_idx=None, inward_buffer=1.0, negate_k=True):
    """
    BROKEN! 
    """
    y_seed_outward, y_seed_inward = y_seed_func
    k = -k_func(x, shoot_par) if negate_k else k_func(x, shoot_par)

    # Inward start
    x_match = x[match_idx]
    x_inward_start = inward_buffer * x_match
    inward_start_idx = min(np.searchsorted(x, x_inward_start), len(x) - 1)

    # Outward integration
    y0, y1 = y_seed_outward(x, shoot_par)
    y_out = numerov(x[:match_idx+2], np.array([y0, y1]), k[:match_idx+2])

    # Inward integration
    x_in = x[match_idx:inward_start_idx+1][::-1]
    k_in = k[match_idx:inward_start_idx+1][::-1]
    y_n, y_n1 = y_seed_inward(x, shoot_par, inward_start_idx)
    y_in_flip = numerov(x_in, np.array([y_n, y_n1]), k_in)
    y_in = y_in_flip[::-1]

    # Normalize at match point
    y_out = y_out * (y_in[0] / y_out[-1])

    # Stitch together
    y_full = np.zeros_like(x)
    y_full[:match_idx+1] = y_out[:-1]
    y_full[match_idx:inward_start_idx+1] = y_in

    # and 0 beyond inward start
    y_full[inward_start_idx+1:] = 0

    # Normalize the full wavefunction
    norm = np.trapezoid(y_full**2, x)
    return y_full / np.sqrt(norm)

def get_wf_fiber(lam, x, k_func, seed_out, k_val, core_idx):
    """
    I'm desperate here.. x > 1 should analytically be K0. So lets 
    pretend that numverov only applies in the core, and then match K0 at the boundary. 
    """
    # Numerical solution in core
    k = -k_func(x, lam)
    y0, y1 = seed_out(x, lam)
    y_out = numerov(x[:core_idx+1], np.array([y0, y1]), -k[:core_idx+1])

    # Match K0 at core boundary by derivative
    gamma = np.sqrt(lam**2 - k_val**2)
    h = x[1] - x[0]
    dy_out = (y_out[-1] - y_out[-2]) / h
    dk0 = -gamma * k1(gamma * x[core_idx])
    scale = dy_out / dk0
    
    # Full solution
    y_full = np.zeros(len(x))
    y_full[:core_idx+1] = y_out
    y_full[core_idx+1:] = scale * k0(gamma * x[core_idx+1:])

    # Ensure continuity of sign at boundary
    if y_full[core_idx] * y_full[core_idx+1] < 0:
        y_full[core_idx+1:] *= -1

    # Rescale y_out to match value at boundary
    y_full[:core_idx+1] *= (y_full[core_idx+1] / y_out[-1])
    
    norm = np.trapezoid(y_full**2, x)
    return y_full / np.sqrt(norm)

def n_fiber(x):
    return np.where(x < 1, 2 - 0.5*x**2, 1.0)

def k_fiber(x, lam, k):
    return 1/(4*x**2) + n_fiber(x)**2 * k**2 - lam**2

def seed_fiber(x, lam, k):
    u2 = n_fiber(x[0])**2 * k**2 - lam**2
    y0 = x[0]**(0.5) * (1 + u2 * x[0]**2 / 8)
    y1 = x[1]**(0.5) * (1 + u2 * x[1]**2 / 8)
    return y0, y1

def seed_fiber_inward(x, lam, k, start_idx):
    gamma = np.sqrt(lam**2 - k**2)
    return np.exp(-gamma * x[start_idx]), np.exp(-gamma * x[start_idx-1])