"""Implementation of scaled conjugate gradient optimization.

This code is adapted from Ian Nabney's Netlab toolbox:
http://www.mathworks.com/matlabcentral/fileexchange/2654-netlab/

"""
import logging

import numpy as np
import scipy.linalg as la


_EPS = np.finfo(np.float).eps


def scg(func, x, grad, options=None, callback=None, *args, **kwargs):
    """Find a local minimum of a function.

    Parameters
    ----------
    f : Function to be minimized.
    x : Initial vector of parameters.
    g : Function to compute the gradient.
    options : Options controlling termination criteria (see Notes).
    callback : Function to call after each iteration.

    Returns
    -------
    A dict recording information from the optimization procedure.

    Notes
    -----

    callback function should have signature (num_iter, f_cur, g_cur, x).

    """
    if options is None:
        options = default_options()

    num_f_eval = 0
    num_g_eval = 0

    def f(x):
        nonlocal num_f_eval
        y = func(x, *args, **kwargs)
        num_f_eval += 1
        return y

    def g(x):
        nonlocal num_g_eval
        y = grad(x, *args, **kwargs)
        num_g_eval += 1
        return y

    sigma_0 = 1.0e-4

    f_old = f(x)
    f_cur = f_old

    g_old = g(x)
    g_cur = g_old

    search_dir = -g_cur

    success = True   # Force calculation of directional derivatives in first iter.
    num_success = 0

    beta = 1.0          # Initial scale parameter.
    beta_min = 1.0e-15  # Lower bound on scale parameter.
    beta_max = 1.0e100  # Upper bound on scale parameter.

    num_iter = 0

    while True:

        if success:

            # Calculate first and second directional derivatives.

            dir_deriv = search_dir @ g_cur

            if dir_deriv >= 0.0:
                search_dir = -g_cur
                dir_deriv = search_dir @ g_cur

            dir_mag = search_dir @ search_dir
            if dir_mag < _EPS:
                break

            sigma = sigma_0 / np.sqrt(dir_mag)
            x_plus = x + sigma * search_dir
            g_plus = g(x_plus)
            theta = (search_dir @ (g_plus - g_cur)) / sigma

        # Increase effective curvature and evaluate step size (alpha).

        delta = theta + beta * dir_mag

        if delta <= 0.0:
            delta = beta * dir_mag
            beta = beta - theta / dir_mag

        alpha = -dir_deriv / delta

        # Calculate the comparison ratio.

        x_new = x + alpha * search_dir
        f_new = f(x_new)

        comp_ratio = 2*(f_new - f_old)/(alpha * dir_deriv)

        if comp_ratio >= 0.0:
            success = True
            num_success += 1
            x = x_new
            f_cur = f_new
        else:
            success = False
            f_cur = f_old

        if success:

            d_small_enough = la.norm(alpha * search_dir, np.inf) < options['d_tol']
            f_small_enough = np.abs(f_new - f_old) < options['f_tol']

            if d_small_enough and f_small_enough:
                break

            else:
                f_old = f_new
                g_old = g_cur
                g_cur = g(x)

                if la.norm(g_cur) == 0.0:
                    break

        # Adjust scale (beta) according to the comparison ratio.

        if comp_ratio < 0.25:
            beta = min(4.0 * beta, beta_max)

        if comp_ratio > 0.75:
            beta = max(0.5 * beta, beta_min)

        # Update search direction using Polak-Ribiere.
        # If we've taken len(x) steps, then restart at negative gradient.

        if num_success == len(x):
            search_dir = -g_cur
            num_success = 0

        else:
            if success:
                gamma = (g_old - g_cur) @ g_cur / dir_deriv
                search_dir = gamma * search_dir - g_cur

        num_iter += 1

        if callback:
            callback(num_iter, f_cur, g_cur, x)

        if num_iter >= options['max_iter']:
            break

    return {'f': f_cur, 'x': x, 'g': g_cur}


def default_options():
    """Return a dict containing default SCG options."""
    options = {'d_tol': 1e-6, 'f_tol': 1e-6, 'max_iter': 200}
    return options
