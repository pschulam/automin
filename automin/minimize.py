import logging

import autograd

from scipy import optimize
from autograd.util import flatten_func


def minimize(func, init_params, method='BFGS'):
    """Minimize a function with respect to its inputs."""

    grad = autograd.grad(func)

    flattened_func, _, _ = flatten_func(func, init_params)
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    callback = _make_callback(flattened_func)

    result = optimize.minimize(
        flattened_func, x, jac=flattened_grad, method=method, callback=callback)

    return unflatten(result['x'])


def _make_callback(func):
    """Create a default callback function."""
    def callback(x):
        """Print current function value to logging.INFO."""
        f = func(x)
        m = 'f={:.4f}'.format(f)
        logging.info(m)
    return callback
