import logging

import autograd.numpy as np
import autograd.numpy.random as rng

from automin import minimize


def main():
    rng.seed(0)

    D = 10

    def f(x): return np.sum(x**2)
    x = rng.normal(size=D)

    x_opt = minimize(f, x)

    print(x_opt)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
