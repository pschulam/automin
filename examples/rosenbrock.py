import logging

import autograd.numpy as np
import autograd.numpy.random as rng

from automin import minimize


def main():
    rng.seed(0)

    def f(x, a=1.0, b=100.0):
        x1, x2 = x
        return (a - x1)**2 + b*(x2 - x1**2)**2

    x = rng.normal(size=2)

    x_opt = minimize(f, x, method='CG')

    print(x_opt)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
