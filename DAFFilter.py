
# -*- coding: UTF-8 -*-
'''
@Project ：DAFFILTER
@File    ：DAFFilter.py
@Author  ：Wu Dayu
@Date    ：2024/7/23 10:27
'''
import time

import matplotlib.pyplot as plt
import numpy as np
import math

class DAFFilter:

    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter_signal(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


if __name__ == '__main__':

    np.random.seed(1)
    # Parameters
    frames = 100
    start = 0
    end = 4 * np.pi
    scale = 0.5

    # The noisy signal
    t = np.linspace(start, end, frames)
    x = [np.sin(t),np.cos(t)]
    x_noisy = x + np.random.normal(scale=scale, size=len(t))
    # The filtered signal
    min_cutoff = 0.15
    beta = 0.01
    x_hat = np.zeros_like(x_noisy)
    x_hat[0] = x_noisy[0]

    one_euro_filter = DAFFilter(t[0], x_noisy[0], min_cutoff=min_cutoff, beta=beta)
    for i in range(0, len(t)):
        x_hat[i] = one_euro_filter.filter_signal(t[i], x_noisy[i])


