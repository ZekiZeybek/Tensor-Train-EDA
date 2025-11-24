import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from src.tt_eda.MPS_TN import MPS
from src.tt_eda.MPS_Opt import Opt_MPS
from scipy.optimize import minimize

def func_build_ackley(d, n, domain_a=-32.768, domain_b=32.768):
    a = domain_a
    b = domain_b
    par_a = 20.
    par_b = 0.2
    par_c = 2. * np.pi
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        sum_sq = np.sum(X**2)
        term1 = -par_a * np.exp(-par_b * np.sqrt(sum_sq / d))
        term2 = -np.exp(np.sum(np.cos(par_c * X)) / d)
        y = term1 + term2 + par_a + np.exp(1.)
        return y
    return func, domain_a, domain_b

def func_build_alpine(d, n, domain_a=-10., domain_b=10.):
    a = domain_a
    b = domain_b
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        term = X * np.sin(X) + 0.1 * X
        y = np.sum(np.abs(term))
        return y
    return func, domain_a, domain_b

def func_build_griewank(d, n, domain_a=-600., domain_b=600.):
    a = domain_a
    b = domain_b
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        term1 = np.sum(X**2) / 4000.0
        indices = np.arange(1, d + 1)
        term2 = np.prod(np.cos(X / np.sqrt(indices)))
        y = term1 - term2 + 1.0
        return y
    return func, domain_a, domain_b

def func_build_michalewicz(d, n, domain_a=0., domain_b=np.pi, m=10.):
    a = domain_a
    b = domain_b
    par_m = m
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        indices = np.arange(1, d + 1)
        term_sin1 = np.sin(X)
        term_sin2_base = np.sin(indices * X**2 / np.pi)
        term_sin2 = (np.maximum(np.abs(term_sin2_base), 1e-8))**(2 * par_m)
        y = -np.sum(term_sin1 * term_sin2)
        return y
    return func, domain_a, domain_b

def func_build_rastrigin(d, n, domain_a=-5.12, domain_b=5.12, A=10.):
    a = domain_a
    b = domain_b
    par_A = A
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        term1 = par_A * d
        term2 = np.sum(X**2 - par_A * np.cos(2 * np.pi * X))
        y = term1 + term2
        return y
    return func, domain_a, domain_b

def func_build_schwefel(d, n, domain_a=-500., domain_b=500.):
    a = domain_a
    b = domain_b
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        term1 = 418.9829 * d
        term2 = np.sum(X * np.sin(np.sqrt(np.abs(X))))
        y = term1 - term2
        return y
    return func, domain_a, domain_b



def func_build_alpine(d, n, domain_a=-10., domain_b=10.):
    a = domain_a
    b = domain_b
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        term = X * np.sin(X) + 0.1 * X
        y = np.sum(np.abs(term))
        return y
    return func, domain_a, domain_b




