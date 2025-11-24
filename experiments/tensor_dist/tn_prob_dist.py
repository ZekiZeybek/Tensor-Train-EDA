
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from src.tt_eda.MPS_TN import MPS
from src.tt_eda.MPS_Opt import Opt_MPS
from typing import List, Dict



def config_to_index(config):
    L = config.shape[0]
    weights = 2 ** np.arange(L-1, -1, -1)
    return int(np.dot(config, weights))

def KL_div(P,Q):
    return np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))

def chisqr_PQ(P,Q,num_samples):
    _, p_value = stats.chisquare(Q * num_samples, P * num_samples)
    return p_value


def hellinger_distance(P, Q):
    """
    Hellinger distance ranging between 0 and 1, with 0 indicating that the distributions are identical and 1 meaning they are maximally different.
    """
    return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2)) / np.sqrt(2)

def sample_uniform_configs_bits(L, num_samples):
    return np.random.randint(0, 2, size=(num_samples, L))

def P_empirical(samples, L):
    indices = np.array([config_to_index(config) for config in samples])
    counts = np.bincount(indices, minlength=2**L)
    freq = counts / counts.sum()
    return np.array(freq)

def test_unif_pdf_MPS(self, L=4, num_samples=1000):

    mps = MPS.uniform_pdf_MPS(length=L,num_phys_leg=1,const_chi=1,phys_leg_dim=2)
    o = Opt_MPS(MPS=mps,tot_num_samples=2)

    P_theoretical = np.array([1/(2**L)]*(2**L))
    samples = mps.seq_sample_from(num_samples=num_samples,seed=2321)
    # samples = o.sample_uniform_configs_bits(L=L,num_samples=num_samples)

    indices = np.array([o.config_to_index(config) for config in samples])
    counts = np.bincount(indices, minlength=2**L)
    freq = counts / counts.sum()
    P_empirical = np.array(freq)
    
    
    kl_div = o.KL_div(P=P_theoretical,Q=P_empirical)
    p_value = o.chisqr_PQ(P=P_theoretical,Q=P_empirical,num_samples=num_samples)
    hell_dist = o.hellinger_distance(P_theoretical,P_empirical)

    print('KL div.: ', kl_div,' Chi-sqr p-val.: ', p_value, ' Helli. dist.: ', hell_dist)
    
    x = np.arange(2**L)
    plt.bar(x - 0.2, P_theoretical, 0.4, label="Theoretical")
    plt.bar(x + 0.2, P_empirical, 0.4, label="Empirical")
    plt.xticks(x, [format(i, f'0{L}b') for i in range(2**L)], rotation=90)
    plt.xlabel("Configuration")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


