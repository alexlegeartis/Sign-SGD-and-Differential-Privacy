import numpy as np

import math
import scipy

code_version = "v23april"

def compute_A_alpha(alpha, q, sigma):  # for now we have only support of integer alpha
    asum = 0
    for k in range(alpha + 1):
        asum += scipy.special.comb(alpha, k) * (1 - q) ** (alpha - k) * q**k * math.exp((k*k-k)/(2 * sigma * sigma))
    return asum


def compute_eps_precise(alpha, q, sigma): # this is Rényi epsilon for 1 step
    return 1/(alpha - 1) * math.log(compute_A_alpha(alpha, q, sigma))


def find_min_eps_precise(alpha, T, epsilon, delta):
    return epsilon / T - math.log(1/delta)/((alpha - 1) * T)


def check_exp_parameters(alpha, q, sigma, T, epsilon, delta):
    try:
        return compute_eps_precise(alpha, q, sigma) < find_min_eps_precise(alpha, T, epsilon, delta)
    except OverflowError:
        print(f"Overflow on {alpha, q, sigma}")
        return False

from functools import lru_cache

@lru_cache(maxsize=100)
def find_min_sigma(q, T, epsilon, delta, alpha_min=2, alpha_max=20, sigma_min=0.5, sigma_max=3):
    # Define your ranges for alpha and sigma
    alphas = np.arange(alpha_min, alpha_max, 1)          # Integer values for alpha
    sigmas = np.linspace(sigma_min, sigma_max, 30)   # Real values for sigma

    opt_sigma = 100000
    opt_alpha = None
    # Fill the lists based on your boolean condition
    for alpha in alphas:
        for sigma in sigmas:
            if check_exp_parameters(alpha, q, sigma, T, epsilon, delta):
                if sigma < opt_sigma:
                    opt_sigma = sigma
                    opt_alpha = alpha

    return opt_sigma

import matplotlib.pyplot as plt
import os

def plot_valid_alpha_sigma(q, T, epsilon, delta, alpha_min=2, alpha_max=20, sigma_min=0.5, sigma_max=3):
    # Define your ranges for alpha and sigma
    alphas = np.arange(alpha_min, alpha_max, 1)          # Integer values for alpha
    sigmas = np.linspace(sigma_min, sigma_max, 30)   # Real values for sigma
    true_alphas, true_sigmas = [], []
    false_alphas, false_sigmas = [], []
    
    opt_sigma = 100000
    opt_alpha = None
    # Fill the lists based on your boolean condition
    for alpha in alphas:
        for sigma in sigmas:
            if check_exp_parameters(alpha, q, sigma, T, epsilon, delta):
                if sigma < opt_sigma or (sigma == opt_sigma and opt_alpha < alpha):
                    opt_sigma = sigma
                    opt_alpha = alpha
                true_alphas.append(alpha)
                true_sigmas.append(sigma)
            else:
                false_alphas.append(alpha)
                false_sigmas.append(sigma)
    print(f"Optimal sigma is {opt_sigma} for {opt_alpha=}")
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(true_alphas, true_sigmas, color='green', label='True', alpha=0.7)
    plt.scatter(false_alphas, false_sigmas, color='red', label='False', alpha=0.7)
    plt.scatter(opt_alpha, opt_sigma, color='purple', label='Optimal Point')

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\sigma$')
    plt.title('Alpha-Sigma Grid Results')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    folder_path = f"../figs/{code_version}/"
    os.makedirs(folder_path, exist_ok=True)
    os.path.join(folder_path, "valid_sigma.pdf")
    plt.show()

def dp_sign_by_sigma(g_t_m, sens_measure2, sigma):
    """
    Differentially Private Sign compressor (dp-sign) implementation.
    
    Parameters:
    - g_t_m (np.ndarray): Gradient vector at time t for model m.
    - epsilon (float): Differential privacy parameter epsilon.
    - delta (float): Differential privacy parameter delta.
    - Delta2 (float): Sensitivity parameter ∆2.
    
    Returns:
    - np.ndarray: DP-compressed sign vector of the same shape as g_t_m.
    """
    l2_norm = np.linalg.norm(g_t_m)
    if l2_norm > sens_measure2:
        g_t_m = (g_t_m / l2_norm) * sens_measure2

    noise_to_privatize = np.random.normal(loc=0.0, scale=sigma * sens_measure2, size=g_t_m.shape)
    g_t_noised = g_t_m + noise_to_privatize
    return np.sign(g_t_noised)


def dp_by_sigma(g_t_m, sens_measure2, sigma):
    """
    Differentially Private Sign compressor (dp-sign) implementation.
    
    Parameters:
    - g_t_m (np.ndarray): Gradient vector at time t for model m.
    - epsilon (float): Differential privacy parameter epsilon.
    - delta (float): Differential privacy parameter delta.
    - Delta2 (float): Sensitivity parameter ∆2.
    
    Returns:
    - np.ndarray: DP-compressed sign vector of the same shape as g_t_m.
    """
    l2_norm = np.linalg.norm(g_t_m)
    if l2_norm > sens_measure2:
        g_t_m = (g_t_m / l2_norm) * sens_measure2

    noise_to_privatize = np.random.normal(loc=0.0, scale=sigma * sens_measure2, size=g_t_m.shape)
    g_t_noised = g_t_m + noise_to_privatize
    return g_t_noised