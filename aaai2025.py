#!/usr/bin/env python3

"""

Empirical Runtime Analysis of PDCoEA on Diagonal

Copyright 2024 Per Kristian Lehre

Reference:

Per Kristian Lehre and Shishen Lin. Towards Runtime Analysis of
Population-Based Co-evolutionary Algorithms on Sparse Binary Zero-Sum
Games. To appear in Proceedings of AAAI 2025.

"""

import pdcoea as pd

import numpy as np
import argparse
import matplotlib.pyplot as plt

def generate_heatmap_data(n, popsize_range, chi_range, num_trials, max_payoff_evals):
    """
    Generates data for the heatmap by running pdcoea with different lambda and chi values.

    Args:
        n: Problem size.
        popsize_range: Range of population sizes to test.
        chi_range: Range of chi values to test.
        max_payoff_evals: Maximum number of payoff evaluations.

    Returns:
        A 2D NumPy array containing the number of iterations for each (lambda, chi) combination.
    """
    iterations_data = np.zeros((len(popsize_range), len(chi_range)))

    for i, population_size in enumerate(popsize_range):
        for j, chi in enumerate(chi_range):
            config = argparse.Namespace(population_size=population_size, n=n, chi=chi, max_payoff_evals=max_payoff_evals, plot=False)
            for _ in range(num_trials):
                payoff_evals = pd.pdcoea(pd.diagonal, pd.diagonal_nash, config)
                iterations_data[i, j] += payoff_evals / num_trials

    return iterations_data

def plot_heatmap(iterations_data, popsize_range, chi_range):
    """
    Plots a heatmap of the iterations data.

    Args:
        iterations_data: 2D NumPy array of iterations data.
        popsize_range: Range of population sizes.
        chi_range: Range of chi values.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(iterations_data, cmap='inferno', interpolation='nearest', origin='upper', 
               extent=[chi_range[0], chi_range[-1], popsize_range[0], popsize_range[-1]],
               aspect='auto')
    plt.colorbar(label='Number of Payoff Evaluations')
    plt.xlabel("chi (Mutation Rate)")
    plt.ylabel("lambda (Population Size)")
    plt.title("PDCoEA Iterations Heatmap")
    plt.show()
    

def main():

    # Configuration for the heatmap
    n = 100
    popsize_range = np.arange(50, 100, 10)  # Example range for lambda
    chi_range = np.arange(0.1, 0.6, 0.1)   # Example range for chi
    max_payoff_evals = 1e8
    num_trials = 10

    # Generate the heatmap data
    iterations_data = generate_heatmap_data(n, popsize_range, chi_range, num_trials, max_payoff_evals)

    # Plot the heatmap
    plot_heatmap(iterations_data, popsize_range, chi_range)

if __name__ == "__main__":
    main()
