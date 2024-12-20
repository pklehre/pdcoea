#!/usr/bin/env python3

"""
Pairwise Dominance Co-Evolutionary Algorithm (PDCoEA)

Copyright 2024 Per Kristian Lehre

Reference:

Lehre, Per Kristian. Runtime Analysis of Competitive Co-evolutionary
Algorithms for Maximin Optimisation of a Bilinear Function.
Algorithmica 86, 2352--2392 (2024). https://doi.org/10.1007/s00453-024-01218-3

"""

import numpy as np
import signal
import argparse
import matplotlib.pyplot as plt

def mutate_pop_inplace(chi, pop):
    
    n = pop.shape[1]
    mask = np.random.rand(*pop.shape) < (chi / n)
    np.logical_xor(pop, mask, out=pop)

    
def pdcoea_generation(g, population_size, chi, preds, prey, preds_next, prey_next):
    """
    One generation of the PDCoEA.    
    """

    # Sample random indices
    i1 = np.random.randint(population_size, size=(population_size))
    i2 = np.random.randint(population_size, size=(population_size))
    j1 = np.random.randint(population_size, size=(population_size))
    j2 = np.random.randint(population_size, size=(population_size))    

    for i in range(population_size):
        x1 = preds[i1[i]]
        x2 = preds[i2[i]]
        y1 = prey[j1[i]]
        y2 = prey[j2[i]]

        # Check pairwise dominance criterion.
        if g(x1,y2) >= g(x1,y1) >= g(x2,y1):
            preds_next[i] = x1
            prey_next[i] = y1
        else:
            preds_next[i] = x2
            prey_next[i] = y2

    # Mutate all the selected individuals.
    mutate_pop_inplace(chi, preds_next)
    mutate_pop_inplace(chi, prey_next)
            
    return preds_next, prey_next, preds, prey
            
def create_pop(population_size, n):
    """
    Creates a population of binary individuals.

    Parameters:
    population_size (int): The number of individuals in the population (called lambda in the literature).
    n (int): The number of bits in each individual.

    Returns:
    numpy.ndarray: A population array of shape (lambda_, n) with binary values.
    """
    return np.random.randint(2, size=(population_size, n), dtype=bool)

def diagonal(x, y):
    """
    Payoff function for the diagonal game.
    """
    if np.sum(x) >= np.sum(y):
        return 1.0
    else:
        return 0.0

# Termination criterion for diagonal payoff function
def diagonal_nash(preds, prey):
    """
    Termination criterion for the diagonal payoff function.
    A pair of predator and prey populations are considered optimal
    if both contains an individual with all one-bits."""
    
    n_pred = preds.shape[1]
    n_prey = prey.shape[1]
    opt_pred = np.max(np.sum(preds,axis=1)) >= n_pred
    opt_prey = np.max(np.sum(prey,axis=1)) >= n_prey
    return opt_pred & opt_prey
    
def pdcoea(payoff, terminate, config):
    """
    Run the Pairwise Dominance Co-Evolutionary Algorithm (PDCoEA).

    Parameters:
    payoff (function): The payoff function to evaluate payoff of the predators and prey.
    terminate (function): The termination criterion function.
    config (argparse.Namespace): Configuration parameters.

    Returns:
    int: The number of payoff evaluations performed.
    """

    population_size = config.population_size
    chi = config.chi
    n = config.n

    preds, prey, preds_next, prey_next = create_pop(population_size, n), create_pop(population_size, n), create_pop(population_size, n), create_pop(population_size, n)

    payoff_evals = 0

    # Plotting setup
    if config.plot:
        plt.figure(figsize=(8, 6))
        plt.xlabel("Number of True in Predator")
        plt.ylabel("Number of True in Prey")
        plt.title("Predator vs. Prey Populations")
        plt.xlim(0, n)  # Set x-axis limits
        plt.ylim(0, n)  # Set y-axis limits  
        scatter = plt.scatter([], [], alpha=0.5)
  
  
    if config.plot:
        plt.fill_between([0, n], [0, n], [0, 0], color='red', alpha=0.1)       

    while (not terminate(preds, prey)) and (payoff_evals < config.max_payoff_evals):

        preds, prey, preds_next, prey_next = pdcoea_generation(payoff, population_size, chi, preds, prey, preds_next, prey_next)
        payoff_evals += 3 * population_size
        
        if config.plot:
            plt.fill_between([0, n], [0, n], [0, 0], color='red', alpha=0.1)       

            num_true_preds = np.sum(preds, axis=1)
            num_true_prey = np.sum(prey, axis=1)
            plt.scatter(num_true_preds, num_true_prey)

            plt.pause(0.1)
            plt.cla()

    return payoff_evals

def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--population_size", default=500, type=int, help="Population size")
    parser.add_argument("--n", default=100, type=int, help="Problem size")
    parser.add_argument("--chi", default=0.3, type=float, help="Mutation rate")
    parser.add_argument("--max_payoff_evals", default=1e7, type=int, help="Maximum number of payoff evaluations")
    parser.add_argument("--plot", action='store_true', help="Plot populations")       
    return parser.parse_args()
    
def main():
    """
    Main function to run PDCoEA. It parses command-line arguments,
    runs the algorithm, and prints the number of payoff evaluations.
    """
    config = parse_arguments()
    payoff_evals = pdcoea(diagonal, diagonal_nash, config)
    print(payoff_evals)
    
if __name__ == "__main__":
    main()
