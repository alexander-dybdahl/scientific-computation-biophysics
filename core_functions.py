"""
Core functions for biophysics Monte Carlo simulation.

This module contains fundamental functions used across monomer and polymer simulations.
"""

import numpy as np
import random
from numba import jit
from scipy.constants import Boltzmann, epsilon_0, elementary_charge

# Physical constants
epsilon_r = 78
a_mono = (23.e-6)**2
a_poly = (91.e-6)**2
alpha_mono = elementary_charge**2 /(4 * np.pi * epsilon_0 * epsilon_r * a_mono)
alpha_poly = alpha_mono * a_mono / a_poly

@jit(nopython=True)
def findNeighbours(N, y, x):
    """
    Find the four nearest neighbors of a grid position with periodic boundary conditions.
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    y, x : int
        Coordinates of the position
        
    Returns:
    --------
    neighbours : ndarray
        Array of shape (4, 2) containing neighbor coordinates [y, x]
    """
    neighbours = np.zeros((4,2))
    
    neighbours[0] = [int(y), int((x + 1) % N)]  # right
    neighbours[1] = [int(y), int((x - 1) % N)]  # left
    neighbours[2] = [int((y + 1) % N), int(x)]  # down
    neighbours[3] = [int((y - 1) % N), int(x)]  # up
    
    return neighbours

@jit(nopython=True)
def inArray(arr, obj):
    """
    Check whether an array with length two is contained in a 2D array.
    
    Parameters:
    -----------
    arr : ndarray
        2D array to search in
    obj : ndarray
        Object to search for
        
    Returns:
    --------
    bool
        True if object is found in array, False otherwise
    """
    inArr = False
    for pos in arr:
        if np.array_equal(pos, obj):
            inArr = True
    return inArr

def checkMonomers(grid, M):
    """
    Check if system contains exactly 2*M monomers.
    
    Parameters:
    -----------
    grid : ndarray
        The grid containing monomers
    M : int
        Half the expected number of monomers
    """
    nonZeroGrid = np.argwhere(grid)
    numberOfMonomers = len(list(nonZeroGrid))

    if numberOfMonomers == 2*M:
        print("Has 2M monomers in system")
    elif numberOfMonomers < 2*M:
        print("Has less than 2M monomers in system")
    else:
        print("Has more than 2M monomers in system")