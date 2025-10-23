"""
Monomer simulation functions for biophysics Monte Carlo simulation.

This module contains functions specific to monomer systems, including
system creation, energy calculations, and Monte Carlo moves.
"""

import numpy as np
import random
from numba import jit
from core_functions import findNeighbours, alpha_mono
from scipy.constants import Boltzmann

@jit(nopython=True)
def createSystemOfMonomers(N, M):
    """
    Create a random system of monomers on an N×N grid.
    
    Parameters:
    -----------
    N : int
        Grid size (N×N)
    M : int
        Half the total number of monomers (total will be 2*M)
        
    Returns:
    --------
    grid : ndarray
        N×N grid with monomers placed randomly
    locations : ndarray
        Coordinates of all monomers in the grid
    """
    grid = np.zeros((N,N))
    
    # Create array with values from -M to M excluding 0
    distinctValues = np.arange(-M, M + 1)
    newDistinctValues = np.delete(distinctValues, M)  # Remove 0
    np.random.shuffle(newDistinctValues)
    
    count = 0
    
    while count < 2 * M:
        x_value = random.randint(0, N-1)
        y_value = random.randint(0, N-1)
        
        if grid[y_value][x_value] == 0:
            grid[y_value][x_value] = newDistinctValues[count]
            count += 1
    
    locations = np.argwhere(grid)
    return grid, locations

@jit(nopython=True)
def totalEnergy(grid):
    """
    Calculate total energy of a monomer system using electrostatic interactions.
    
    Parameters:
    -----------
    grid : ndarray
        N×N grid containing monomers
        
    Returns:
    --------
    float
        Total energy of the system in Joules
    """
    N = len(grid)
    energy = 0
    
    locations = np.argwhere(grid != 0)
    
    for i in range(len(locations)):
        neighbours = findNeighbours(N, locations[i][0], locations[i][1])
        
        for j in range(len(neighbours)):
            if grid[int(neighbours[j][0])][int(neighbours[j][1])] != 0:
                energy += np.sign(grid[int(locations[i][0])][int(locations[i][1])] * 
                                grid[int(neighbours[j][0])][int(neighbours[j][1])])
    
    return energy * alpha_mono / 2

@jit(nopython=True)
def move_monomer_fast(grid, monomer_value, direction):
    """
    Move a specific monomer in a given direction.
    
    Parameters:
    -----------
    grid : ndarray
        N×N grid containing monomers
    monomer_value : int
        Value identifying the monomer to move
    direction : int
        Direction to move (0=right, 1=left, 2=down, 3=up)
    """
    monomer_coordinates = np.argwhere(grid == monomer_value)[0]
    
    N = grid.shape[0]
    array_of_all_neighbor_coordinates = findNeighbours(N=N, y=monomer_coordinates[0], 
                                                      x=monomer_coordinates[1])
    
    desired_neigbor_coordinate = array_of_all_neighbor_coordinates[direction]
    
    if grid[int(desired_neigbor_coordinate[0])][int(desired_neigbor_coordinate[1])] == 0:
        grid[int(desired_neigbor_coordinate[0])][int(desired_neigbor_coordinate[1])] = monomer_value
        grid[int(monomer_coordinates[0])][int(monomer_coordinates[1])] = 0

@jit(nopython=True)
def MC(N_s, N, M, T, initGrid=None, initial=True):
    """
    Run Monte Carlo simulation for monomer system using Metropolis algorithm.
    
    Parameters:
    -----------
    N_s : int
        Number of Monte Carlo steps
    N : int
        Grid size (N×N)
    M : int
        Half the number of monomers
    T : float
        Temperature in Kelvin
    initGrid : ndarray, optional
        Initial grid configuration
    initial : bool
        Whether to create new system or continue from initGrid
        
    Returns:
    --------
    grid : ndarray
        Final grid configuration
    epsilon : ndarray
        Energy evolution over time
    initialGrid : ndarray
        Initial grid configuration
    typeMove : ndarray
        Count of different move types [favorable, random, rejected]
    """
    if initial == True:
        grid, locations = createSystemOfMonomers(N,M)
    else:
        grid = initGrid
        locations = np.argwhere(grid)
        
    initialGrid = np.copy(grid)
    typeMove = np.zeros(3)
    epsilon = np.zeros(N_s)

    EnergyOfState = totalEnergy(grid)
    epsilon[0] = EnergyOfState

    for i in range(N_s):
        oldGrid = grid.copy()
        
        choosenMonomer = 0
        while choosenMonomer == 0:
            choosenMonomer = np.random.randint(-M, M - 1)
                
        k = np.random.randint(0,3)
        move_monomer_fast(grid, choosenMonomer, k)

        EnergyOfState_new = totalEnergy(grid)

        if EnergyOfState_new < EnergyOfState:
            EnergyOfState = EnergyOfState_new
            typeMove[0] += 1
        elif random.random() < np.exp(- (1 / (Boltzmann * T)) * (EnergyOfState_new - EnergyOfState)):  
            EnergyOfState = EnergyOfState_new
            typeMove[1] += 1
        else:
            grid = oldGrid.copy()
            typeMove[2] += 1
                    
        epsilon[i] = EnergyOfState
    
    return grid, epsilon, initialGrid, typeMove