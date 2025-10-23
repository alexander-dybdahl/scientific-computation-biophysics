"""
Analysis tools for biophysics Monte Carlo simulation.

This module contains functions for analyzing simulation results,
including clustering analysis and statistical measurements.
"""

import numpy as np
from numba import jit
from statistics import mean
from core_functions import findNeighbours

@jit(nopython=True)
def cluster(clusterGrid, N, k, M):
    """
    Perform cluster analysis on a grid using recursive algorithm.
    
    Parameters:
    -----------
    clusterGrid : ndarray
        Grid to analyze for clusters
    N : int
        Grid size
    k : int
        Recursion step counter
    M : int
        Maximum monomer value
        
    Returns:
    --------
    clusterGrid : ndarray
        Grid with cluster labels
    numberOfClusters : int
        Total number of clusters found
    sizes : ndarray
        Array containing size of each cluster
    """
    clusterGrid = clusterGrid.copy()
    locationsOfMonomers = np.argwhere(clusterGrid)
    
    if k < len(locationsOfMonomers):
        neighborsOfK = findNeighbours(N, locationsOfMonomers[k][0], locationsOfMonomers[k][1]).astype(np.int64)
        
        for element in neighborsOfK:
            for i in range(len(locationsOfMonomers)):
                if np.array_equal(locationsOfMonomers[i], element):
                    clusterGrid = np.where(int(clusterGrid[int(element[0])][int(element[1])]) != clusterGrid, 
                                         clusterGrid, 
                                         clusterGrid[int(locationsOfMonomers[k][0])][int(locationsOfMonomers[k][1])])

        return cluster(clusterGrid, N, k + 1, M)
    
    # Assign unique positive values to clusters
    for i in range(-M, M):
        if i != 0:
            clusterGrid = np.where(i != clusterGrid, clusterGrid, M + 10 * i)

    clusterNumbers = np.unique(clusterGrid)
    numberOfClusters = len(clusterNumbers) - 1

    sizes = np.zeros(numberOfClusters + 1)
    for i in range(len(clusterNumbers)):
        for location in locationsOfMonomers:
            if int(clusterGrid[int(location[0])][int(location[1])]) == int(clusterNumbers[i]):
                sizes[i] += 1

    return clusterGrid, numberOfClusters, sizes

def t_equili(T, T_l):
    """
    Estimate equilibration time as function of temperature.
    
    Parameters:
    -----------
    T : float
        Temperature
    T_l : float
        Lower temperature bound
        
    Returns:
    --------
    int
        Estimated equilibration time steps
    """
    t_max = 100000
    s = 1/200
    C = 10000
    return int(t_max * np.exp(- s * (T - T_l)) + C)

def meanSize(N, M, T0, T1, n):
    """
    Calculate mean cluster size as function of temperature.
    
    Parameters:
    -----------
    N : int
        Grid size
    M : int
        Number of monomer types
    T0, T1 : float
        Temperature range
    n : int
        Number of measurements
        
    Returns:
    --------
    T : ndarray
        Temperature array
    meanSize_t : list
        Mean cluster sizes
    meanSTD_t : list
        Standard deviations
    diff_t : list
        Difference from extreme values
    """
    from monomer_simulation import MC, createSystemOfMonomers
    
    t_r = 1000
    T = np.linspace(T0, T1, 10)
    meanSize_t = list()
    meanSTD_t = list() 
    diff_t = list()

    for t in T:
        t_eq = t_equili(t, T0)
        N_s = t_eq + t_r * n

        grid_eq, epsilon_eq, initialGrid, typeMove = MC(t_eq, N, M, t)

        grid_i = grid_eq
        meanSize_i = list()
        for i in range(n):
            grid_i, epsilon_i, initialGrid, typeMove = MC(t_r, N, M, t, initGrid=grid_i, initial=False)
            
            clusterGrid, numberOfClusters, sizes = cluster(grid_i, N, 0, M)
            meanSize_i.append(sum(sizes)/numberOfClusters)
            
        diff_t.append(np.amax(meanSize_i) - mean(meanSize_i))
        meanSTD_t.append(np.std(meanSize_i))
        meanSize_t.append(mean(meanSize_i))

    return T, meanSize_t, meanSTD_t, diff_t

def meanSizePolyEdition(initialGrid, N, T, n):
    """
    Calculate mean cluster size for polymer systems.
    
    Parameters:
    -----------
    initialGrid : ndarray
        Initial grid configuration
    N : int
        Grid size
    T : float
        Temperature
    n : int
        Number of measurements
        
    Returns:
    --------
    tuple
        Mean cluster size divided by L, mean number of clusters
    """
    from polymer_simulation import MC_Polymers
    
    grid = initialGrid.copy()
    t_eq = t_equili(T, 100)
    t_r = 1000
    
    grid_eq, epsilon_eq, initialGrid = MC_Polymers(t_eq, grid, T, move="rigid")
    
    grid_i = grid_eq
    meanDOfL = list()
    meanNumberOfClusters = list()
    
    for i in range(n):
        grid_i, epsilon_i, initialGrid = MC_Polymers(t_r, grid_i, T, move="rigid")
        
        clusterGrid, numberOfClusters, sizes = cluster(grid_i, N, 0, 25)  # Assuming M=25
        
        meanDOfL.append(sum(sizes)/numberOfClusters)
        meanNumberOfClusters.append(numberOfClusters)
    
    return mean(meanDOfL), mean(meanNumberOfClusters)

def functionsOfL(N, M, L, T):
    """
    Calculate cluster properties as functions of polymer length L.
    
    Parameters:
    -----------
    N : int
        Grid size
    M : int
        Number of polymer types
    L : ndarray
        Array of polymer lengths to test
    T : float
        Temperature
        
    Returns:
    --------
    tuple
        L values, mean d/L, mean number of clusters
    """
    from polymer_simulation import createSystemOfPolymers
    
    dOfL = np.zeros(len(L))
    mOfL = np.zeros(len(L))
    n = 10  # number of measurements

    for i in range(len(L)):
        print(i)
        grid = createSystemOfPolymers(N, M, L[i])
        
        meanDOfL, meanNumberOfClusters = meanSizePolyEdition(grid, N, T, n)
        
        dOfL[i] = meanDOfL/L[i]
        mOfL[i] = meanNumberOfClusters

    return L, dOfL, mOfL