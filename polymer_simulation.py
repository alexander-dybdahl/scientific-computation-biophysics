"""
Polymer simulation functions for biophysics Monte Carlo simulation.

This module contains functions specific to polymer systems, including
system creation, rigid and flexible moves, and Monte Carlo simulation.
"""

import numpy as np
import random
from numba import jit
from itertools import chain
from core_functions import findNeighbours, inArray, alpha_poly
from scipy.constants import Boltzmann

def illegalPlacement(x, y, grid):
    """
    Check if a position is already occupied by a monomer.
    
    Parameters:
    -----------
    x, y : int
        Coordinates to check
    grid : ndarray
        Grid to check
        
    Returns:
    --------
    bool
        True if position is occupied, False otherwise
    """
    if grid[y][x] != 0:
        return True
    else:
        return False

def randomMonomer(N):
    """
    Generate random coordinates for monomer placement.
    
    Parameters:
    -----------
    N : int
        Grid size
        
    Returns:
    --------
    tuple
        Random (x, y) coordinates
    """
    return random.randint(0, N-1), random.randint(0, N-1)

def findRandomNeighbor(existingPositions, N, grid):
    """
    Find a random neighboring position for polymer expansion.
    
    Parameters:
    -----------
    existingPositions : list
        Current positions of polymer monomers
    N : int
        Grid size
    grid : ndarray
        Current grid state
        
    Returns:
    --------
    tuple
        Random neighboring position (x, y)
    """
    allowedPositions = list()
    
    for i in range(len(existingPositions)):
        x, y = existingPositions[i]
        if grid[y][(x + 1)%N] == 0:
            allowedPositions.append([(x + 1)%N, y])
        if grid[y][(x - 1)%N] == 0:
            allowedPositions.append([(x - 1)%N, y])
        if grid[(y + 1)%N][x] == 0:
            allowedPositions.append([x, (y + 1)%N])
        if grid[(y - 1)%N][x] == 0:
            allowedPositions.append([x, (y - 1)%N])
    
    # Remove duplicates
    uniqueAllowedPositions = []
    for item in allowedPositions: 
        if item not in uniqueAllowedPositions:
            uniqueAllowedPositions.append(item)

    randomIndex = random.randint(0, len(uniqueAllowedPositions)-1)
    x, y = uniqueAllowedPositions[randomIndex]
    return x, y

def createSystemOfPolymers(N, M, L):
    """
    Create a system of polymers on an N×N grid.
    
    Parameters:
    -----------
    N : int
        Grid size (N×N)
    M : int
        Number of polymer types (±1 to ±M)
    L : int
        Length of each polymer
        
    Returns:
    --------
    ndarray
        N×N grid containing polymer system
    """
    grid = np.zeros([N, N])
    unique_M = chain(range(-M, 0), range(1, M+1))

    for i in unique_M:
        ran_x, ran_y = randomMonomer(N)
        while illegalPlacement(ran_x, ran_y, grid):
            ran_x, ran_y = randomMonomer(N)
        
        randomPosition = [ran_x, ran_y]
        grid[ran_y][ran_x] = i
        
        monomerPositionArray = list()
        monomerPositionArray.append(randomPosition)
          
        for j in range(1, L):
            try:
                x, y = findRandomNeighbor(monomerPositionArray, N, grid)
            except ValueError:
                print("fail")
                return createSystemOfPolymers(N,M,L)
            
            monomerPositionArray.append([x, y])
            grid[y][x] = i
    return grid

@jit(nopython=True)
def totalEnergyPolymerSystem(grid):
    """
    Calculate total energy of a polymer system.
    
    Parameters:
    -----------
    grid : ndarray
        N×N grid containing polymers
        
    Returns:
    --------
    float
        Total energy of the system
    """
    energy = 0
    non_zero_indexes = np.argwhere(grid)

    for i in range(len(non_zero_indexes)):
        neighbours = findNeighbours(len(grid), non_zero_indexes[i][0], non_zero_indexes[i][1])

        for j in range(len(non_zero_indexes)):
            if (inArray(neighbours, np.array([int(non_zero_indexes[j][0]), int(non_zero_indexes[j][1])])) and 
                np.not_equal(grid[int(non_zero_indexes[i][0])][int(non_zero_indexes[i][1])], 
                           grid[int(non_zero_indexes[j][0])][int(non_zero_indexes[j][1])])):
                energy += np.sign(grid[int(non_zero_indexes[i][0])][int(non_zero_indexes[i][1])] * 
                                grid[int(non_zero_indexes[j][0])][int(non_zero_indexes[j][1])])
    
    return energy * alpha_poly / 2

@jit(nopython=True)
def rigidMove(k, direction, grid):
    """
    Perform a rigid move of an entire polymer.
    
    Parameters:
    -----------
    k : int
        Polymer identifier
    direction : int
        Direction to move (0=right, 1=left, 2=down, 3=up)
    grid : ndarray
        Current grid state
        
    Returns:
    --------
    ndarray
        Updated grid after move (or original if move illegal)
    """
    N = len(grid)
    newGrid = grid.copy()
    isolatedGrid = np.zeros((N,N))
    locationOfMonomers = np.argwhere(grid)

    # Extract polymer k into isolated grid
    for y, x in locationOfMonomers:
        if grid[y, x] == k:
            newGrid[y, x] = 0
            isolatedGrid[y, x] = k

    # Apply movement
    if direction == 0:  # right
        for i in range(len(isolatedGrid)):
            isolatedGrid[i] = np.roll(isolatedGrid[i], 1)
    if direction == 1:  # left
        for i in range(len(isolatedGrid)):
            isolatedGrid[i] = np.roll(isolatedGrid[i], -1)
    if direction == 2:  # down
        isolatedGrid2 = np.transpose(isolatedGrid)
        for i in range(len(isolatedGrid2)):
            isolatedGrid2[i] = np.roll(isolatedGrid2[i], 1)
        isolatedGrid = np.transpose(isolatedGrid2)
    if direction == 3:  # up
        isolatedGrid2 = np.transpose(isolatedGrid)
        for i in range(len(isolatedGrid2)):
            isolatedGrid2[i] = np.roll(isolatedGrid2[i], -1)
        isolatedGrid = np.transpose(isolatedGrid2)

    # Check if move is legal
    isolatedMonomers = np.argwhere(isolatedGrid)
    allowedMove = True

    for y, x in isolatedMonomers:
        if newGrid[y, x] != 0:
            allowedMove = False
            break
    
    if allowedMove == True:
        isolatedGrid = isolatedGrid.astype(np.int64)
        return newGrid + isolatedGrid
    else:
        return grid

@jit(nopython=True)
def MC_Polymers(N_s, grid, T, move="rigid"):
    """
    Run Monte Carlo simulation for polymer system.
    
    Parameters:
    -----------
    N_s : int
        Number of Monte Carlo steps
    grid : ndarray
        Initial grid configuration
    T : float
        Temperature in Kelvin
    move : str
        Type of move ("rigid" or "flexibility")
        
    Returns:
    --------
    grid : ndarray
        Final grid configuration
    epsilon : ndarray
        Energy evolution over time
    typeMove : ndarray
        Count of different move types
    """
    initialGrid = grid.copy()
    typeMove = np.zeros(3)
    
    epsilon = np.zeros(N_s)
    energyOfState = totalEnergyPolymerSystem(grid)
    epsilon[0] = energyOfState

    valuesOfPolymers = np.unique(grid)
    directions = np.arange(0,3)

    for i in range(N_s):
        k = np.random.choice(valuesOfPolymers)
        direction = np.random.choice(directions)
    
        oldGrid = grid.copy()
        if move == "rigid":
            newGrid = rigidMove(k, direction, grid)
        elif move == "flexibility":
            newGrid = mediumFlexibilityMove(k, direction, grid)
        else:
            newGrid = rigidMove(k, direction, grid)  # default to rigid

        energyOfState_temp = totalEnergyPolymerSystem(newGrid)
        
        if energyOfState_temp < energyOfState:
            grid = newGrid.copy()
            energyOfState = energyOfState_temp
            typeMove[0] += 1
        elif np.random.random() < np.exp(- (1 / (Boltzmann * T)) * (energyOfState_temp - energyOfState)):
            grid = newGrid.copy()   
            energyOfState = energyOfState_temp
            typeMove[1] += 1
        else:
            grid = oldGrid.copy()
            typeMove[2] += 1

        epsilon[i] = energyOfState
    
    return grid, epsilon, typeMove

@jit(nopython=True)
def isBroken(grid, N, k):
    """
    Check if a polymer remains connected after a move.
    
    Parameters:
    -----------
    grid : ndarray
        Grid containing the polymer
    N : int
        Grid size
    k : int
        Polymer identifier
        
    Returns:
    --------
    bool
        True if polymer is broken, False if connected
    """
    isolatedGrid = np.zeros((N,N))
    locationOfMonomers = np.argwhere(grid)

    m = 1
    for y, x in locationOfMonomers:
        if grid[y, x] == k:
            isolatedGrid[y, x] = m
            m += 1

    # Use a simplified connectivity check
    # Check if all monomers are in one connected component
    visited = np.zeros((N, N), dtype=np.bool_)
    component_count = 0
    
    for y in range(N):
        for x in range(N):
            if isolatedGrid[y, x] != 0 and not visited[y, x]:
                component_count += 1
                if component_count > 1:
                    return True
                # DFS to mark all connected monomers
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if visited[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    neighbors = findNeighbours(N, cy, cx)
                    for ny, nx in neighbors:
                        ny, nx = int(ny), int(nx)
                        if (isolatedGrid[ny, nx] != 0 and not visited[ny, nx]):
                            stack.append((ny, nx))
    
    return False  # Only one component found

@jit(nopython=True)
def mediumFlexibilityMove(k, direction, grid):
    """
    Perform flexible movement where segments can move independently while maintaining connectivity.
    
    Parameters:
    -----------
    k : int
        Polymer identifier
    direction : int
        Direction to move (0=right, 1=left, 2=down, 3=up)
    grid : ndarray
        Current grid state
        
    Returns:
    --------
    ndarray
        Updated grid after move (or original if move illegal)
    """
    N = len(grid)
    newGrid = grid.copy()
    isolatedGrid = np.zeros((N,N))
    locationOfMonomers = np.argwhere(grid)

    # Extract polymer k into isolated grid
    for y, x in locationOfMonomers:
        if grid[y, x] == k:
            newGrid[y, x] = 0
            isolatedGrid[y, x] = k

    # Set movement parameter and handle transposition for vertical moves
    if direction == 0:  # right
        a = 1
    elif direction == 1:  # left
        a = -1
    elif direction == 2:  # down
        a = 1
        isolatedGrid = np.transpose(isolatedGrid)
        newGrid = np.transpose(newGrid)
    elif direction == 3:  # up
        a = -1
        isolatedGrid = np.transpose(isolatedGrid)
        newGrid = np.transpose(newGrid)

    locationOfMonomersOnLine = np.argwhere(isolatedGrid)
    
    # Check each row/column for possible movement
    for i in range(len(isolatedGrid)):
        move = True
        for j in range(len(isolatedGrid[i])):
            if not move:
                break
            for l in range(len(locationOfMonomersOnLine)):
                if np.array_equal(locationOfMonomersOnLine[l], np.array([i,j])):
                    if newGrid[i,(j + a)%N] != 0:
                        move = False
                        break
        if move:
            isolatedGrid[i] = np.roll(isolatedGrid[i], a)
                
    # Transpose back if needed
    if direction == 2 or direction == 3:
        isolatedGrid = np.transpose(isolatedGrid)
        newGrid = np.transpose(newGrid)

    # Check if polymer remains connected
    allowedMove = True
    if isBroken(isolatedGrid, N, k):
        allowedMove = False 

    if allowedMove == True:
        return newGrid + isolatedGrid
    else:
        return grid