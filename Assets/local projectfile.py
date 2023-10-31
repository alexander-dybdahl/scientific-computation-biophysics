import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from itertools import chain
import time
from matplotlib.colors import BoundaryNorm

startTime = time.time()

def illegalPlacement(x, y, grid):
    if grid[y][x] != 0:
        return True
    else:
        return False

def randomMonomer():
    return random.randint(0, N-1), random.randint(0, N-1)

def findRandomNeighbor(existingPositions, N, grid, boundary=False):

    allowedPositions = list()  ### må ha argument! Lister funke bra!
    for i in range(len(existingPositions)):
        x, y = existingPositions[i]
        if boundary == True:
            if x + 1 < N:
                if grid[y][x + 1] == 0:
                    allowedPositions.append([x + 1, y])
                    #print("1")
            if x - 1 >= 0:
                if grid[y][x - 1] == 0:
                    allowedPositions.append([x - 1, y])
                    #print("2")
            if y + 1 < N:
                if grid[y + 1][x] == 0:
                    allowedPositions.append([x, y + 1])
                    #print("3")
            if y - 1 >= 0:
                if grid[y - 1][x] == 0:
                    allowedPositions.append([x, y - 1])
                    #print("4")
        if boundary == False:
            if x + 1 >= N:
                x = 0
            if y + 1 >= N:
                y = 0

            if grid[y][x + 1] == 0:
                allowedPositions.append([x + 1, y])
            if grid[y][x - 1] == 0:
                allowedPositions.append([x - 1, y])
            if grid[y + 1][x] == 0:
                allowedPositions.append([x, y + 1])
            if grid[y - 1][x] == 0:
                allowedPositions.append([x, y - 1])
            
    
    uniqueAllowedPositions = [] # Lager ny liste med kun unike posisjoner
    for item in allowedPositions:
        if item not in uniqueAllowedPositions:
            uniqueAllowedPositions.append(item)

    randomListPosition = random.randint(0, len(uniqueAllowedPositions)-1)
    x, y = uniqueAllowedPositions[randomListPosition]
    return x, y

def generate_grid_of_monomers(N, M, L): ### N gridsize, M number of polymers totalt?, L Polymersize 
    grid = np.zeros([N, N])
    unique_M = chain(range(-M, 0), range(1, M+1))
    
    for i in unique_M:
        ran_x, ran_y = randomMonomer()
        while illegalPlacement(ran_x, ran_y, grid):
            ran_x, ran_y = randomMonomer()
        
        randomPosition = [ran_x, ran_y]
        grid[ran_y][ran_x] = i
        
        monomerPositionArray = list()
        monomerPositionArray.append(randomPosition)
        
        
        for j in range(1, L):
            x, y = findRandomNeighbor(monomerPositionArray, N, grid)
            monomerPositionArray.append([x, y])
            grid[y][x] = i
        
    return grid, monomerPositionArray

def visualize_grid(grid, N):

    plt.imshow(grid,interpolation='none',cmap="PRGn")
    plt.colorbar()
    plt.show()


    """
    negative_grid = np.zeros([N, N])
    positive_grid = np.zeros([N, N])

    for i in range(0, N-1):
        for j in range(0, N-1):
            val = grid[j][i]
            if val < 0:
                negative_grid[j][i] = val
            if val > 0:
                positive_grid[j][i] = val


    plt.imshow(positive_grid, cmap="PRGn")
    plt.show()
    plt.imshow(negative_grid, cmap="Purples")
    plt.show()
    plt.imshow(np.zeros([N, N]), cmap="Set3")
    plt.show()
    
    
    """


def findNearestNeighbors(N, monomerPositionArray, grid):
    nearestNeighbors = list()
    
    for x, y in monomerPositionArray:

        unique_M = grid[y][x]
    
        if x + 1 >= N:
            x = 0
        if y + 1 >= N:
            y = 0

        if grid[y][x + 1] != unique_M and grid[y][x + 1] != 0:
            nearestNeighbors.append([x + 1, y])
        if grid[y][x - 1] != unique_M and grid[y][x - 1] != 0:
            nearestNeighbors.append([x - 1, y])
        if grid[y + 1][x] != unique_M and grid[y + 1][x] != 0:
            nearestNeighbors.append([x, y + 1])
        if grid[y - 1][x] != unique_M and grid[y - 1][x] != 0:
            nearestNeighbors.append([x, y - 1])
    
    uniqueNN = [] # Lager ny liste med kun unike posisjoner
    for item in nearestNeighbors:
        if item not in uniqueNN:
            uniqueNN.append(item)

    return uniqueNN


N = 16
M = 2
L = 20

grid, monomerPositionArray = generate_grid_of_monomers(N, M, L)

#print(grid)
print("This took ", round(time.time() - startTime, 4), "seconds")

print(findNearestNeighbors(N, monomerPositionArray, grid))

visualize_grid(grid, N)




