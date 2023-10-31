#%pip install numba
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import copy
import matplotlib.ticker as ticker

from scipy.constants import Boltzmann, epsilon_0, elementary_charge
from itertools import chain
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import colors
from matplotlib import cm
from operator import indexOf
from sqlalchemy import distinct
from sympy import apart

from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

start_time = time.time()
epsilon_r = 79
a = 23.e-6
alpha = elementary_charge**2 /(4 * np.pi * epsilon_0 * epsilon_r * a**2)


##Oppgave 1a)
def createSystemOfMonomers(N, M):
    coordinates = [(x,y) for x in range(N) for y in range(N)]           #Lager liste med alle koordinatene i griden på liste form

    locations = random.sample(coordinates, 2*M)                        #velger 2 * M forskjellige slike koordinater

    grid = np.zeros(shape=(N,N)) 
    distinctValues = list(chain(range(-M, 0), range(1, M+1)))                                     #Lager så griden som NxN matrise
    random.shuffle(distinctValues)                                      #Stokker disse verdiene tilfeldig om

    for i in range(2*M):                                                #Setter hver tilfeldige verdi for monomer i griden lik en tilfeldigverdi
        grid[locations[i][0]][locations[i][1]] = distinctValues[i]
    return grid, locations                                              #Returnerer både girden og lokasjonen av monomerene

N = 20
M = 15

grid, locations = createSystemOfMonomers(N, M)


def checkMonomers(grid, M): #Oppgaven spør å sjekke om det er 2M monomerer i systemet
    nonZeroGrid = np.transpose(np.nonzero(grid))
    numberOfMonomers = len(list(nonZeroGrid))
    if numberOfMonomers == 2*M:
        print("Has 2M monomers in system")
    elif numberOfMonomers < 2*M:
        print("Has less than 2M monomers in system")
    else:
        print("Has more than 2M monomers in system")

checkMonomers(grid, M)


##Oppgave 1b)

"""
def plotSystem(grid, title, n_s=0, m=0, t=0, l=0, cluster=False, polymer=False):
    n = len(grid)
    a=23.e-6

    titleString = title + f", N = {n}"
    if n_s != 0:
        titleString += f", $N_s$ = {n_s}"
    if m != 0:
        titleString += f", M = {m}"
    if t != 0:
        titleString += f", T = {t}"
    if l != 0:
        titleString += f", L = {l}"

    if cluster == True:
        vnorm = TwoSlopeNorm(vcenter = 0.)
        grids[i] = np.ma.masked_where(grids[i] == 0., grids[i])
        cmap = cm.get_cmap("twilight_shifted").copy()
        cmap.set_bad(color="paleturquoise")
        im = axs[i].imshow(grids[i][::-1], interpolation="none", cmap=cmap, norm=vnorm)
    elif polymer == True:
        a=91.e-6
        plt.figure(figsize=(6,6))
        cmap = colors.ListedColormap(["crimson","deeppink","mediumorchid","purple","paleturquoise","g","limegreen","springgreen","lawngreen"])
        plt.pcolormesh(grid[::-1], cmap= cmap)
    else:
        plt.figure(figsize=(6,6))                                   
        cmap = colors.ListedColormap(["purple","paleturquoise","g"])
        plt.pcolormesh(grid[::-1], vmin= -1, vmax= 1, cmap=cmap)                                        #Plotter en grid med pcolormesh

    plt.title(titleString)
    plt.xlabel(f"all units in a = {a}m")
    plt.show()

"""

def plotSystems(grids, info, title):
    if len(shape(grids[0])) == 1:
        grids = [grids]
        info = [info]
    numPlots = len(grids)
    a=23.e-6

    titleString = list()
    for i in range(0, numPlots):
        n = len(grids[i])
        titleString.append(info[0][i] + f", N = {n}")
        if info[1][i] != 0:
            titleString[i] += f", $N_s$ = {info[1][i]}"
        if info[2][i] != 0:
            titleString[i] += f", M = {info[2][i]}"
        if info[3][i] != 0:
            titleString[i] += f", T = {info[3][i]}"
        if info[4][i] != 0:
            titleString[i] += f", L = {info[4][i]}"

    fig, axs = plt.subplots(1,numPlots, figsize=(11, 5))
    fig.subplots_adjust(top=0.8, wspace=0.5)
    fig.suptitle(title, fontsize=16)

    for i in range(0,numPlots):
        if info[5][i] == True: # cluster=True
            vnorm = TwoSlopeNorm(vcenter = 0.)
            grids[i] = np.ma.masked_where(grids[i] == 0., grids[i])
            cmap = cm.get_cmap("twilight_shifted").copy()
            cmap.set_bad(color="paleturquoise")
            im = axs[i].imshow(grids[i][::-1], interpolation="none", cmap=cmap, norm=vnorm)
        elif info[6][i] == True: # polymer=True
            a=91.e-6
            cmap = colors.ListedColormap(["crimson","deeppink","mediumorchid","purple","paleturquoise","g","limegreen","springgreen","lawngreen"])
            im = axs[i].pcolormesh(grids[i][::-1], cmap=cmap)
        else:
            cmap = colors.ListedColormap(["purple","paleturquoise","g"])
            im = axs[i].pcolormesh(grids[i][::-1], vmin= -1, vmax= 1, cmap=cmap)

        axs[i].set_title(titleString[i])
        axs[i].set(xlabel=f"all units in a = {a}m")
    fig.show()

info = [["Initial"],[0],[M],[0],[0],[False],[False]]
plotSystems(grid, info, "System of monomers")


##Oppgave 1c)

def findNeighbours(N, i, j):                                               #Denne funksjonen finner naboene til en gitt koordinat i griden
    neighbours = list()
    above = [(i - 1) % N, j]
    below = [(i + 1) % N, j]
    left = [i, (j - 1) % N]
    right = [i, (j + 1) % N]
    
    neighbours.extend((above, below, left, right))
    return neighbours

##Oppgave 1d)

def totalEnergy(grid):                                      #Finner total energi
    energy = 0
                                                            #nonzero gir ut x og y koordinatene respektivt til monomerene
    non_zero_indexes = np.nonzero(grid)  

    x_values = non_zero_indexes[0]
    y_values = non_zero_indexes[1]   

    for i in range(len(non_zero_indexes[0])):

        neighbours =  findNeighbours(np.shape(grid)[0], x_values[i], y_values[i])

        for j in range(len(x_values)):
            if [x_values[j], y_values[j]] in neighbours:
                energy += np.sign(grid[x_values[i]][y_values[i]] * grid[x_values[j]][y_values[j]])
    return energy * alpha / 2 #Dele på to for å ikke få overlapp

##Oppgave 1e)
 
def newLocationOfMonomer(direction, i, j, N):
    if direction == "left":
        return [(i - 1) % N, j] 
    if direction == "right":
        return [(i + 1) % N, j] 
    if direction == "up":
        return [i, (j + 1) % N] 
    if direction == "down":
        return [i, (j - 1) % N] # Feil mtp "down"

def MC(N_s, N, M, T, initGrid=None, initial=True):
    if initial == True:
        grid, locations = createSystemOfMonomers(N, M)
    else:
        grid = initGrid
        locations = np.transpose(np.nonzero(grid)).tolist()
        locations = [tuple(location) for location in locations]
        

    initialGrid = copy.copy(grid)

    epsilon = list()

    EnergyOfState = totalEnergy(grid)

    directions = ["left", "right", "up", "down"]

    epsilon.append(EnergyOfState)

    for i in range(N_s):
        choosenMonomer = random.randint(0, len(locations)-1)         #Velger indexen i array av x og y verdier som gir pos til monomer
        x_val = locations[choosenMonomer][0]
        y_val = locations[choosenMonomer][1]
        
        valueOfChoosenMonomer = grid[x_val][y_val] # Feil å skrive x, y

        neighbouringLocations = findNeighbours(N, x_val, y_val) 
        neighbouringMonomers = list()
        
        for location in neighbouringLocations:
            if tuple(location) in locations:
                neighbouringMonomers.append(location)

        direction = directions[random.randint(0, len(directions) - 1)]
        
        newLocation = newLocationOfMonomer(direction, x_val, y_val, N)
    
        if newLocation not in neighbouringMonomers:
            oldGrid = copy.copy(grid)
            grid[x_val][y_val] = 0
            grid[newLocation[0]][newLocation[1]] = valueOfChoosenMonomer

            EnergyOfState_new = totalEnergy(grid)

            if EnergyOfState_new < EnergyOfState:
                EnergyOfState = copy.copy(EnergyOfState_new) 
                locations.remove((x_val, y_val))
                locations.append((newLocation[0], newLocation[1]))

                   
            elif random.random() < np.exp(- (1 / (Boltzmann * T)) * (EnergyOfState_new - EnergyOfState)):  
                EnergyOfState = copy.copy(EnergyOfState_new)
                locations.remove((x_val, y_val))
                locations.append((newLocation[0], newLocation[1]))
            
            else:
                grid = copy.copy(oldGrid)
        epsilon.append(EnergyOfState)
    
    return grid, epsilon, initialGrid

def plotEnergy(epsilon, title, n_s=0, m=0, t=0, l=0, comparison=False, epsilon2=0, t2=0, move="", move2=""):
    temp = np.linspace(0, len(epsilon), len(epsilon))
    n = len(grid)

    titleString = title + f", N = {n}"

    if comparison == True:
        if t2 == 0:
            titleString = f"Comparison between {move} and {move2} moves"
        else:
            titleString = f"Comparison between $T_1$ = {t} and $T_2$ = {t2}"
    if n_s != 0:
        titleString += f", $N_s$ = {n_s}"
    if m != 0:
        titleString += f", M = {m}"
    if t != 0 and comparison == False:
        titleString += f", T = {t}"
    if l != 0:
        titleString += f", L = {l}"

    if comparison == False:
        label1 = f"$E_{{{t}}}$(t)"
    else:
        if t2 == 0:
            label1 = f"$E_{{{move}}}$(t)"
            label2 = f"$E_{{{move2}}}$(t)"
        else:
            label1 = f"$E_{{{t}}}$(t)"
            label2 = f"$E_{{{t2}}}$(t)"

    plt.figure(figsize=(8,6))
    plt.title(titleString)
    plt.plot(temp, epsilon, "r", label=label1)
    if comparison == True:
        plt.plot(temp, epsilon2, "b", label=label2)
        plt.legend()
    plt.xlabel("t")
    plt.ylabel("Joule")
    plt.show()

plot = False
if plot == True:
    N_s = 1000
    N = 15
    M = 25

    T = 200
    grid_200, epsilon_200, initialGrid_200 = MC(N_s, N, M, T)
    grids = [initialGrid_200, grid_200]
    info = [["Initial", "Final"],[0, N_s],[M, M],[T, T],[0, 0],[False, False],[False, False]]
    plotMultipleSystems(grids, info, f"System of monomers after {N_s} steps with T = {T}")

    T2 = 500
    grid_500, epsilon_500, initialGrid_500 = MC(N_s, N, M, T2, initGrid=initialGrid_200, initial=False)
    grids = [initialGrid_500, grid_500]
    info = [["Initial", "Final"],[0, N_s],[M, M],[T2, T2],[0, 0],[False, False],[False, False]]
    plotMultipleSystems(grids, info, f"System of monomers after {N_s} steps with T = {T2}")

    plotEnergy(epsilon_200, "Energy comparison", m=M, t=T, comparison=True, epsilon2=epsilon_500, t2=T2)


##Oppgave 1f)

def cluster(grid, N):
    locationOfMonomers = np.transpose(np.nonzero(grid))
    
    clusterGrid = np.zeros([N,N])
    monomersLeft = locationOfMonomers.tolist()
    monomersNext = list()
    clusterSize = list()
    
    val = 1

    while len(monomersLeft) > 0:
        
        y0, x0 = monomersLeft[0]
        monomersLeft.remove([y0, x0])
        monomersNext.append([y0, x0])
        size = 0
    
        while len(monomersNext) > 0:
            
            y, x = monomersNext[0]
            clusterGrid[y,x] = val
            size += 1
            monomersNext.remove([y, x])

            right = [y, (x+1)%N]
            if right in monomersLeft:
                monomersLeft.remove(right)
                monomersNext.append(right)

            left = [y, (x-1)%N]
            if left in monomersLeft:
                monomersLeft.remove(left)
                monomersNext.append(left)

            under = [(y+1)%N, x]
            if under in monomersLeft:
                monomersLeft.remove(under)
                monomersNext.append(under)

            above = [(y-1)%N, x]
            if above in monomersLeft:
                monomersLeft.remove(above)
                monomersNext.append(above)
        val += 1
        clusterSize.append(size)

    return clusterGrid, clusterSize

plot = False
if plot == True:
    T = 200
    clusterGrid_200, clusterSize_200 = cluster(grid_200, 15)
    grids = [grid_200, clusterGrid_200]
    info = [["Initial", "Cluster grid"],[0, N_s],[M, M],[T2, T2],[0, 0],[False, True],[False, False]]
    plotMultipleSystems(grids, info, f"System after clustering after {N_s} steps with T = {T}")

    T2 = 500
    clusterGrid_500, clusterSize_500 = cluster(grid_500, 15)
    grids = [grid_500, clusterGrid_500]
    info = [["Initial", "Cluster grid"],[0, N_s],[M, M],[T2, T2],[0, 0],[False, True],[False, False]]
    plotMultipleSystems(grids, info, f"System after clustering after {N_s} steps with T = {T2}")

from statistics import mean

t_max = 100000
s = 1/200
C = 10000

def t_equili(T, T_l):
    return int(t_max * np.exp(- s * (T - T_l)) + C)

def meanSize(N, M, T0, T1, n):
    t_r = 1000
    T = np.linspace(T0, T1, n)
    meanSize_t = list()

    for t in T:
        t_eq = t_equili(t, T0)
        N_s = t_eq + t_r * n
        print(N_s)

        grid_eq, epsilon_eq, initialGrid = MC(t_eq, N, M, t)

        grid_i = grid_eq
        meanSize_i = list()
        for i in range(n):
            grid_i, epsilon_i, initialGrid = MC(t_r, N, M, t, initGrid=grid_i, initial=False)
            grid_i = copy.copy(grid_i)
            
            clusterGrid, sizes = cluster(grid_i, N)
            meanSize_i.append(mean(sizes))
        
        meanSize_t.append(mean(meanSize_i))

    return T, meanSize_t

plot = False
if plot == True:
    n = 10
    N = 15
    M = 25
    T0 = 100
    T1 = 1000

    T, d = meanSize(N, M, T0, T1, n)

    plt.title(r"$\langle{d}\rangle$ after $t_{equil}$,"+f" with {n} samples between $T_0$={T0} and $T_1$={T1}")
    plt.plot(T, d)
    plt.xlabel("T")
    plt.ylabel(r"$\langle{d}\rangle$")
    plt.show()
    
##Oppgave 2a)
def illegalPlacement(x, y, grid):
    if grid[y][x] != 0:
        return True
    else:
        return False

def randomMonomer(N):
    return random.randint(0, N-1), random.randint(0, N-1)

def findRandomNeighbor(existingPositions, N, grid):
    allowedPositions = list()  ### må ha argument! Lister funke bra!
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
  
    uniqueAllowedPositions = [] # Lager ny liste med kun unike posisjoner
    for item in allowedPositions: # det går an å gjør om til dictionary også tilbake. 
        if item not in uniqueAllowedPositions:
            uniqueAllowedPositions.append(item)

    randomListPosition = random.randint(0, len(uniqueAllowedPositions)-1)
    x, y = uniqueAllowedPositions[randomListPosition]
    return x, y


def createSystemOfPolymers(N, M, L): ### N gridsize, M number of polymers totalt?, L Polymersize 
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
            x, y = findRandomNeighbor(monomerPositionArray, N, grid)
            monomerPositionArray.append([x, y])
            grid[y][x] = i   
    return grid

plot = False
if plot == True:
    N = 20
    M = 2
    L = 20

    grid = createSystemOfPolymers(N, M, L)

    plotSystem(grid, "Initial polymer system", m=M, l=L, polymer=True)
    
##Oppgave 2b)
def totalEnergyPolymerSystem(grid):                                      #Finner total energi
    energy = 0
                                                            #nonzero gir ut x og y koordinatene respektivt til monomerene
    non_zero_indexes = np.nonzero(grid)  

    x_values = non_zero_indexes[0]
    y_values = non_zero_indexes[1]   

    for i in range(len(non_zero_indexes[0])):

        neighbours =  findNeighbours(np.shape(grid)[0], x_values[i], y_values[i])

        for j in range(len(x_values)):
            if [x_values[j], y_values[j]] in neighbours and grid[x_values[i]][y_values[i]] != grid[x_values[j]][y_values[j]]:
                energy += np.sign(grid[x_values[i]][y_values[i]] * grid[x_values[j]][y_values[j]])
    return energy * alpha / 2 #Dele på to for å ikke få overlapp

##Oppgave 2c)

def rigidMove(k, direction, grid):
    newGrid = copy.copy(grid)
    isolatedGrid = np.zeros([N,N])
    locationOfMonomers = np.transpose(np.nonzero(grid))

    for y, x in locationOfMonomers:
        if grid[y, x] == k:
            newGrid[y, x] = 0
            isolatedGrid[y, x] = k

    if direction == 0: #right
        newIsolatedGrid = np.roll(isolatedGrid,1,1)

    if direction == 1: #left
        newIsolatedGrid = np.roll(isolatedGrid,-1,1)
    
    if direction == 2: #up
        newIsolatedGrid = np.roll(isolatedGrid,-1,0)
        
    if direction == 3: #down
        newIsolatedGrid = np.roll(isolatedGrid,1,0)

    isolatedMonomers = np.transpose(np.nonzero(newIsolatedGrid))

    allowedMove = True

    for y, x in isolatedMonomers:
            if newGrid[y, x] != 0:
                allowedMove = False
                break
    
    if allowedMove == True:
        return newGrid + newIsolatedGrid
    else:
        return grid

plot = False
if plot == True:
    N = 20
    M = 2
    L = 20

    grid = createSystemOfPolymers(N, M, L)

    k = -2
    direction = 0
    direction_text = ["right", "left", "up", "down"]

    movedGrid = rigidMove(k, direction, grid)

    grids = [grid, movedGrid]
    info = [["Initial", "Rigid move"],[0, 0],[M, M],[0, 0],[L, L],[False, False],[True, True]]
    plotMultipleSystems(grids, info, "System of polymers after a rigid move one step " + f"{direction_text[direction]}")

##Oppgave 2d)

def MC_Polymers(N_s, N, M, L, T, move="rigid", initGrid=None, initial=True):
    if initial == True:
        grid = createSystemOfPolymers(N, M, L)
    else:
        grid = initGrid

    epsilon = list()

    initialGrid = copy.copy(grid)

    energyOfState = totalEnergyPolymerSystem(grid)
    epsilon.append(energyOfState)

    directions = [0,1,2,3] #right, left, up, down

    valuesOfPolymers = list(chain(range(-M, 0), range(1, M+1)))
    
    for i in range(N_s):
        k = random.choice(valuesOfPolymers)
        direction = random.choice(directions)
        
        oldGrid = copy.copy(grid)
        if move == "rigid":
            newGrid = rigidMove(k, direction, grid)
        elif move == "flexibility":
            newGrid = mediumFlexibilityMove(k, direction, grid)

        energyOfState_new = totalEnergyPolymerSystem(newGrid)
        
        if energyOfState_new < energyOfState:
            grid = copy.copy(newGrid)
            energyOfState = copy.copy(energyOfState_new)

        elif random.random() < np.exp(- (1 / (Boltzmann * T)) * (energyOfState_new - energyOfState)):
            grid = copy.copy(newGrid) #Kommer til å telle"øke energi" selv om griden holder seg konstant
            energyOfState = copy.copy(energyOfState_new)

        else:
            grid = copy.copy(oldGrid)

        epsilon.append(energyOfState)
    
    return grid, epsilon, initialGrid

plot = False
if plot == True:
    N_s = 10000
    N = 20
    M = 4
    L = 12
    T = 200

    grid, epsilon, initialGrid = MC_Polymers(N_s, N, M, L, T)

    #plotSystem(initialGrid, "Initial polymer system", m=M, l=L, polymer=True)
    #plotSystem(grid, "System of polymers after", n_s=N_s, m=M, l=L, polymer=True)

    grids = [initialGrid, grid]
    info = [["Initial", "Final"],[0, N_s],[M, M],[T, T],[L, L],[False, False],[True, True]]
    plotMultipleSystems(grids, info, f"System of polymers after {N_s} steps with T = {T} using rigid moves")

    plotEnergy(epsilon, "Energy polymer system")

##Oppgave 2e og d)

def isBroken(grid, N, k):
    newGrid = copy.copy(grid)
    isolatedGrid = np.zeros([N,N])
    locationOfMonomers = np.transpose(np.nonzero(grid))

    for y, x in locationOfMonomers:
        if grid[y, x] == k:
            newGrid[y, x] = 0
            isolatedGrid[y, x] = k

    clusterGrid, size = cluster(isolatedGrid, N)

    if len(size) > 1:
        return True
    else:
        return False


def mediumFlexibilityMove(k, direction, grid):
    N = len(grid)
    newGrid = copy.copy(grid)

    isolatedGrid = np.zeros([N,N])
    for y, x in np.transpose(np.nonzero(grid)):
        if grid[y, x] == k:
            isolatedGrid[y, x] = k

    if direction == 0:
        a, b, c, d = 0, 1, -1, 1

    if direction == 1:
        a, b, c, d = -1, -1, -1, 1

    if direction == 2:
        a, b, c, d = 0, 1, 1, 0

    if direction == 3:
        a, b, c, d = -1, -1, 1, 0


    if direction >= 2:
        grid = np.transpose(grid)
        newGrid = copy.copy(grid)
        isolatedGrid = np.transpose(isolatedGrid)

    movedGrid = list()
    emptyList = np.zeros(N)

    for i in range(N):
        row = grid[i]
        choosenMonomers = list()
        allowedMove = True
        for j in range(N):
            if row[a + b*j] == k:
                choosenMonomers.append([i,a + b*j])

            if row[a + b*j] != k and row[a + b*j] != 0 and row[a + b*(j + c)%N] == k:
                allowedMove = False

        if allowedMove == True:
            for y, x in choosenMonomers:
                newGrid[y, x] = 0
            movedGrid.append(isolatedGrid[i])
        else:
            movedGrid.append(emptyList.tolist())

    if direction >= 2:
        grid = np.transpose(grid)
        newGrid = np.transpose(newGrid)
        movedGrid = np.transpose(movedGrid)

    newIsolatedGrid = np.roll(movedGrid,-b*c,d)

    if isBroken(newGrid + newIsolatedGrid, N, k):
        allowedMove = False

    if allowedMove == True:
        return newGrid + newIsolatedGrid
    else:
        return grid


plot = False
if plot == True:
    N = 20
    M = 2
    L = 20

    grid = createSystemOfPolymers(N, M, L)

    k = -2
    direction = 3

    movedGrid = mediumFlexibilityMove(k, direction, grid)

    grids = [grid, movedGrid]
    info = [["Initial", "Medium flexibility"],[0, 0],[M, M],[0, 0],[L, L],[False, False],[True, True]]
    plotMultipleSystems(grids, info, "System of polymers after a medium flexibility move one step " + f"{direction_text[direction]}")


plot = False
if plot == True:
    N_s = 1000
    N = 20
    M = 4
    L = 12
    T = 200

    grid, epsilon, initialGrid = MC_Polymers(N_s, N, M, L, T, move="flexibility")

    grids = [initialGrid, grid]
    info = [["Initial", "Final"],[0, N_s],[M, M],[T, T],[L, L],[False, False],[True, True]]
        plotMultipleSystems(grids, info, f"System of polymers after {N_s} steps with T = {T} using medium flexibility moves")

    plotEnergy(epsilon, "Energy polymer system", n_s=N_s, m=M, t=T, l=L)


plot = False
if plot == True:
    N_s = 10000
    N = 20
    M = 4
    L = 12
    T = 200

    grid, epsilon, initialGrid = MC_Polymers(N_s, N, M, L, T)
    grid2, epsilon2, initialGrid2 = MC_Polymers(N_s, N, M, L, T, initGrid=initialGrid, initial=False, move="flexibility")

    grids = [initialGrid, grid]
    info = [["Initial", "Rigid"],[0, N_s],[M, M],[T, T],[L, L],[False, False],[True, True]]
    plotMultipleSystems(grids, info, f"System of polymers after {N_s} steps with T = {T} using rigid moves")

    grids2 = [initialGrid2, grid2]
    info2 = [["Initial", "Medium flexible"],[0, N_s],[M, M],[T, T],[L, L],[False, False],[True, True]]
    plotMultipleSystems(grids2, info2, f"System of polymers after {N_s} steps with T = {T} using medium flexibility moves")

    plotEnergy(epsilon, "Energy polymer system", t=T, comparison=True, epsilon2=epsilon2, move="rigid", move2="medium flexibility")
