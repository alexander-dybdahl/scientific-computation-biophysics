"""
Visualization functions for biophysics Monte Carlo simulation.

This module contains functions for plotting and visualizing
simulation results, including system states and energy evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

def plotSystems(grids, info, title):
    """
    Plot multiple system grids side by side.
    
    Parameters:
    -----------
    grids : list
        List of grids to plot
    info : list
        2D list containing plot information [titles, N_s, M, T, L, cluster_flag, polymer_flag]
    title : str
        Main figure title
    """
    numPlots = len(grids)
    a = (23.e-6)**2

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

    if len(grids) == 1:
        fig, axs = plt.subplots(1,1, figsize=(8, 8))
        axs = [axs,axs]
    else:
        fig, axs = plt.subplots(1,numPlots, figsize=(7*numPlots, 7))
    fig.subplots_adjust(top=0.8, wspace=0.3)
    fig.suptitle(title, fontsize=16)

    backgroundColor = "paleturquoise"
    for i in range(0,numPlots):
        if info[5][i] == True:  # cluster coloring
            vnorm = TwoSlopeNorm(vcenter = 0.)
            grids[i] = np.ma.masked_where(grids[i] == 0., grids[i])
            cmap = cm.get_cmap("twilight_shifted").copy()
            cmap.set_bad(color=backgroundColor)
            im = axs[i].pcolormesh(grids[i][::-1], cmap=cmap, norm=vnorm)
        elif info[6][i] == True:  # polymer coloring
            a = (91.e-6)**2
            vnorm = TwoSlopeNorm(vcenter = 0.)
            grids[i] = np.ma.masked_where(grids[i] == 0., grids[i])
            cmap = cm.get_cmap("PRGn").copy()
            cmap.set_bad(color=backgroundColor)
            im = axs[i].pcolormesh(grids[i][::-1], cmap=cmap, norm=vnorm)
        else:  # monomer coloring
            cmap = colors.ListedColormap(["purple",backgroundColor,"g"])
            im = axs[i].pcolormesh(grids[i][::-1], vmin= -1, vmax= 1, cmap=cmap)

        axs[i].set_title(titleString[i])
        axs[i].set(xlabel=f"all units in a = {a}m")

def plotEnergy(epsilon, title, n_s=0, m=0, t=0, l=0, comparison=False, epsilon2=0, t2=0, move="", move2=""):
    """
    Plot energy evolution over Monte Carlo steps.
    
    Parameters:
    -----------
    epsilon : ndarray
        Energy evolution array
    title : str
        Plot title
    n_s, m, t, l : int/float
        System parameters for title
    comparison : bool
        Whether to plot comparison between two energy curves
    epsilon2 : ndarray
        Second energy curve for comparison
    t2 : float
        Second temperature for comparison
    move, move2 : str
        Move types for comparison labels
    """
    t_val = np.linspace(0, len(epsilon), len(epsilon))
    
    titleString = title
    if n_s != 0:
        titleString += f", $N_s$ = {n_s}"
    if m != 0:
        titleString += f", M = {m}"
    if t != 0 and comparison == False:
        titleString += f", T = {t}"
    if l != 0:
        titleString += f", L = {l}"

    if comparison == True:
        if t2 == 0:
            titleString = f"Comparison between {move} and {move2} moves"
            label1 = f"$E_{{{move}}}$(t)"
            label2 = f"$E_{{{move2}}}$(t)"
        else:
            titleString = f"Comparison between $T_1$ = {t} and $T_2$ = {t2}"
            label1 = f"$E_{{{t}}}$(t)"
            label2 = f"$E_{{{t2}}}$(t)"
    else:
        label1 = f"$E_{{{t}}}$(t)" if t != 0 else "E(t)"

    plt.figure(figsize=(12,8))
    plt.title(titleString)
    plt.plot(t_val, epsilon, "r", label=label1)
    if comparison == True: 
        plt.plot(t_val, epsilon2, "b", label=label2)
        plt.legend()
    plt.xlabel("t")
    plt.ylabel("Joule")
    plt.show()

def compareMeanSize(d1, d2, t0, t1, title):
    """
    Compare mean cluster sizes between two datasets.
    
    Parameters:
    -----------
    d1, d2 : list
        Data arrays to compare
    t0, t1 : float
        Temperature range
    title : str
        Plot title
    """
    t_val = np.linspace(t0, t1, 10)
    
    titleString = f"Comparison between mean d_1 and mean d_2, N = 10, T_0 = {t0}, T_1 = {t1}"

    plt.figure(figsize=(12,8))
    plt.title(titleString)
    plt.plot(t_val, d1, "r", label="mean d_1")
    plt.plot(t_val, d2, "b", label="mean d_2")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("d")
    plt.show()

def plotMoveTypeDistribution(typeMove_poly, typeMove_mono):
    """
    Plot distribution of different move types.
    
    Parameters:
    -----------
    typeMove_poly : ndarray
        Move type counts for polymer system
    typeMove_mono : ndarray
        Move type counts for monomer system
    """
    plt.figure(figsize=(12,8))
    plt.title("Move type distribution")
    plt.bar(["Favorable","Random","None"], typeMove_poly, width=0.1, align="center", label="Polymer")
    plt.bar(["Favorable","Random","None"], typeMove_mono, width=0.1, align="edge", label="Monomer")
    plt.legend()
    plt.xlabel("Type movement")
    plt.ylabel("Times moved")
    plt.show()

# This function already exists in visualization.py