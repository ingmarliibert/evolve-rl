#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt



f1 = "crossover_alldata.npz"
f2 = "crossover_annealing_alldata.npz"
f3 = "annealing_alldata.npz"
f4 = "original_alldata.npz"

d1 = np.load(f1)["data"]


def plot_crossover(ax, filename, title):
    d = np.load(filename)["data"]
    mean = np.mean(d, axis=1)
    std = np.std(d, axis=1)
    
    ax.set_title(title)
    ax.plot(np.arange(len(mean)), mean, c="g", label="mean")
    ax.plot(np.arange(len(std)), std, c="r", label="std")
    ax.legend()
    


def plot_annealing(ax, filename, title):
    d = np.load(filename)["data"]
    mean = np.mean(d, axis=1)
    std = np.std(d, axis=1)
    
    
    
    
    ax.set_title(title)
    ax.plot(np.arange(len(mean)), mean, c="g", label="mean")
    ax.plot(np.arange(len(std)), std, c="r", label="std")

    ax2 = ax.twinx()
    
    x = np.arange(len(mean))
    y = 0.5*np.exp(-x/40)
    
    ax2.plot(x, y, c="y", label="mutation power")
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="center right")
    #ax2.set_yscale('log')
    

fig, axs = plt.subplots(2,2, tight_layout=True, figsize=(20,16))

plot_annealing(axs[0,0], f3, "annealing only")
plot_annealing(axs[0,1], f3, "annealing and crossover")
plot_crossover(axs[1,0], f1, "crossover only")
plot_crossover(axs[1,1], f4, "original")

plt.savefig("general_statistics.png")




