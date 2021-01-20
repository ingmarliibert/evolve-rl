#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

f1 = "crossover_alldata.npz"
f2 = "crossover_annealing_alldata.npz"
f3 = "annealing_alldata.npz"
f4 = "original_alldata.npz"


def plot_crossover():
    global f1, f4
    d1 = np.load(f1)["data"]
    d4 = np.load(f4)["data"]
    mean1 = np.mean(d1, axis=1)
    std1 = np.std(d1, axis=1)
    mean4 = np.mean(d4, axis=1)
    std4 = np.std(d4, axis=1)
    
    print(mean1.shape, mean4.shape)
    x = np.arange(max(len(mean1), len(mean4)))
    
    font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)
    
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    
    ax.set_title("Without annealing")
    ax.plot(np.arange(len(mean1)), mean1, c="lime", label="mean, crossover")
    ax.plot(np.arange(len(mean1)), std1, c="orangered", label="std, crossover")
    ax.plot(np.arange(len(mean4)), mean4, c="darkgreen", label="mean, vanilla")
    ax.plot(np.arange(len(mean4)), std4, c="darkred", label="std, vanilla")
    ax.legend(loc=(0.68,0.4), fontsize="small")
    
    ax.set_xlabel("Generation number")
    ax.set_ylabel("Score")
    
    plt.show()
    # plt.savefig("non_annealing.png")
    
    
def plot_annealing():
    global f3, f2
    d3 = np.load(f3)["data"]
    d2 = np.load(f2)["data"]
    mean3 = np.mean(d3, axis=1)
    std3 = np.std(d3, axis=1)
    mean2 = np.mean(d2, axis=1)
    std2 = np.std(d2, axis=1)
    
    font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    
    
    
    x = np.arange(max(len(mean3), len(mean2)))
    
    ax.set_title("With annealing")
    ax.plot(x, mean3, c="lime", label="mean, annealing")
    ax.plot(x, std3, c="orangered", label="std, annealing")
    ax.plot(x, mean2, c="darkgreen", label="mean, annealing+crossover")
    ax.plot(x, std2, c="darkred", label="std, annealing+crossover") 
    
    
    ax2 = ax.twinx()
    
    
    y = 0.5*np.exp(-x/40)
    
    ax2.plot(x, y, c="y", label="mutation power")
    
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=(0.55,0.4), fontsize="small")
    
    ax.set_xlabel("Generation number")
    ax2.set_ylabel("Mutation power")
    ax.set_ylabel("Score")
    
    plt.show()
    # plt.savefig("annealing_plot.png")

plot_crossover()
plot_annealing()
