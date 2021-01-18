#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt 

f1 = "crossover_data.txt"
f2 = "annealing_data.txt"
f3 = "original_data.txt"
f4 = "crossover_annealing_data.txt"



def read_file(filename):
    times = []
    elite_scores = []
    with open(filename) as f:
        count = 0
        for line in f.readlines():
            if count == 0:
                line = line.strip().split(",")
                for el in line:
                    elite_scores.append(float(el))
                    
            elif count == 1:
                line = line.strip().split(",")
                for el in line:
                    times.append(float(el))
                    
            else:
                continue 
                
            count += 1
                
    return times, elite_scores
    
    
def plot_annealing_elites():
    global f2, f4
    
    ann_times, ann_scores = read_file(f2)
    comb_times, comb_scores = read_file(f4)

    fig, ax = plt.subplots(1,1)
    ax.plot(ann_times, ann_scores, c="r", label="annealing")
    ax.plot(comb_times, comb_scores, c="g", label="crossover and annealing")
    
    plt.legend()
    
    ax.set_xlim([0, 200])
    
    plt.savefig("annealing_elites_time.png")
    plt.close(fig)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(ann_scores)), ann_scores, c="r", label="annealing")
    ax.plot(np.arange(len(comb_scores)), comb_scores, c="g", label="crossover and annealing")
    
    plt.legend()
    
    ax.set_xlim([0, 20])
    
    plt.savefig("annealing_elites_generation.png")
    plt.close(fig)
   
   
def plot_non_annealing_elites():
    global f1, f3
    
    cross_times, cross_scores = read_file(f1)   
    orig_times, orig_scores = read_file(f3)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(cross_times, cross_scores, c="r", label="crossover")
    ax.plot(orig_times, orig_scores, c="g", label="original")
    
    plt.legend()
    
    
    plt.savefig("non_annealing_elites_time.png")
    plt.close(fig)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(cross_times)), cross_scores, c="r", label="crossover")
    ax.plot(np.arange(len(orig_scores)), orig_scores, c="g", label="original")
    
    plt.legend()
    
    
    plt.savefig("non_annealing_elites_generation.png")
    plt.close(fig)
    

# plot_annealing_elites()
# plot_non_annealing_elites()

