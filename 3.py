#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:41:39 2021

@author: somak
"""

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import json
import decimal
decimal.getcontext().prec = 4
import multiprocessing
from joblib import Parallel, delayed


def calculate_expected_loss(control_simulation, treatment_simulation, treatment_won, min_difference_delta=0):
    control_simulation = np.array(control_simulation)
    treatment_simulation = np.array(treatment_simulation)
    
    loss_control = (treatment_simulation - min_difference_delta) - control_simulation
    loss_treatment = (control_simulation - min_difference_delta) - treatment_simulation
    
    all_loss_control = treatment_won * loss_control
    all_loss_treatment = (1 - treatment_won) * loss_treatment
    
    return np.mean(all_loss_control), np.mean(all_loss_treatment)


def plot_loss(control_data, treatment_data, prior_alpha, prior_beta, epsilon, level_of_significance, min_datasize):
    control_data = np.array(control_data)
    treatment_data = np.array(treatment_data)
    sample_size = min(len(control_data), len(treatment_data))
    
    consumed_sample_size = min_datasize - 2
    consumed_sample_size_vec = []
    
    control_conversions = sum(control_data[0 : consumed_sample_size])
    treatment_conversions = sum(treatment_data[0 : consumed_sample_size])
    loss_control_vec = []
    loss_treatment_vec = []
    
    while consumed_sample_size <= sample_size:
        consumed_sample_size += 2
        consumed_sample_size_vec.append(consumed_sample_size)
        control_conversions += sum(control_data[(consumed_sample_size - 2) : (consumed_sample_size)])
        treatment_conversions += sum(treatment_data[(consumed_sample_size - 2) : (consumed_sample_size)])
            
        control_posterior_simulation = np.random.beta(prior_alpha + control_conversions, prior_beta + consumed_sample_size - control_conversions, size=1000)
        treatment_posterior_simulation = np.random.beta(prior_alpha + treatment_conversions, prior_beta + consumed_sample_size - treatment_conversions, size=1000)
        treatment_won = (treatment_posterior_simulation >= control_posterior_simulation).astype(int)
            
        expected_loss_control, expected_loss_treatment = calculate_expected_loss(control_posterior_simulation, treatment_posterior_simulation, treatment_won)
        loss_control_vec.append(expected_loss_control)
        loss_treatment_vec.append(expected_loss_treatment)
        
    
    plt.plot(np.array([min_datasize, consumed_sample_size]), np.array([epsilon, epsilon]), color = 'b', linewidth = 1)    
    plt.plot(consumed_sample_size_vec, loss_control_vec, color = 'r', label = 'control loss',  linewidth = 0.6)
    plt.plot(consumed_sample_size_vec, loss_treatment_vec, color = 'g', label = 'treatment loss', linewidth = 0.6)
    plt.legend(loc = 'upper right')
    plt.xlabel('sample size')
    plt.ylabel('expected loss')
    #plt.show()
    plt.savefig('plot11', dpi = 800)
    

def plot_posterior(control_data, treatment_data, prior_alpha, prior_beta):
    control_posterior_alpha = prior_alpha + sum(control_data)
    control_posterior_beta = prior_beta + len(control_data) - sum(control_data)
    
    treatment_posterior_alpha = prior_alpha + sum(treatment_data)
    treatment_posterior_beta = prior_beta + len(treatment_data) - sum(treatment_data)
    
    control_posterior_mean = control_posterior_alpha/(control_posterior_alpha + control_posterior_beta)
    treatment_posterior_mean = treatment_posterior_alpha/(treatment_posterior_alpha + treatment_posterior_beta)
    
    leftlim = min(control_posterior_mean, treatment_posterior_mean) - 2*abs(control_posterior_mean - treatment_posterior_mean)
    rightlim = max(control_posterior_mean, treatment_posterior_mean) + 2*abs(control_posterior_mean - treatment_posterior_mean)
    
    x_axis = np.arange(leftlim, rightlim, 0.0001)
    #rv_control = ss.beta(control_posterior_alpha, control_posterior_beta)
    #rv_treatment = ss.beta(treatment_posterior_alpha, treatment_posterior_beta)
    
    plt.plot(x_axis, ss.beta.pdf(x_axis, control_posterior_alpha, control_posterior_beta), color = 'r', label = 'control rate')
    plt.plot(x_axis, ss.beta.pdf(x_axis, treatment_posterior_alpha, treatment_posterior_beta), color = 'g', label = 'treatment rate')
    plt.legend(loc = 'upper right')
    plt.ylabel('Posterior Densities')
    #plt.show()
    plt.savefig('plot12', dpi = 800)
    
    
control_data = np.random.binomial(n = 1, p = 0.2, size = 12000)
treatment_data = np.random.binomial(n = 1, p = 0.21, size = 12000)

plot_loss(control_data, treatment_data, 1, 1, 0.001, 0.05, 500)
plot_posterior(control_data, treatment_data, 1, 1)
         
            
            
        
    