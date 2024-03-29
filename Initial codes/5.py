# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 00:20:24 2021

@author: somak
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import beta
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


def reqd_sample_size(prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, control_prop = 0.5, min_simulation_control = 300, sample_size_bound_control = 10000):
    number_of_control_win = 0
    number_of_treatment_win = 0
    sample_size_control = min_simulation_control
    sample_size_treatment = np.ceil(min_simulation_control * (1 - control_prop)/control_prop).astype(int)
    flag = 0
    
    control_conversions = np.random.binomial(n = sample_size_control, p = control_cr, size = 1)
    treatment_conversions = np.random.binomial(n = sample_size_treatment, p = treatment_cr, size = 1)
    
    while flag == 0:
        sample_size_control += 20
        sample_size_treatment += np.ceil(20 * (1 - control_prop)/control_prop).astype(int)
        control_conversions += np.random.binomial(n = 20, p = control_cr, size = 1)
        treatment_conversions += np.random.binomial(n = np.ceil(20 * (1 - control_prop)/control_prop).astype(int), p = treatment_cr, size = 1)
        
        control_posterior_simulation = np.random.beta(prior_alpha + control_conversions, prior_beta + sample_size_control - control_conversions, size=5000)
        treatment_posterior_simulation = np.random.beta(prior_alpha + treatment_conversions, prior_beta + sample_size_treatment - treatment_conversions, size=5000)
        treatment_won = (treatment_posterior_simulation >= control_posterior_simulation).astype(int)
        
        expected_loss_control, expected_loss_treatment = calculate_expected_loss(control_posterior_simulation, treatment_posterior_simulation, treatment_won)
        
        if expected_loss_treatment <= epsilon:
            number_of_treatment_win += 1
            flag = 1
        #elif expected_loss_control <= epsilon:
         #   number_of_control_win += 1
          #  flag = 1
        elif sample_size_control >= sample_size_bound_control:
            flag = 1
            
    return sample_size_control

#reqd_sample_size(7, 15, 0.32, 0.368, 0.0015)

def calculate_reqd_samplesize_control_distbn(n, prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, control_prop = 0.5, power_list = [0.9], min_simulation_control=300, sample_size_bound_control=10000):
    num_cores = multiprocessing.cpu_count()
    inputs = range(1, n)
    
    processed_list = []
    processed_list = Parallel(n_jobs=num_cores)(delayed(reqd_sample_size)(prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, control_prop, min_simulation_control, sample_size_bound_control) for i in inputs)
        
    return np.quantile(processed_list, power_list, axis = 0)


print(calculate_reqd_samplesize_control_distbn(1000, 1, 1, 0.2, 0.21, 0.0005, 0.9, [0.8, 0.85, 0.9, 0.95, 0.98], 4200, 100000))
