#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:26:06 2021

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


def who_won_bayesian(control_data, treatment_data, prior_alpha, prior_beta, epsilon, min_datasize = 300):
    control_data = np.array(control_data)
    treatment_data = np.array(treatment_data)
    
    number_of_control_win = 0
    number_of_treatment_win = 0
    consumed_sample_size = min_datasize
    flag = 0
    data_length = min(len(control_data), len(treatment_data))
    
    control_conversions = sum(control_data[0 : consumed_sample_size])
    treatment_conversions = sum(treatment_data[0 : consumed_sample_size])
    
    while(flag == 0):
        consumed_sample_size += 20
        if consumed_sample_size <= data_length:
            control_conversions += sum(control_data[(consumed_sample_size - 20) : (consumed_sample_size)])
            treatment_conversions += sum(treatment_data[(consumed_sample_size - 20) : (consumed_sample_size)])
            
            control_posterior_simulation = np.random.beta(prior_alpha + control_conversions, prior_beta + consumed_sample_size - control_conversions, size=1000)
            treatment_posterior_simulation = np.random.beta(prior_alpha + treatment_conversions, prior_beta + consumed_sample_size - treatment_conversions, size=1000)
            treatment_won = (treatment_posterior_simulation >= control_posterior_simulation).astype(int)
            
            expected_loss_control, expected_loss_treatment = calculate_expected_loss(control_posterior_simulation, treatment_posterior_simulation, treatment_won)
            
            if expected_loss_treatment <= epsilon:
                number_of_treatment_win += 1
                flag = 1
            elif expected_loss_control <= epsilon:
                number_of_control_win += 1
                flag = 1
            
        elif consumed_sample_size >= data_length:
            flag = 1
            
    #returndict = {"control": number_of_control_win, "treatment": number_of_treatment_win}
    #return returndict
    return number_of_control_win, number_of_treatment_win


def who_won_classical(control_data, treatment_data, level_of_significance):
    number_of_control_win = 0; number_of_treatment_win = 0
    
    t_stat, p_val = ss.ttest_ind(control_data, treatment_data)
    if p_val <= level_of_significance:
        number_of_treatment_win += 1
    else:
        number_of_control_win += 1
        
    #returndict = {"control": number_of_control_win, "treatment": number_of_treatment_win}
    #return returndict
    return number_of_control_win, number_of_treatment_win


def single_simulation(sample_size, control_cr, treatment_cr, epsilon, level_of_significance, min_datasize = 300):
    control_data = np.random.binomial(n = 1, p = control_cr, size = sample_size)
    treatment_data = np.random.binomial(n = 1, p = treatment_cr, size = sample_size)
    
    bayesian_control_win, bayesian_treatment_win = who_won_bayesian(control_data, treatment_data, 1, 1, epsilon, min_datasize)
    classical_control_win, classical_treatment_win = who_won_classical(control_data, treatment_data, level_of_significance)
    
    #return_dict = {'bayesian_control_win': bayesian_control_win,
     #              'bayesian_treatment_win': bayesian_treatment_win,
      #             'classical_control_win': classical_control_win,
       #            'classical_treatment_win': classical_treatment_win}
    return bayesian_control_win, bayesian_treatment_win, classical_control_win, classical_treatment_win



def simulation_study(N, sample_size, control_cr, treatment_cr, epsilon, level_of_significance, min_datasize = 300):
    num_cores = multiprocessing.cpu_count()
    inputs = range(N)
    
    output = []
    
    output = Parallel(n_jobs=num_cores)(delayed(single_simulation)(sample_size, control_cr, treatment_cr, epsilon, level_of_significance, min_datasize) for i in inputs)
    
    bayesian_control_prop = 0
    bayesian_treatment_prop = 0
    classical_control_prop = 0
    classical_treatment_prop = 0
    
    for i in range(N):
        bayesian_control_prop += output[i][0]/N
        bayesian_treatment_prop += output[i][1]/N
        classical_control_prop += output[i][2]/N
        classical_treatment_prop += output[i][3]/N
        
    print(f'''
          bayesian control proportion = {bayesian_control_prop}, 
          bayesian treatment proportion = {bayesian_treatment_prop},
          classical control proportion = {classical_control_prop},
          classical treatment proportion = {classical_treatment_prop}.
          ''')


print(simulation_study(10000, 1120, 0.3, 0.35, 0.001, 0.05))
