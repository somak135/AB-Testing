import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import beta
import matplotlib.pyplot as plt
import json
import decimal
decimal.getcontext().prec = 4

def calculate_expected_loss(control_simulation, treatment_simulation, treatment_won, min_difference_delta=0):
    loss_control = [max((j - min_difference_delta) - i, 0) for i,j in zip(control_simulation, treatment_simulation)]
    loss_treatment = [max(i - (j - min_difference_delta), 0) for i,j in zip(control_simulation, treatment_simulation)]

    all_loss_control = [int(i)*j for i,j in zip(treatment_won, loss_control)]
    all_loss_treatment = [(1 - int(i))*j for i,j in zip(treatment_won, loss_treatment)]

    expected_loss_control = np.mean(all_loss_control)
    expected_loss_treatment = np.mean(all_loss_treatment)
    return expected_loss_control, expected_loss_treatment

def prop_of_correct_decision_bayesian(n, prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, variant_sample_size = 850, min_simulations_per_experiment=0):
    s = 0
    for simulation in range(0, n):
        control_simulations = np.random.binomial(n=1, p=control_cr, size=variant_sample_size)
        treatment_simulations = np.random.binomial(n=1, p=treatment_cr, size=variant_sample_size)
        
        sample_size = 0
        control_conversions = 0
        treatment_conversions = 0
        
        for i in range(variant_sample_size):
            sample_size += 1
            control_conversions += control_simulations[i]
            treatment_conversions += treatment_simulations[i]

        control_pdfs = np.random.beta(prior_alpha + control_conversions, prior_beta + sample_size - control_conversions, size=1000)
        treatment_pdfs = np.random.beta(prior_alpha + treatment_conversions, prior_beta + sample_size - treatment_conversions, size=1000)
        treatment_pdf_higher = [i <= j for i,j in zip(control_pdfs, treatment_pdfs)]

        expected_loss_control, expected_loss_treatment = calculate_expected_loss(control_pdfs, treatment_pdfs, treatment_pdf_higher)

        if (simulation >= min_simulations_per_experiment) and (expected_loss_treatment <= epsilon):
                s +=1
                
    return (s/n)


x = prop_of_correct_decision_bayesian(1000, 7, 15, 0.32, (0.32*1.1), 0.0015)
print(x)
                
                
            