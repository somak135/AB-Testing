#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 01:15:26 2021

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



def calculate_expected_loss(control_simulation, treatment_simulation, treatment_won, min_difference_delta=0):
    loss_control = [max((j - min_difference_delta) - i, 0) for i,j in zip(control_simulation, treatment_simulation)]
    loss_treatment = [max(i - (j - min_difference_delta), 0) for i,j in zip(control_simulation, treatment_simulation)]

    all_loss_control = [int(i)*j for i,j in zip(treatment_won, loss_control)]
    all_loss_treatment = [(1 - int(i))*j for i,j in zip(treatment_won, loss_treatment)]

    expected_loss_control = np.mean(all_loss_control)
    expected_loss_treatment = np.mean(all_loss_treatment)
    return expected_loss_control, expected_loss_treatment


def run_multiple_experiment_simulations(n, prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, variant_sample_size=10000, min_simulations_per_experiment=0):
    output = pd.DataFrame()

    for simulation in range(0,n):
        records = []
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
                records.append({'simulation': simulation+1, 'sample': sample_size, 'treatment_cr': (treatment_conversions/sample_size), 'control_cr': (control_conversions/sample_size), 'treatment_expected_loss': expected_loss_treatment, 'control_expected_loss': expected_loss_control, 'winner': 'treatment'})
            elif (simulation >= min_simulations_per_experiment) and expected_loss_control <= epsilon:
                records.append({'simulation': simulation+1, 'sample': sample_size, 'treatment_cr': (treatment_conversions/sample_size), 'control_cr': (control_conversions/sample_size), 'treatment_expected_loss': expected_loss_treatment, 'control_expected_loss': expected_loss_control, 'winner': 'control'})
            else:
                records.append({'simulation': simulation+1, 'sample': sample_size, 'treatment_cr': (treatment_conversions/sample_size), 'control_cr': (control_conversions/sample_size), 'treatment_expected_loss': expected_loss_treatment, 'control_expected_loss': expected_loss_control, 'winner': 'inconclusive'})

        simulation_results = pd.DataFrame.from_records(records)
        output = pd.concat([output, simulation_results])    
    
    return output

#standard_simulations = run_multiple_experiment_simulations(2, 7, 15, 0.32, 0.32*(1.15), 0.0015)


def plot_simulations(file_name):
    simulations = pd.read_csv(file_name)
    no_of_simulations = simulations['simulation'].max()

    _, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(1, no_of_simulations+1):

        filtered = simulations[simulations['simulation'] == i]


        if i == 1:
            control = ax1.plot(filtered['sample'], filtered['control_expected_loss'], label='control')
            treatment = ax1.plot(filtered['sample'], filtered['treatment_expected_loss'], label='treatment')
            threshold = ax1.plot(filtered['sample'], np.full((10000,), 0.0015), label='threshold', linestyle='dashed')

            ax2.plot(filtered['sample'], filtered['control_cr'], label='control', color=control[0].get_color())
            ax2.plot(filtered['sample'], filtered['treatment_cr'], label='treatment', color=treatment[0].get_color())
        else:
            ax1.plot(filtered['sample'], filtered['control_expected_loss'], linewidth=0.25, color=control[0].get_color(), alpha=0.3)
            ax1.plot(filtered['sample'], filtered['treatment_expected_loss'], linewidth=0.25, color=treatment[0].get_color(), alpha=0.3)
            ax2.plot(filtered['sample'], filtered['control_cr'], linewidth=0.25, color=control[0].get_color(), alpha=0.3)
            ax2.plot(filtered['sample'], filtered['treatment_cr'], linewidth=0.25, color=treatment[0].get_color(), alpha=0.3)
        
        
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Expected Loss')
        ax1.set_title('Expected Loss Simulation')
        ax1.legend()
        
        
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Conversion Rates')
        ax2.set_title('Conversion Rates Simulation')
        ax2.legend()
    
    ax1.plot(filtered['sample'], np.full((10000,), 0.0015), label='threshold', linestyle='dashed', color=threshold[0].get_color())

    plt.savefig('wtfplot.png', dpi = 500)
    plt.close()


plot_simulations('experiment_simulations_37.csv')
