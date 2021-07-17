import ipywidgets as widgets
from IPython.display import Markdown, display
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import beta
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import decimal
decimal.getcontext().prec = 4
import multiprocessing
from joblib import Parallel, delayed

def printmd(string, color='red'):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))


############## BAYESIAN PROCESS ##################
def calculate_expected_loss(control_simulation, treatment_simulation, treatment_won, min_difference_delta=0):
    control_simulation = np.array(control_simulation)
    treatment_simulation = np.array(treatment_simulation)
    
    loss_control = (treatment_simulation - min_difference_delta) - control_simulation
    loss_treatment = (control_simulation - min_difference_delta) - treatment_simulation
    
    all_loss_control = treatment_won * loss_control
    all_loss_treatment = (1 - treatment_won) * loss_treatment
    
    return np.mean(all_loss_control), np.mean(all_loss_treatment)


def bayesian_sample_size(prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control):
    number_of_control_win = 0
    number_of_treatment_win = 0
    sample_size_control = min_simulation_control
    sample_size_treatment = np.ceil(min_simulation_control * (1 - control_prop)/control_prop).astype(int)
    flag = 0
    
    control_conversions = np.random.binomial(n = sample_size_control, p = control_cr, size = 1)
    treatment_conversions = np.random.binomial(n = sample_size_treatment, p = treatment_cr, size = 1)
    
    while flag == 0:
        sample_size_control += 50
        sample_size_treatment += round(50 * (1 - control_prop)/control_prop)
        control_conversions += np.random.binomial(n = 50, p = control_cr, size = 1)
        treatment_conversions += np.random.binomial(n = round(50 * (1 - control_prop)/control_prop), p = treatment_cr, size = 1)
        
        control_posterior_simulation = np.random.beta(prior_alpha + control_conversions, prior_beta + sample_size_control - control_conversions, size=5000)
        treatment_posterior_simulation = np.random.beta(prior_alpha + treatment_conversions, prior_beta + sample_size_treatment - treatment_conversions, size=5000)
        treatment_won = (treatment_posterior_simulation >= control_posterior_simulation).astype(int)
        
        expected_loss_control, expected_loss_treatment = calculate_expected_loss(control_posterior_simulation, treatment_posterior_simulation, treatment_won)
        
        if expected_loss_treatment <= treatment_threshold*(expected_loss_control+expected_loss_treatment):
            number_of_treatment_win += 1
            flag = 1
        elif sample_size_control >= sample_size_bound_control:
            flag = 1
            
    return max(sample_size_control, round(sample_size_treatment*control_prop/(1 - control_prop)))


def calculate_bayesian_samplesize_control_distbn(n, prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control):
    num_cores = multiprocessing.cpu_count()
    inputs = range(1, n)
    
    processed_list = []
    processed_list = Parallel(n_jobs=num_cores)(delayed(bayesian_sample_size)(prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control) for i in inputs)
    return processed_list

def bayesian_samplesize_multiple_power(progressbar, n, prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, power_list, min_simulation_control, sample_size_bound):
    progressbar.value = 0
    display(progressbar)
    N = 10
    k = np.ceil(n/10).astype(int) + 1
    complete_list = np.array([])
    for i in range(N):
        list1 = calculate_bayesian_samplesize_control_distbn(k, prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound)
        complete_list = np.append(complete_list, list1)
        progressbar.value += 1
        
    progressbar.layout.visibility = 'hidden'
    return np.quantile(complete_list, power_list, axis = 0)


############## CLASSICAL PROCESS 1 - sided ######################

def classical_1_samplesize_multiple_power(control_cr, treatment_cr, control_group, level_of_sig, power_list):
    p1 = control_cr
    p2 = treatment_cr
    q1 = 1 - p1
    q2 = 1 - p2
    delta = abs(p2 - p1)
    k = (1 - control_group)/control_group
    pbar = (p1 + k*p2)/(1 + k)
    qbar = 1 - pbar
    
    term1 = np.sqrt(pbar * qbar * (1+1/k)) * norm.ppf(1 - level_of_sig)  ##########
    sample_size = []
    for i in range(len(power_list)):
        term2 = np.sqrt(p1 * q1 + p2 * q2/k) * norm.ppf(power_list[i])
        sample_size.append(((term1 + term2)**2)/(delta**2))
    
    return sample_size

############## CLASSICAL PROCESS 2 - sided ######################
def classical_2_samplesize_multiple_power(control_cr, treatment_cr, control_group, level_of_sig, power_list):
    p1 = control_cr
    p2 = treatment_cr
    q1 = 1 - p1
    q2 = 1 - p2
    delta = abs(p2 - p1)
    k = (1 - control_group)/control_group
    pbar = (p1 + k*p2)/(1 + k)
    qbar = 1 - pbar
    
    term1 = np.sqrt(pbar * qbar * (1+1/k)) * norm.ppf(1 - level_of_sig/2)  ##########
    sample_size = []
    for i in range(len(power_list)):
        term2 = np.sqrt(p1 * q1 + p2 * q2/k) * norm.ppf(power_list[i])
        sample_size.append(((term1 + term2)**2)/(delta**2))
    
    return sample_size

################ WIDGETS AND DISPLAY FUNCTIONS ####################

style = {'description_width': 'initial'}
control_cr = widgets.BoundedFloatText(
            value = 0.2,
            min = 0,
            step = 0.1,
            max = 1,
            description = f'<b>Baseline conversion rate</b>',
            disabled = False,
            style = style
)

expected_lift = widgets.BoundedFloatText(
                value = 0.05,
                min = 0,
                step = 0.005,
                max = 1,
                description = f'<b>Expected lift</b>',
                readout_format = '.4f',
                disabled = False,
                style = style
)

eps_values = np.arange(1,11,1)
eps_values_text = []
for i in range(len(eps_values)):
    x = f'{eps_values[i]}%'
    eps_values_text.append(x)

epsilon = widgets.SelectionSlider(
        options = eps_values_text,
        value = '5%',
        description = f'<b>Expected Loss Threshold</b>',
        readout = True,
        style = style
)

level_of_sig_values = np.arange(1,11,1)
level_of_sig_text = []
for i in range(len(level_of_sig_values)):
    x = f'{level_of_sig_values[i]}%'
    level_of_sig_text.append(x)
    
level_of_sig = widgets.SelectionSlider(
                options = level_of_sig_text,
                value = '5%',
                description = f'<b>Level of Significance</b>',
                readout = True,
                style = style
)

values = np.arange(5, 100, 5)
values_text = []
for i in range(len(values)):
    values_text.append(values[i].astype(str) + '%')

control_prop = widgets.Dropdown(
                options = values_text,
                value = '50%',
                description = f'<b>Proportion of Control samples</b>',
                style = style
)

power_values = np.arange(75, 100, 1)
power_values_text = []
for i in range(len(power_values)):
    power_values_text.append(power_values[i].astype(str) + '%')

power = widgets.SelectMultiple(
        options = power_values_text,
        value = ['90%'],
        description = f'<b>Power/Conclusive probability</b>',
        style = style
)

power_label = widgets.Label('ctrl + Click for multiple selection')
power_box = widgets.HBox([power, power_label])

method_choice = widgets.SelectMultiple(
                options = ['Classical(One sided)', 'Classical(Two sided)', 'Bayesian'],
                value = ['Classical(Two sided)'],
                description = f'<b>Method</b>',
                style = style
)


Bayesian_loading = widgets.IntProgress(
    value=0,
    min=0,
    max=10,
    description='Bayesian running:',
    bar_style='info',
    style={'bar_color': '#00FF00', 'description_width': 'initial'},
    orientation='horizontal'
)

def threshold_display(arr):
    if ('Classical(One sided)' in arr) or ('Classical(Two sided)' in arr):
        printmd('**Enter Level of Significance for Classical calculation:**')
        printmd('A Classical test concludes when the observed p-value is less than the **Level of significance**. Lower the value of the level of significance, more is the confidence in declaring a winner and higher is the sample size requirement. The convension is to set a very low value for this threshold and the default is 5%.', color = 'black')
        display(level_of_sig)
            
    if 'Bayesian' in arr:
        printmd('**Enter Expected Loss Threshold for Bayesian calculation:**')
        printmd('A Bayesian test concludes when (Expected loss of one variant)/(Sum of expected losses of the two variants) is less than the **Expected loss threshold**. Lower the value of the threshold, more is the confidence in declaring a winner and higher is the sample size requirement. The convension is to set a very low value for this threshold and the default is 5%.', color = 'black')
        display(epsilon)
        
def samplesize_calculate(progressbar, arr, control_cr, expected_lift, power_list, control_prop, level_of_sig, epsilon):
    if expected_lift == 0:
        printmd("**Error : 0% detectable lift is not a valid input for detectable lift.**")
    else:
        power_numeric = []
        for i in range(len(power_list)):
            power_numeric.append(int(power_list[i][:-1])/100)
        
        level_of_sig = int(level_of_sig[:-1])/100
        epsilon = int(epsilon[:-1])/100
        ref_size = classical_2_samplesize_multiple_power(control_cr, control_cr+expected_lift, control_prop, level_of_sig, power_numeric)
        
        if 'Classical(One sided)' in arr:
            classical_size = classical_1_samplesize_multiple_power(control_cr, control_cr+expected_lift, control_prop, level_of_sig, power_numeric)
            printmd('**Required sample size for one sided Classical test:**\n')
            for i in range(len(classical_size)):
                print(f'Power {power_list[i]} : Required sample sizes for control group: {np.ceil(classical_size[i])} \t test group: {np.ceil(classical_size[i]*(1 - control_prop)/control_prop)}')
                
        if 'Classical(Two sided)' in arr:
            classical_size = classical_2_samplesize_multiple_power(control_cr, control_cr+expected_lift, control_prop, level_of_sig, power_numeric)
            printmd('**Required sample size for two sided Classical test:**\n')
            for i in range(len(classical_size)):
                print(f'Power {power_list[i]} : Required sample sizes for control group: {np.ceil(classical_size[i])} \t test group: {np.ceil(classical_size[i]*(1 - control_prop)/control_prop)}')
        
        if 'Bayesian' in arr:
            progressbar.layout.visibility = 'visible'
            bayesian_size = bayesian_samplesize_multiple_power(progressbar, 5000, 1, 1, control_cr, control_cr+expected_lift, epsilon, control_prop, power_numeric, classical_size[0]/10, classical_size[len(power_list)-1])
            printmd('**Required sample size for Bayesian test:**\n')
            for i in range(len(bayesian_size)):
                print(f'Power {power_list[i]} : Required sample sizes for control group: {np.ceil(bayesian_size[i])} \t test group: {np.ceil(bayesian_size[i]*(1 - control_prop)/control_prop)}')