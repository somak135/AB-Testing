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
        sample_size_treatment += round(20 * (1 - control_prop)/control_prop)
        control_conversions += np.random.binomial(n = 20, p = control_cr, size = 1)
        treatment_conversions += np.random.binomial(n = round(20 * (1 - control_prop)/control_prop), p = treatment_cr, size = 1)
        
        control_posterior_simulation = np.random.beta(prior_alpha + control_conversions, prior_beta + sample_size_control - control_conversions, size=5000)
        treatment_posterior_simulation = np.random.beta(prior_alpha + treatment_conversions, prior_beta + sample_size_treatment - treatment_conversions, size=5000)
        treatment_won = (treatment_posterior_simulation >= control_posterior_simulation).astype(int)
        
        expected_loss_control, expected_loss_treatment = calculate_expected_loss(control_posterior_simulation, treatment_posterior_simulation, treatment_won)
        
        if expected_loss_treatment <= epsilon and treatment_won.mean()>=0.95:
            number_of_treatment_win += 1
            flag = 1
        #elif expected_loss_control <= epsilon:
         #   number_of_control_win += 1
          #  flag = 1
        elif sample_size_control >= sample_size_bound_control:
            flag = 1
            
    return sample_size_control


def calculate_reqd_samplesize_control_distbn(n, prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, control_prop = 0.5, min_simulation_control=300, sample_size_bound_control=10000):
    num_cores = multiprocessing.cpu_count()
    inputs = range(1, n)
    
    processed_list = []
    processed_list = Parallel(n_jobs=num_cores)(delayed(reqd_sample_size)(prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, control_prop, min_simulation_control, sample_size_bound_control) for i in inputs)
    return processed_list

def final_samplesize_multiple_power(progressbar, n, prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, control_prop = 0.5, power_list = [0.9], min_simulation_control = 300, sample_size_bound = 10000):
    progressbar.value = 0
    display(progressbar)
    N = 10
    k = np.ceil(n/10).astype(int) + 1
    complete_list = np.array([])
    for i in range(N):
        list1 = calculate_reqd_samplesize_control_distbn(k, prior_alpha, prior_beta, control_cr, treatment_cr, epsilon, control_prop, min_simulation_control, sample_size_bound)
        complete_list = np.append(complete_list, list1)
        progressbar.value += 1
        
    progressbar.layout.visibility = 'hidden'
    return np.quantile(complete_list, power_list, axis = 0)


############## CLASSICAL PROCESS ######################
def calculate_reqd_samplesize_classical(control_cr, treatment_cr, control_group, level_of_sig, power_list):
    p1 = control_cr
    p2 = treatment_cr
    q1 = 1 - p1
    q2 = 1 - p2
    delta = abs(p2 - p1)
    k = (1 - control_group)/control_group
    pbar = (p1 + k*p2)/(1 + k)
    qbar = 1 - pbar
    
    term1 = np.sqrt(pbar * qbar * (1+1/k)) * norm.ppf(1 - level_of_sig/2)
    sample_size = []
    for i in range(len(power_list)):
        term2 = np.sqrt(p1 * q1 + p2 * q2/k) * norm.ppf(power_list[i])
        sample_size.append(((term1 + term2)**2)/(delta**2))
    
    return sample_size



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

detectable_lift = widgets.BoundedFloatText(
                value = 0.05,
                min = 0,
                step = 0.005,
                max = 1,
                description = f'<b>Minimum detectable lift</b>',
                readout_format = '.4f',
                disabled = False,
                style = style
)

eps_default = control_cr.value * 0.005 #default is 0.5% of baseline conversion rate is loss threshold
eps_min = control_cr.value * 0.001 #minimum possible value of loss threshold is 0.1% of baseline conversion rate
eps_max = control_cr.value * 0.1 #maximum possible value of loss threshold is 10% of baseline conversion rate
epsilon = widgets.FloatSlider(
        value = eps_default,
        min = eps_min,
        max = eps_max,
        step = eps_min,
        description = f'<b>Expected Loss Threshold</b>',
        readout = False,
        style = style
)
eps_label = widgets.Label()
eps_box = widgets.HBox([epsilon, eps_label])
mylink = widgets.jslink((epsilon, 'value'), (eps_label, 'value'))

level_of_sig_default = 0.05
level_of_sig_min = 0.01
level_of_sig_max = 0.1
level_of_sig = widgets.FloatSlider(
                value = level_of_sig_default,
                min = level_of_sig_min,
                max = level_of_sig_max,
                step = 0.01,
                description = f'<b>Level of Significance</b>',
                readout_format = '.2f',
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
                options = ['Classical', 'Bayesian'],
                value = ['Classical'],
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

def threshold_display(arr, control_cr):
    if len(arr) == 1:
        if arr[0] == 'Classical':
            printmd('**Enter Level of Significance for Classical calculation:**')
            display(level_of_sig)
            
        if arr[0] == 'Bayesian':
            printmd('**Enter Expected Loss Threshold for Bayesian calculation:**')
            epsilon.value = control_cr * 0.005
            epsilon.min = control_cr * 0.001
            epsilon.max = control_cr * 0.01
            epsilon.step = epsilon.min
            printmd(f'The **expected loss threshold** is our tolerance of loss in conversion rate if the test displays a wrong result. The convension is to consider a very small value for this threshold. The default value is set at 0.5% of the current conversion rate which is {control_cr} × 0.005 = {control_cr*0.005}.', color = 'black')
            display(eps_box)
            
    if len(arr) == 2:
        printmd('**Enter Level of Significance for Classical calculation and Expected Loss Threshold for Bayesian calculation:**')
        epsilon.value = control_cr * 0.005
        epsilon.min = control_cr * 0.001
        epsilon.max = control_cr * 0.01
        epsilon.step = epsilon.min
        display(level_of_sig)
        printmd(f'The **expected loss threshold** is our tolerance of loss in conversion rate if the test displays a wrong result. The convension is to consider a very small value for this threshold. The default value is set at 0.5% of the current conversion rate which is {control_cr} × 0.005 = {control_cr*0.005}.', color = 'black')
        display(eps_box)
        
        
def samplesize_calculate(progressbar, arr, control_cr, detectable_lift, power_list, control_prop, level_of_sig, epsilon):
    if detectable_lift == 0:
        printmd("**Error : 0% detectable lift is not a valid input for detectable lift.**")
    else:
        power_numeric = []
        for i in range(len(power_list)):
            power_numeric.append(int(power_list[i][:-1])/100)
    
        classical_size = calculate_reqd_samplesize_classical(control_cr, control_cr+detectable_lift, control_prop, level_of_sig, power_numeric)
        if len(arr) == 1:
            if arr[0] == 'Classical':
                printmd('**Required sample size by Classical method:**\n')
                for i in range(len(classical_size)):
                    print(f'Power {power_list[i]} : Required sample sizes for control : {np.ceil(classical_size[i])} \t test : {np.ceil(classical_size[i]*(1 - control_prop)/control_prop)}')
            if arr[0] == 'Bayesian':
                progressbar.layout.visibility = 'visible'
                bayesian_size = final_samplesize_multiple_power(progressbar, 5000, 1, 1, control_cr, control_cr+detectable_lift, epsilon, control_prop, power_numeric, min_simulation_control = classical_size[0]/10, sample_size_bound = classical_size[len(power_list)-1])
                printmd('**Required sample size by Bayesian method:**\n')
                for i in range(len(bayesian_size)):
                    print(f'Power {power_list[i]} : Required sample size for control : {np.ceil(bayesian_size[i])} \t test : {np.ceil(bayesian_size[i]*(1 - control_prop)/control_prop)}')
        if len(arr) == 2:
            printmd('**Required sample size by Classical method:**\n')
            for i in range(len(classical_size)):
                print(f'Power {power_list[i]} : Required sample size for control : {np.ceil(classical_size[i])} \t test : {np.ceil(classical_size[i]*(1 - control_prop)/control_prop)}')
            progressbar.layout.visibility = 'visible'
            bayesian_size = final_samplesize_multiple_power(progressbar, 5000, 1, 1, control_cr, control_cr+detectable_lift, epsilon, control_prop, power_numeric, min_simulation_control = classical_size[0]/10, sample_size_bound = classical_size[len(power_list)-1])
            printmd('**Required sample size by Bayesian method:**\n')
            for i in range(len(bayesian_size)):
                print(f'Power {power_list[i]} : Required sample size for control : {np.ceil(bayesian_size[i])} \t test : {np.ceil(bayesian_size[i]*(1 - control_prop)/control_prop)}')
