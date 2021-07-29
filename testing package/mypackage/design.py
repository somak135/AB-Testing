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

"""
printmd function prints Markdown text.

Parameters:
    string: A valid markdown text string which will be printed.
    color: Color, default is red.
"""
def printmd(string, color='red'):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))


#################################################

"""
classical_1_samplesize_multiple_power function calculates required sample size for a one-sided A/B test.
Null hypothesis: p1 = p2 vs Alternative hypothesis: p1 < p2

Parameters:
    control_cr: The control/baseline converion rate, considered p1
    treatment_cr: The expected treatment conversion rate (baseline conversion rate + nonzero expected lift), considered p2
    control_group: The proportion of control/baseline samples in the entire sample (float between 0 to 1, exclusive)
    level_of_sig: Level of significance for the test
    power_list: A float array containing required power(s) for the test
    
Returns:
    sample_size: Required sample size for the control/baseline group
"""
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

###################################################

"""
classical_2_samplesize_multiple_power function calculates required sample size for a two-sided A/B test.
Null hypothesis: p1 = p2 vs Alternative hypothesis: p1 != p2

Parameters:
    control_cr: The control/baseline converion rate, considered p1
    treatment_cr: The expected treatment conversion rate (baseline conversion rate + nonzero expected lift), considered p2
    control_group: The proportion of control/baseline samples in the entire sample (float between 0 to 1, exclusive)
    level_of_sig: Level of significance for the test
    power_list: A float array containing required power(s) for the test
    
Returns:
    sample_size: Required sample size for the control/baseline group
"""
def classical_2_samplesize_multiple_power(control_cr, treatment_cr, control_group, level_of_sig, power_list):
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
    
#####################################################

"""
calculate_expected_loss function takes two arrays and computes mean(max(array1 - array2, 0)) and mean(max(array2 - array1, 0)).

Parameters:
    control_simulation: First input array, considered array1.
    treatment_simulation: Second input array, considered array2.
    treatment_won: This is the logical array (treatment_simulation > control_simulation) converted to a 0-1 numpy array.
    
Returns:
    np.mean(all_loss_control): This is mean(max(array2 - array1, 0))
    np.mean(all_loss_treatment): This is mean(max(array1 - array2, 0))
"""
def calculate_expected_loss(control_simulation, treatment_simulation, treatment_won, min_difference_delta=0):
    control_simulation = np.array(control_simulation)
    treatment_simulation = np.array(treatment_simulation)
    
    loss_control = (treatment_simulation - min_difference_delta) - control_simulation
    loss_treatment = (control_simulation - min_difference_delta) - treatment_simulation
    
    all_loss_control = treatment_won * loss_control
    all_loss_treatment = (1 - treatment_won) * loss_treatment
    
    return np.mean(all_loss_control), np.mean(all_loss_treatment)

###########################################################

"""
bayesian_sample_size function iteratively simulates two datasets -- one with conversion rate control_cr and another with treatment_cr until Bayesian A/B test can declare treatment to be significantly better than control/baseline.

Parameters:
    prior_alpha: First shape parameter for prior distributions of the conversion rates
    prior_beta: Second shape paramter for prior distributions of the conversion rates
    control_cr: The control/baseline converion rate
    treatment_cr: The expected treatment conversion rate (baseline conversion rate + nonzero expected lift)
    treatment_threshold: A float value between 0 and 1 which works as threshold for the Bayesian test to conclude (lower the value, more precise is the test)
    control_prop: The proportion of control/baseline samples in the entire sample (float between 0 to 1, exclusive)
    min_simulation_control: The minimum size of simulated sample for the control/baseline that must be present for concluding the test (to avoid small sample biases)
    sample_size_bound_control: The maximum size of simulated sample for the control/baseline that will be prsent for concluding the test
    
Returns:
    sample_size_required: The required sample size to conclude the Bayesian test
"""
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
        
        if expected_loss_treatment < treatment_threshold*(treatment_cr - control_cr) and expected_loss_treatment <= treatment_threshold*(expected_loss_control+expected_loss_treatment):
            number_of_treatment_win += 1
            flag = 1
        elif sample_size_control >= sample_size_bound_control:
            flag = 1
    
    sample_size_required = max(sample_size_control, round(sample_size_treatment*control_prop/(1 - control_prop)))
    return sample_size_required

##############################################################

"""
calculate_bayesian_samplesize_control_distbn function calls the bayesian_sample_size function n times. The n returned values are stored in an array which gives the empirical distribution of the required sample size for control/baseline group.

Parameters:
    n: The number of times bayesian_sample_size function is called. Higher the number, more precise is the empirical distribution.
    prior_alpha: First shape parameter for prior distributions of the conversion rates
    prior_beta: Second shape paramter for prior distributions of the conversion rates
    control_cr: The control/baseline converion rate
    treatment_cr: The expected treatment conversion rate (baseline conversion rate + nonzero expected lift)
    treatment_threshold: A float value between 0 and 1 which works as threshold for the Bayesian test to conclude (lower the value, more precise is the test)
    control_prop: The proportion of control/baseline samples in the entire sample (float between 0 to 1, exclusive)
    min_simulation_control: The minimum size of simulated sample for the control/baseline that must be present for concluding the test (to avoid small sample biases)
    sample_size_bound_control: The maximum size of simulated sample for the control/baseline that will be prsent for concluding the test
    
Returns:
    processed_list: The array of length n containing the sample size requirements of n many Bayesian A/B tests with the specified parameters.
"""
def calculate_bayesian_samplesize_control_distbn(n, prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control):
    num_cores = multiprocessing.cpu_count()
    inputs = range(1, n)
    
    processed_list = []
    processed_list = Parallel(n_jobs=num_cores)(delayed(bayesian_sample_size)(prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control) for i in inputs)
    return processed_list

#################################################################

"""
bayesian_samplesize_multiple_power function calls the calculate_bayesian_samplesize_control_distbn function and creates the array of length n and returns specified quantile of the array. It also enables displaying a iPywidget progressbar. 

Parameters:
    progressbar: This is a iPywidget IntProgress bar.
    n: The number of times bayesian_sample_size function is called. Higher the number, more precise is the empirical distribution.
    prior_alpha: First shape parameter for prior distributions of the conversion rates
    prior_beta: Second shape paramter for prior distributions of the conversion rates
    control_cr: The control/baseline converion rate
    treatment_cr: The expected treatment conversion rate (baseline conversion rate + nonzero expected lift)
    treatment_threshold: A float value between 0 and 1 which works as threshold for the Bayesian test to conclude (lower the value, more precise is the test)
    control_prop: The proportion of control/baseline samples in the entire sample (float between 0 to 1, exclusive)
    power_list: A float array containing required power(s) for the test
    min_simulation_control: The minimum size of simulated sample for the control/baseline that must be present for concluding the test (to avoid small sample biases)
    sample_size_bound_control: The maximum size of simulated sample for the control/baseline that will be prsent for concluding the test
    
Returns:
    sample_size_design: Required sample size for the control/baseline group (with specified power(s))
"""
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
    sample_size_design = np.quantile(complete_list, power_list, axis = 0)
    return sample_size_design

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
                description = f'<b>Expected lift (Absolute)</b>',
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
            bayesian_size = bayesian_samplesize_multiple_power(progressbar, 10000, 1, 1, control_cr, control_cr+expected_lift, epsilon, control_prop, power_numeric, ref_size[0]/10, 1.2*ref_size[len(ref_size)-1])
            printmd('**Required sample size for Bayesian test:**\n')
            for i in range(len(bayesian_size)):
                print(f'Power {power_list[i]} : Required sample sizes for control group: {np.ceil(bayesian_size[i])} \t test group: {np.ceil(bayesian_size[i]*(1 - control_prop)/control_prop)}')