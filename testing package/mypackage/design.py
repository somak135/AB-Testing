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
    """
printmd function prints Markdown text.

Parameters:
    string: A valid markdown text string which will be printed.
    color: Color, default is red.
"""
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))


#################################################

def classical_1_samplesize_multiple_power(control_cr, treatment_cr, control_group, level_of_sig, power_list):
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
    p1 = control_cr
    p2 = treatment_cr
    q1 = 1 - p1
    q2 = 1 - p2
    delta = abs(p2 - p1)
    k = (1 - control_group)/control_group
    pbar = (p1 + k*p2)/(1 + k)
    qbar = 1 - pbar
    
    term1 = np.sqrt(pbar * qbar * (1+1/k)) * norm.ppf(1 - level_of_sig) ##########calculating exact sample size using formula: look at page 404 of ISBN 978-1-305-26892-0 for reference. For one sided test alpha/2 is replaced with alpha.
    sample_size = []
    for i in range(len(power_list)):
        term2 = np.sqrt(p1 * q1 + p2 * q2/k) * norm.ppf(power_list[i]) ##########calculating exact sample size using formula
        sample_size.append(((term1 + term2)**2)/(delta**2))
    
    return sample_size

###################################################

def classical_2_samplesize_multiple_power(control_cr, treatment_cr, control_group, level_of_sig, power_list):
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
    p1 = control_cr
    p2 = treatment_cr
    q1 = 1 - p1
    q2 = 1 - p2
    delta = abs(p2 - p1)
    k = (1 - control_group)/control_group
    pbar = (p1 + k*p2)/(1 + k)
    qbar = 1 - pbar
    
    term1 = np.sqrt(pbar * qbar * (1+1/k)) * norm.ppf(1 - level_of_sig/2) ##########calculating exact sample size using formula: look at page 404 of ISBN 978-1-305-26892-0 for reference.
    sample_size = []
    for i in range(len(power_list)):
        term2 = np.sqrt(p1 * q1 + p2 * q2/k) * norm.ppf(power_list[i]) ##########calculating exact sample size using formula
        sample_size.append(((term1 + term2)**2)/(delta**2))
    
    return sample_size
    
#####################################################

def calculate_expected_loss(control_simulation, treatment_simulation, treatment_won):
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
    control_simulation = np.array(control_simulation)
    treatment_simulation = np.array(treatment_simulation)
    
    loss_control = treatment_simulation - control_simulation
    loss_treatment = control_simulation - treatment_simulation
    
    all_loss_control = treatment_won * loss_control ###turning the negative entries in the loss_control array to 0
    all_loss_treatment = (1 - treatment_won) * loss_treatment ###turning the negative entries in the loss_treatment array to 0
    
    return np.mean(all_loss_control), np.mean(all_loss_treatment)

###########################################################

def bayesian_sample_size(prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control, seed):
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
    seed: seed for simulation
    
Returns:
    sample_size_required: The required sample size to conclude the Bayesian test
"""
    number_of_control_win = 0 ### initialize
    number_of_treatment_win = 0 ### initialize
    flag = 0
    
    ### First step: set starting sample sizes and generate dataset
    np.random.seed(seed)
    sample_size_control = min_simulation_control ### First set control/baseline sample size to min_simulation_control
    sample_size_treatment = np.ceil(min_simulation_control * (1 - control_prop)/control_prop).astype(int) ### Set the treatment sample size accordingly scaled
    control_conversions = np.random.binomial(n = sample_size_control, p = control_cr, size = 1) ### Generate control/baseline samples of size sample_size_control
    treatment_conversions = np.random.binomial(n = sample_size_treatment, p = treatment_cr, size = 1) ### Generate treatment samples of size sample_size_treatment
    
    ### Second step: Iteratively generate dataset until experiment concludes
    
    while flag == 0:
        sample_size_control += 50 ### Increase control/baseline sample size by batchsize of 50
        sample_size_treatment += round(50 * (1 - control_prop)/control_prop) ### Increase treatment sample size by scaling accordingly
        control_conversions += np.random.binomial(n = 50, p = control_cr, size = 1) ### Generate new 50 control/baseline samples
        treatment_conversions += np.random.binomial(n = round(50 * (1 - control_prop)/control_prop), p = treatment_cr, size = 1) ### Generate new treatment samples as many required
        
        control_posterior_simulation = np.random.beta(prior_alpha + control_conversions, prior_beta + sample_size_control - control_conversions, size=5000) ### Monte carlo simulation of size 5000 from obtained posterior distribution of control/baseline conversion rate
        treatment_posterior_simulation = np.random.beta(prior_alpha + treatment_conversions, prior_beta + sample_size_treatment - treatment_conversions, size=5000) ### Monte carlo simulation of size 5000 from obtained posterior distribution of treatment conversion rate
        treatment_won = (treatment_posterior_simulation >= control_posterior_simulation).astype(int)
        
        expected_loss_control, expected_loss_treatment = calculate_expected_loss(control_posterior_simulation, treatment_posterior_simulation, treatment_won) ### Calling the calculate_expected_loss function to find the control/baseline and treatment expected losses
        
        ### Following are the conclusion criterions of a Bayesian test
        if expected_loss_treatment < treatment_threshold*(treatment_cr - control_cr) and expected_loss_treatment <= treatment_threshold*(expected_loss_control+expected_loss_treatment):
            number_of_treatment_win += 1
            flag = 1
        elif sample_size_control >= sample_size_bound_control:
            flag = 1
    
    sample_size_required = max(sample_size_control, round(sample_size_treatment*control_prop/(1 - control_prop))) ### This step to avoid any underestimation due to integer rounding off
    return sample_size_required

##############################################################

def calculate_bayesian_samplesize_control_distbn(n, prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control, seed):
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
    seed: seed for simulation
    
Returns:
    processed_list: The array of length n containing the sample size requirements of n many Bayesian A/B tests with the specified parameters.
"""
    num_cores = multiprocessing.cpu_count()
    inputs = range(1, n)
    
    processed_list = []
    processed_list = Parallel(n_jobs=num_cores)(delayed(bayesian_sample_size)(prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control, (135+seed+i)) for i in inputs) ### 135 here is a random value and may be changed which will change the seed
    return processed_list

#################################################################

def bayesian_samplesize_multiple_power(progressbar, n, prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, power_list, min_simulation_control, sample_size_bound_control):
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
    progressbar.value = 0 ### Initialize the progressbar
    display(progressbar) ### Display the progressbar
    N = 10 ### Performing the entire calculation in 10 parts. Seems optimal, also progressbar will show 1/10-th progress at completion of each part
    k = np.ceil(n/10).astype(int) + 1 ###Determining size of each part
    complete_list = np.array([])
    for i in range(N):
        list1 = calculate_bayesian_samplesize_control_distbn(k, prior_alpha, prior_beta, control_cr, treatment_cr, treatment_threshold, control_prop, min_simulation_control, sample_size_bound_control, (k*i))
        complete_list = np.append(complete_list, list1) ### Append result of each part into complete_list
        progressbar.value += 1 ### Display progress in progressbar
        
    progressbar.layout.visibility = 'hidden' ### Hide progressbar on completion of the process
    sample_size_design = np.quantile(complete_list, power_list, axis = 0) ### Determine the required quantiles
    return sample_size_design

################ WIDGETS AND DISPLAY FUNCTIONS ####################

style = {'description_width': 'initial'} ### All widgets will have this style

control_cr = widgets.BoundedFloatText( ### This is a BoundedFloatText widget to enter the control/baseline conversion rate
            value = 0.2,
            min = 0,
            step = 0.1,
            max = 1,
            description = f'<b>Baseline conversion rate</b>',
            disabled = False,
            style = style
)

expected_lift = widgets.BoundedFloatText( ### This is a BoundedFloatText widget to enter the expected lift
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

epsilon = widgets.SelectionSlider( ### This is a slider widget to choose the treatment_threshold parameter for Bayesian A/B testing. Defaults to 5%.
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
    
level_of_sig = widgets.SelectionSlider( ### This is a slider widget to choose the level of significance parameter for classical A/B testing. Defaults to 5%.
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

control_prop = widgets.Dropdown( ### This is a dropdown widget to select the proportion of control/baseline samples among the two samples combined
                options = values_text,
                value = '50%',
                description = f'<b>Proportion of Control samples</b>',
                style = style
)

power_values = np.arange(75, 100, 1)
power_values_text = []
for i in range(len(power_values)):
    power_values_text.append(power_values[i].astype(str) + '%')

power = widgets.SelectMultiple( ### This is a selection widget to select the power of the test required to be designed. Allows multiple selection.
        options = power_values_text,
        value = ['90%'],
        description = f'<b>Power/Conclusive probability</b>',
        style = style
)

power_label = widgets.Label('ctrl + Click for multiple selection')
power_box = widgets.HBox([power, power_label])

method_choice = widgets.SelectMultiple( ### This is a selection widget to select what kind of test to design. This also allows multiple selection.
                options = ['Classical(One sided)', 'Classical(Two sided)', 'Bayesian'],
                value = ['Classical(Two sided)'],
                description = f'<b>Method</b>',
                style = style
)


Bayesian_loading = widgets.IntProgress( ### This is a progressbar only shown for designing Bayesian test
    value=0,
    min=0,
    max=10,
    description='Bayesian running:',
    bar_style='info',
    style={'bar_color': '#00FF00', 'description_width': 'initial'},
    orientation='horizontal'
)

def threshold_display(arr):
    """
    threshold_display function displays the appropriate widget to select level of significance for designing classical A/B test or/and expected loss threshold to design Bayesian A/B test.
    
    Parameters:
        arr: A string array containing some/all elements of 'Classical(One sided)', 'Classical(Two sided)', 'Bayesian'
    """
    if ('Classical(One sided)' in arr) or ('Classical(Two sided)' in arr):
        printmd('**Enter Level of Significance for Classical calculation:**')
        printmd('A Classical test concludes when the observed p-value is less than the **Level of significance**. Lower the value of the level of significance, more is the confidence in declaring a winner and higher is the sample size requirement. The convension is to set a very low value for this threshold and the default is 5%.', color = 'black')
        display(level_of_sig)
            
    if 'Bayesian' in arr:
        printmd('**Enter Expected Loss Threshold for Bayesian calculation:**')
        printmd('A Bayesian test concludes when (Expected loss of one variant)/(Sum of expected losses of the two variants) is less than the **Expected loss threshold**. Lower the value of the threshold, more is the confidence in declaring a winner and higher is the sample size requirement. The convension is to set a very low value for this threshold and the default is 5%.', color = 'black')
        display(epsilon)
        
def samplesize_calculate(progressbar, arr, control_cr, expected_lift, power_list, control_prop, level_of_sig, epsilon):
    """
    samplesize_calculate function takes all the input values of the required parameters to design a test and puts together required function to finally calculate and display the samplesize requirements of two groups.
    
    Parameters:
        progressbar: This is a iPywidget IntProgress bar.
        arr: A string array containing some/all elements of 'Classical(One sided)', 'Classical(Two sided)', 'Bayesian'
        control_cr: The control/baseline converion rate
        expected_lift: The lift(absolute) expected in conversion rate of treatment
        power_list: A string array of the form ['80%', '90%'] containing the required power(s)
        control_prop: The proportion of control/baseline samples in the entire sample (float between 0 to 1, exclusive)
        level_of_sig: Level of significance for classical test
        epsilon: A float value between 0 and 1 which works as threshold(the treatment_threshold) for the Bayesian test to conclude (lower the value, more precise is the test)
    """
    if expected_lift == 0: ### A test with expected lift 0 cannot be designed
        printmd("**Error : 0% detectable lift is not a valid input for detectable lift.**")
    else:
        power_numeric = []
        for i in range(len(power_list)):
            power_numeric.append(int(power_list[i][:-1])/100) ### Converted the strings in the power_list array to float values
        
        level_of_sig = int(level_of_sig[:-1])/100 ### Convert the level_of_sig widget string value to float
        epsilon = int(epsilon[:-1])/100 ### Convert the epsilon widget string value to float
        ref_size = classical_2_samplesize_multiple_power(control_cr, control_cr+expected_lift, control_prop, level_of_sig, power_numeric) ### Calculate the classical sample size requirement which is to be used for setting minimum sample size requirement and maximum sample size bound for Bayesian test designing
        
        if 'Classical(One sided)' in arr:
            classical_size = classical_1_samplesize_multiple_power(control_cr, control_cr+expected_lift, control_prop, level_of_sig, power_numeric) ### Calculate sample size for one sided classical test 
            printmd('**Required sample size for one sided Classical test:**\n')
            for i in range(len(classical_size)):
                print(f'Power {power_list[i]} : Required sample sizes for control group: {np.ceil(classical_size[i])} \t test group: {np.ceil(classical_size[i]*(1 - control_prop)/control_prop)}') ###printed
                
        if 'Classical(Two sided)' in arr:
            classical_size = classical_2_samplesize_multiple_power(control_cr, control_cr+expected_lift, control_prop, level_of_sig, power_numeric) ### Calculate sample size for two sided classical test 
            printmd('**Required sample size for two sided Classical test:**\n')
            for i in range(len(classical_size)):
                print(f'Power {power_list[i]} : Required sample sizes for control group: {np.ceil(classical_size[i])} \t test group: {np.ceil(classical_size[i]*(1 - control_prop)/control_prop)}') ### printed
        
        if 'Bayesian' in arr:
            progressbar.layout.visibility = 'visible'
            bayesian_size = bayesian_samplesize_multiple_power(progressbar, 10000, 1, 1, control_cr, control_cr+expected_lift, epsilon, control_prop, power_numeric, ref_size[0]/10, 1.2*ref_size[len(ref_size)-1]) ### Calculate sample size for Bayesian test 
            printmd('**Required sample size for Bayesian test:**\n')
            for i in range(len(bayesian_size)):
                print(f'Power {power_list[i]} : Required sample sizes for control group: {np.ceil(bayesian_size[i])} \t test group: {np.ceil(bayesian_size[i]*(1 - control_prop)/control_prop)}') ### printed