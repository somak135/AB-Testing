import ipywidgets as widgets
from IPython.display import Markdown, display
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
#from scipy.stats import norm, t
import matplotlib.pyplot as plt
import json
import decimal
decimal.getcontext().prec = 4
import multiprocessing
from joblib import Parallel, delayed
import io
from scipy.special import btdtri
from tabulate import tabulate
import random


def printmd(string, color='red'):
    """
    printmd function prints Markdown text.

    Parameters:
        string: A valid markdown text string which will be printed.
        color: Color, default is red.
    """
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

########################################################

def upload_file():
    """
    upload_file function creates and displays a iPython widget clicking on which a .csv file can be uploaded.

    Returns:
        uploader: Returns the widget.
    """
    printmd('**Upload the .csv data file with control/treatment in first column and binary observations in second column:**')
    uploader = widgets.FileUpload(accept = '.csv', multiple = False)
    display(uploader)
    return uploader

#########################################################

def read_file(uploader):
    """
    read_file function reads the file uploaded using upload_file iPython widget and converts it into a Pandas object.

    Parameters:
        uploader: A nonempty iPython FileUpload widget used for .csv file
    
    Returns:
        df: The Pandas object created from the .csv file
        name_list: A string array containing the first two distinct names in the first column of df
    """
    input_file = list(uploader.value.values())[0]
    content = input_file['content']
    content = io.StringIO(content.decode('utf-8'))
    df = pd.read_csv(content)
    name1 = df[df.columns[0]].unique()[0]  ## name of variation1
    name2 = df[df.columns[0]].unique()[1]  ## name of variation2
    name_list = [name1, name2]
    return df, name_list

###########################################################

def do_classical1_test(df, level_of_sig, name_list, baseline):
    """
do_classical1_test function performs a one-sided A/B test to compare the means of two Bernouli datasets. In the end, plots the two means and corresponding 95% confidence intervals using plt.bar from matplotlib. Also the p-value is printed and depending on the level of significance the result of the test is printed. 
Null hypothesis: p1 = p2 vs Alternative hypothesis: p1 < p2

Parameters:
    df: A dataset with two columns such that first column contains the name of corresponding group. e.g.:
        Name     Observation
        Control   0
        Control   0
        Control   1
        Treatment 0
        Control   1
        Treatment 1
    level_of_sig: The level of significance for the statistical test.
    name_list: The name of the two groups whose mean are to be compared.
    baseline: The name of the baseline group. Its mean is taken to be p1. 
    """
    str1 = df.columns[0]  ## column name of variations
    str2 = df.columns[1]  ## column name of observations
    
    variation1_name = baseline ## name of variation 1
    name_list.remove(baseline)
    variation2_name =  name_list[0] ## name of variation 2
    
    dataset1 = df[df[str1] == variation1_name] ### extract the dataset for variation 1
    dataset2 = df[df[str1] == variation2_name] ### extract the dataset for variation 2
    
    ci_dataset1 = np.sqrt(dataset1[str2].mean()*(1 - dataset1[str2].mean())/len(dataset1[str2]))*ss.norm.ppf(1-level_of_sig/2) ### compute width of two sided 100*(1-level_of_sig)% confidence interval for variation 1 conversion rate
    ci_dataset2 = np.sqrt(dataset1[str2].mean()*(1 - dataset1[str2].mean())/len(dataset2[str2]))*ss.norm.ppf(1-level_of_sig/2 ) ### compute width of two sided 100*(1-level_of_sig)% confidence interval for variation 2 conversion rate
    
    p1 = plt.bar(0, dataset1[str2].mean(), color = 'red', edgecolor = 'black', yerr = ci_dataset1, capsize = 15, label = variation1_name, width = 0.2) ### plot bar diagram showing the avg. conversion rate and the confidence intervals for variation 1 
    p2 = plt.bar(0.4, dataset2[str2].mean(), color = 'green', edgecolor = 'black', yerr = ci_dataset2, capsize = 15, label = variation2_name, width = 0.2) ### plot bar diagram showing the avg. conversion rate and the confidence intervals for variation 2
    
    plt.legend(handles=[p1, p2], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([0, 0.4], [variation1_name, variation2_name])
    plt.show()
    
    t_stat, p_val = ss.ttest_ind(dataset1[str2], dataset2[str2], alternative = 'less') ### Carry out the one-sided ttest comparing the conversion rates
    
    ### Following are the criterions to conclude with appropriate level of significance
    if p_val <= level_of_sig:
        printmd(f'**The variation \'{variation2_name}\' has significantly higher conversion rate compared to the the \'{variation1_name}\'. The p value of the test is {p_val:.4f}.**', color = 'green')

    if p_val > level_of_sig:
        printmd(f'**The test is inconclusive. P-value is {p_val: .4f}.**')

##############################################################

def do_classical2_test(df, level_of_sig, name_list, baseline):
    """
do_classical2_test function performs a two-sided A/B test to compare the means of two Bernouli datasets. In the end, plots the two means and corresponding 95% confidence intervals using plt.bar from matplotlib. Also the p-value is printed and depending on the level of significance the result of the test is printed. 
Null hypothesis: p1 = p2 vs Alternative hypothesis: p1 != p2 

Parameters:
    df: A dataset with two columns such that first column contains the name of corresponding group. e.g.:
        Name     Observation
        Control   0
        Control   0
        Control   1
        Treatment 0
        Control   1
        Treatment 1
    level_of_sig: The level of significance for the statistical test.
    name_list: The name of the two groups whose mean are to be compared.
    baseline: The name of the baseline group. Its mean is taken to be p1.    
"""
    str1 = df.columns[0]  ## column name of variations
    str2 = df.columns[1]  ## column name of observations
    
    variation1_name = baseline ## name of variation 1
    name_list.remove(baseline)
    variation2_name =  name_list[0] ## name of variation 2
    
    dataset1 = df[df[str1] == variation1_name] ### extract the dataset for variation 1
    dataset2 = df[df[str1] == variation2_name] ### extract the dataset for variation 2
    
    ci_dataset1 = np.sqrt(dataset1[str2].mean()*(1 - dataset1[str2].mean())/len(dataset1[str2]))*ss.norm.ppf(1-level_of_sig/2) ### compute width of two sided 100*(1-level_of_sig)% confidence interval for variation 1 conversion rate
    ci_dataset2 = np.sqrt(dataset1[str2].mean()*(1 - dataset1[str2].mean())/len(dataset2[str2]))*ss.norm.ppf(1-level_of_sig/2 ) ### compute width of two sided 100*(1-level_of_sig)% confidence interval for variation 2 conversion rate
    
    p1 = plt.bar(0, dataset1[str2].mean(), color = 'red', edgecolor = 'black', yerr = ci_dataset1, capsize = 15, label = variation1_name, width = 0.2) ### plot bar diagram showing the avg. conversion rate and the confidence intervals for variation 1 
    p2 = plt.bar(0.4, dataset2[str2].mean(), color = 'green', edgecolor = 'black', yerr = ci_dataset2, capsize = 15, label = variation2_name, width = 0.2) ### plot bar diagram showing the avg. conversion rate and the confidence intervals for variation 2
    
    plt.legend(handles=[p1, p2], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([0, 0.4], [variation1_name, variation2_name])
    plt.show()
    
    t_stat, p_val = ss.ttest_ind(dataset1[str2], dataset2[str2]) ### Carry out the two-sided ttest comparing the conversion rates
    
    ### Following are the criterions to conclude with appropriate level of significance
    if t_stat<=0 and p_val <= level_of_sig:
        printmd(f'**The variation \'{variation2_name}\' has significantly higher conversion rate compared to the the \'{variation1_name}\'. The p value of the test is {p_val:.4f}.**', color = 'green')
        
    if t_stat>=0 and p_val <= level_of_sig:
        printmd(f'**The variation \'{variation1_name}\' has significantly higher conversion rate compared to the \'{variation2_name}\'. The p value of the test is {p_val:.4f}.**', color = 'red')
        
    if p_val > level_of_sig:
        printmd(f'**The test is inconclusive. P-value is {p_val: .4f}.**')

####################################################################

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
    
    loss_control = (treatment_simulation) - control_simulation
    loss_treatment = (control_simulation) - treatment_simulation
    
    all_loss_control = treatment_won * loss_control ###turning the negative entries in the loss_control array to 0
    all_loss_treatment = (1 - treatment_won) * loss_treatment ###turning the negative entries in the loss_treatment array to 0
    
    return np.mean(all_loss_control), np.mean(all_loss_treatment)

######################################################################

def do_bayesian_test(df, eps, exp_lift, name_list, baseline):
    """
do_bayesian_test function performs a Bayesian A/B test. Test concludes when one variation has expected loss below a threshold and another variation have expected loss above a threshold. Finally returns useful metrics in a table and plots the posterior distributions with shaded 95% confidence intervals.

Parameters: 
    df: A dataset with two columns such that first column contains the name of corresponding group. e.g.:
        Name     Observation
        Control   0
        Control   0
        Control   1
        Treatment 0
        Control   1
        Treatment 1
    eps: For test to conclude, winner need to have expected loss below eps, loser need to have expected loss above (1-eps)
    exp_lift: Expected difference between the means of the two groups.
    name_list: The name of the two groups.
    baseline: The name of the baseline group. 
"""
    str1 = df.columns[0] ## column name of variations
    str2 = df.columns[1] ## column name of observations
    
    variation1_name = baseline ## name of variation 1
    name_list.remove(baseline)
    variation2_name = name_list[0] ## name of variation 2
    
    dataset1 = df[df[str1] == variation1_name] ### extract the dataset for variation 1
    dataset2 = df[df[str1] == variation2_name] ### extract the dataset for variation 2
    
    variation1_data = np.array(dataset1[str2]) ### from dataset 1, extract just the observation column
    variation2_data = np.array(dataset2[str2]) ### from dataset 2, extract just the observation column
    variation1_conversions = sum(variation1_data)
    variation2_conversions = sum(variation2_data)
    variation1_sample_size = len(variation1_data)
    variation2_sample_size = len(variation2_data)
    
    variation1_expected_cr = variation1_data.mean() ## to be reported
    variation2_expected_cr = variation2_data.mean() ## to be reported
    variation1_cr_lci = btdtri(1+variation1_conversions, 1+variation1_sample_size - variation1_conversions, 0.025) ### Lower end of a 95% confidence interval of variation 1 conversion rate
    variation1_cr_uci = btdtri(1+variation1_conversions, 1+variation1_sample_size - variation1_conversions, 0.975) ### Higher end of a 95% confidence interval of variation 1 conversion rate
    variation2_cr_lci = btdtri(1+variation2_conversions, 1+variation2_sample_size - variation2_conversions, 0.025) ### Lower end of a 95% confidence interval of variation 2 conversion rate
    variation2_cr_uci = btdtri(1+variation2_conversions, 1+variation2_sample_size - variation2_conversions, 0.975) ### Higher end of a 95% confidence interval of variation 2 conversion rate
    
    np.random.seed(135)
    variation1_cr_samples = np.random.beta(1+variation1_conversions, 1+variation1_sample_size-variation1_conversions, size=1000000) ### Monte carlo simulation of size 10^6 from obtained posterior distribution of variation 1 conversion rate
    variation2_cr_samples = np.random.beta(1+variation2_conversions, 1+variation2_sample_size-variation2_conversions, size=1000000) ### Monte carlo simulation of size 10^6 from obtained posterior distribution of variation 2 conversion rate
    
    variation2_lift = ((variation2_cr_samples - variation1_cr_samples)/variation1_cr_samples)
    variation2_exp_lift = variation2_lift.mean() ##to be reported as the expected improvement
    
    variation2_won = (variation2_cr_samples >= variation1_cr_samples).astype(int)
    variation1_exp_loss, variation2_exp_loss = calculate_expected_loss(variation1_cr_samples, variation2_cr_samples, variation2_won) ### Calling the calculate_expected_loss function to find the variation 1 and variation 2 expected losses
    
    ### Create the table which will be displayed to user
    report_table = tabulate([['Variation', 'Avg. conversion', 'Avg. loss w.r.t. baseline(relative)', 'Expected improvement(relative)'], [None, None, None, None],
                   [f'{variation1_name}', f'{variation1_expected_cr*100:.3f}%', '(baseline)', '(baseline)'],
                   [f'{variation2_name}', f'{variation2_expected_cr*100:.3f}%', f'{variation2_exp_loss/variation1_exp_loss*100:.3f}%', f'{variation2_exp_lift*100:.3f}%']], numalign = "right", stralign = "center")
    
    
    ### conclusion criterions for a Bayesian A/B test
    if variation1_exp_loss < eps*(exp_lift) and variation1_exp_loss < eps*(variation1_exp_loss+variation2_exp_loss) and variation2_exp_loss > (1-eps)*(variation1_exp_loss+variation2_exp_loss):
        result = f'{variation1_name} has significantly higher conversion rate.'
    elif variation2_exp_loss < eps*(exp_lift) and variation2_exp_loss < eps*(variation1_exp_loss+variation2_exp_loss) and variation1_exp_loss > (1-eps)*(variation1_exp_loss+variation2_exp_loss):
        result = f'{variation2_name} has significantly higher conversion rate. Expect a relative improvement of {variation2_exp_lift*100:.3f}% over {variation1_name}.'
    else:
        result = f'The test is inconclusive.'
    
    ### Print the verdict
    printmd(f'**Avg. loss w.r.t. baseline(relative)**: When the reported value of this metric is k%, if you risk losing 100 units by sticking to baseline, you would risk losing k units by implementing {variation2_name}.', color = 'black')
    print(report_table)
    printmd(f'**{result}**', color = 'black')
    
    ### posterior plots
    leftlim = min(min(variation1_expected_cr, variation2_expected_cr) - 2*abs(variation1_expected_cr - variation2_expected_cr), min(variation1_cr_lci, variation2_cr_lci)) ### creating the left end of the x axis
    rightlim = max(max(variation1_expected_cr, variation2_expected_cr) + 2*abs(variation1_expected_cr - variation2_expected_cr), max(variation1_cr_uci, variation2_cr_uci)) ### creating the right end of the x axis
    x_axis = np.arange(leftlim, rightlim, 0.0001)
    posterior_plot, ax = plt.subplots(1)
    plt.plot(x_axis, ss.beta.pdf(x_axis, 1+variation1_conversions, 1+variation1_sample_size - variation1_conversions), color = 'r', label = variation1_name) ### plot the variation 1 posterior density
    plt.plot(x_axis, ss.beta.pdf(x_axis, 1+variation2_conversions, 1+variation2_sample_size - variation2_conversions), color = 'g', label = variation2_name) ### plot the variation 2 posterior density
    plt.axvspan(variation1_cr_lci, variation1_cr_uci, alpha = 0.3, color = 'red') ### shade confidence region of variation 1
    plt.axvspan(variation2_cr_lci, variation2_cr_uci, alpha = 0.3, color = 'green') ### shade confidence region of variation 2
    plt.axvline(x = variation1_expected_cr, color = 'red', linestyle = ':') ## Dotted line at the mean of the posterior density of variation 1 conversion rate
    plt.axvline(x = variation2_expected_cr, color = 'green', linestyle = ':') ## Dotted line at the mean of the posterior density of variation 2 conversion rate
    plt.legend()
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel('Posterior Density')
    ax.set_xlabel('Conversion rates')
    plt.title('Conversion rates with shaded 95% credible intervals')

    
################ WIDGETS AND DISPLAY FUNCTIONS ####################

style = {'description_width': 'initial'} ### All widgets will have this style

def do_test(df, level_of_sig, epsilon, expected_lift, name_list, baseline, string):
    """
    do_test function takes the dataset, specified method choice and the specified parameters. It puts together the appropriate functions and performs the test and displays result.
    
    Parameters:
        df: A dataset with two columns such that first column contains the name of corresponding group. e.g.:
        Name     Observation
        Control   0
        Control   0
        Control   1
        Treatment 0
        Control   1
        Treatment 1
        level_of_sig: The level of significance for the classical test
        epsilon: For Bayesian test to conclude, winner need to have expected loss below eps, loser need to have expected loss above (1-eps)
        expected_lift: (Absolute) lift expected in the conversion rate of non-baseline variation
        name_list: The name of the two groups whose mean are to be compared.
        baseline: The name of the baseline group. Its mean is taken to be p1. 
        string: A string which is either 'Classical(One sided)', 'Classical(Two sided)' or 'Bayesian' i.e. specifies the desired method
    """
    level_of_sig = int(level_of_sig[:-1])/100
    epsilon = int(epsilon[:-1])/100
    
    if string == 'Classical(One sided)':
        do_classical1_test(df, level_of_sig, name_list, baseline)
        
    if string == 'Classical(Two sided)':
        do_classical2_test(df, level_of_sig, name_list, baseline)
        
    if string == 'Bayesian':
        do_bayesian_test(df, epsilon, expected_lift, name_list, baseline)
    
method_choice = widgets.Select(
                options = ['Classical(One sided)', 'Classical(Two sided)', 'Bayesian'],
                value = 'Classical(Two sided)',
                description = f'<b>Method</b>',
                style = style
)

level_of_sig_values = np.arange(1, 11, 1)
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
#eps_label = widgets.Label()
#eps_box = widgets.HBox([epsilon, eps_label])
#mylink = widgets.jslink((epsilon, 'value'), (eps_label, 'value'))

blank_list = []
ask_baseline = widgets.Select( ### This is a selection widget which displayes the name of the two variations in the input dataset and asks to choose the baseline one.
        options = blank_list,
        description = f'<b>Choose the baseline variation<b>',
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

def display_ask_baseline_widget(widget, name_list):
    """
    display_ask_baseline_widget function feeds the widget the two names of the variations from which user would choose the baseline.
    
    Parameters:
        widget: The iPython Selection widget which would be displayed
        name_list: A string array containing the two names of the variations
    """
    widget.options = name_list
    widget.value = name_list[0]
    display(widget)

def threshold_display(arr):
    """
    threshold_display function displays the appropriate widget to select level of significance for classical A/B test or expected loss threshold for Bayesian A/B test.
    
    Parameters:
        arr: A string which is either 'Classical(One sided)', 'Classical(Two sided)' or 'Bayesian' i.e. specifies the desired method
    """
    if arr == 'Classical(One sided)' or arr == 'Classical(Two sided)':
        printmd('**Enter Level of Significance:**')
        printmd('A Classical test concludes when the observed p-value is less than the **Level of significance**. Lower the value of the level of significance, more is the confidence in declaring a winner. The convension is to set a very low value for this threshold and the default is 5%.', color = 'black')
        display(level_of_sig)
            
    if arr == 'Bayesian':
        printmd('**Enter expected loss threshold:**')
        printmd('A Bayesian test concludes when (Expected loss of one variant)/(Expected lift) is less than the **Expected loss threshold**. Lower the value of the threshold, more is the confidence in declaring a winner. The convension is to set a very low value for this threshold and the default is 5%.', color = 'black')
        display(epsilon); display(expected_lift)