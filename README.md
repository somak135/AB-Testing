# june-2021

## A/B Testing method
* `testing package` folder contains the package and the two main .ipynb files.
    + `design_AB_test.ipynb` finds required sample size for a test.
    + `conduct_AB_test.ipynb` conducts a AB test on an input data set.



* `AB_testing_env.yml` is the environment file with the necessary packages.



* `example_data.csv` is a simulated data set on which the test may be conducted. It has only two columns -- first column is for the variation and second column is the binary 0/1 observations. A data set has to be summarized in this format to run this program. Following is a preview.

| Variation | Observation |
|:---------:|:-----------:|
|  Control  |      0      |
|  Control  |      0      |
| Treatment |      1      |
|  Control  |      1      |
| Treatment |      1      |
| Treatment |      0      |



* `text` folder contains resources.



* `Initial codes` folder contains different .py and .ipynb files created during the entire development process and before summarizing it.


### Brief Steps to run the Notebooks

1. Download the `AB_testing_env.yml` file and create a virtual Python3 environment using the .yml environment file.

2. Activate the newly created environment and open Jupyter notebook. We first talk about how to run the `design_AB_test.ipynb` notebook.
  + Blindly run cell 1.
  + Run cell 2. After running cell 2, a few iPython widgets will be displayed. The required inputs go in here. The _Power/Conclusive probability_ widget and _Method_ supports multiple selection by using ctrl+click or command+click(mac).
  + Run cell 3. A slider widget is shown to specify level of significance and/or expected loss threshold. The preset values are default.
  + Run cell 4. It would show the result. If Bayesian was selected in _Method_ in cell 2, it will take some time to display the result. Following are the examples of cell 4 when the program is running and after it has run.
  
  ![alt text](https://github.com/somak135/AB-Testing/blob/main/text%20%26%20images/design_running.jpeg)
  
  ![alt text](https://github.com/somak135/AB-Testing/blob/main/text%20%26%20images/design_complete.jpeg)
  
3. Activate the newly created environment if not done already. Fire up Jupyter notebook. Now we shall learn how to run the `conduct_AB_test.ipynb` notebook.
  + Run cell 1 blindly.
  + Run cell 2. It would give the button to upload .csv file containing the data set. The format of this .csv file is described above. The `example_data.csv` in the parent folder is an example file you may use to test the tool. Every time you need to change the file, please rerun this cell.
  + Run cell 3. Here you would need to specify the name of your baseline variant and your preferred method to carry out a test.
  + Run cell 4. If your preferred method was classical, you would need to specify the level of significance(default 5%) of the test by adjusting the slider. If your preferred method was Bayesian, you would need to mention the expected loss threshold(default 5%) and expected lift(absolute).
  + Run cell 5. The verdict and the plots with required metrics will be displayed. Following are the results by running a Classical(Two sided) test and a Bayesian test respectively with the `example_data.csv` file.
  
  ![alt text](https://github.com/somak135/AB-Testing/blob/main/text%20%26%20images/classicaltest.png)
  
  ![alt text](https://github.com/somak135/AB-Testing/blob/main/text%20%26%20images/bayesiantest.png)
