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
  + 
