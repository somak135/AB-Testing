#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 01:40:23 2021

@author: somak
"""
import numpy as np
import time

start = time.time()
n = 200000
inputs = range(1, n)

def my_function(i):
  mylist = np.random.beta(1, i, size=1000)
  return mylist.mean()

my_function(1)

processed_list = []
for i in inputs:
  x = my_function(i)
  processed_list.append(x)
  
end = time.time()
print(end - start)

#####################################

import multiprocessing
from joblib import Parallel, delayed

def func(n):
    start = time.time()
    num_cores = multiprocessing.cpu_count()
    #n = 20
    inputs = range(1, n)

    def my_function(i):
    mylist = np.random.beta(1, i, size=1000)
    return mylist.mean()

    processed_list = []

    if __name__ == "__main__":
        processed_list = Parallel(n_jobs=num_cores)(delayed(my_function)(i) for i in inputs)
    

    end = time.time()
    print(processed_list)

print(num_cores)
print(end - start)