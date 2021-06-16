#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 21:06:23 2021

@author: somak
"""
import time


## AFTER RUNNING 1.py

start = time.time()
print(calculate_reqd_samplesize_distbn(1000, 1, 1, 0.2, 0.21, 0.0005, [0.9, 0.98], 2525, 25255))
end = time.time()
end - start


## AFTER RUNNING 2.py

start = time.time()
print(simulation_study(1000, 1390, 0.2, 0.19, 0.0005, 0.05))
end = time.time()
end - start

