#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:10:20 2021

@author: somak
"""

import numpy as np
import matplotlib.pyplot as plt

power = [0.80, 0.85, 0.90, 0.95]
sample_size_classical = [25255, 28914, 33868, 41932]
sample_size_bayesian = [7580, 8840, 10700, 14420]

plt.plot(power, sample_size_classical, 'o--', color = 'r', label = 'Classical')
plt.plot(power, sample_size_bayesian, 'o--', color = 'g', label = 'Bayesian')
plt.legend(loc = 'upper left')
plt.title('Required sample size vs Power : Illustrative chart')
plt.xlabel('Power of test')
plt.ylabel('Required sample size')
plt.savefig('sample_size_comparison', dpi = 800)
plt.close()



