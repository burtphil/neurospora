# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/burt/.spyder2/.temp.py
"""
### import packages

import numpy as np
import os
import pandas as pd
import xlrd 
### change working directory to raw data

os.chdir('//home//burt//Desktop//Neurospora//all_raw_data')


### read data
test = pd.read_excel('con-data for t12 25-75_ a.xlsx')
