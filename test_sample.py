#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:58:54 2017

@author: burt
"""

import pandas as pd


df = pd.DataFrame({'Data': [10, 20, 30, 20, 15, 30, 45]})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Get the xlsxwriter objects from the dataframe writer object.
workbook  = writer.book
worksheet = writer.sheets['Sheet1']

# Light red fill with dark red text.
format1 = workbook.add_format({'font_color': '#9C0006'})


worksheet.conditional_format('B2:AQ18', {'type':     'cell',
                                    'criteria': 'greater than',
                                    'value':     20,
                                    'format':    format1})

# Add a format. Green fill with dark green text.
format2 = workbook.add_format({'bg_color': '#C6EFCE',
                               'font_color': '#006100'})


worksheet.conditional_format('B2:AQ18', {'type':     'cell',
                                    'criteria': 'greater than',
                                    'value':     20,
                                    'format':    format1})

workbook.close()