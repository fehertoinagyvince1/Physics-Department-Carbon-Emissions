# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:54:32 2023

@author: feher
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Loading Data into a Pandas Data Frames

Data_file_path = r"C:\Users\feher\OneDrive - Imperial College London\UROP 2023\Copy of 230509 Huxley & Blackett Data V1.0.xlsx"

Blackett_heating_data = pd.read_excel(Data_file_path, sheet_name= 'BLKT(h)')
Blackett_electricity_data = pd.read_excel(Data_file_path, sheet_name= 'BLKT(e)')
Huxley_heating_data = pd.read_excel(Data_file_path, sheet_name= 'HXLY(h)')
Huxley_electricity_data = pd.read_excel(Data_file_path, sheet_name= 'HXLY(e)')



def set_first_row_as_labels(df):
    # Extract the first row as the labels
    labels = df.iloc[0]

    # Set the labels as the new index
    df = df[1:]   
    df.columns = labels
    df = df.drop(['Site Group', 'Site', 'Monitoring Point', 'Type', 'Unit of Measurement', 'Date'], axis = 1)
    return df

Blackett_heating_data = set_first_row_as_labels(Blackett_heating_data)
Blackett_electricity_data = set_first_row_as_labels(Blackett_electricity_data)
Huxley_heating_data = set_first_row_as_labels(Huxley_heating_data)
Huxley_electricity_data = set_first_row_as_labels(Huxley_electricity_data) 

#%%%




