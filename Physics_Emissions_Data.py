# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:54:32 2023

@author: feher
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Loading Data into a Pandas Data Frames

Data_file_path = r"C:\Users\feher\OneDrive - Imperial College London\UROP 2023\Huxley_Blackett_Data.xlsx"

Blackett_heating_data = pd.read_excel(Data_file_path, sheet_name= 'BLKT(h)')
Blackett_electricity_data = pd.read_excel(Data_file_path, sheet_name= 
                                          'BLKT(e)')
Huxley_heating_data = pd.read_excel(Data_file_path, sheet_name= 'HXLY(h)')
Huxley_electricity_data = pd.read_excel(Data_file_path, sheet_name= 'HXLY(e)')



def set_first_row_as_labels(df):
    """
    

    Parameters
    ----------
    df : DataFrame
        Insert the data frame you want to modify.

    Returns
    -------
    df : DataFrame
        Returns a DataFrame that has the time as the column name and gets rid 
        of the non-numerical data.

    """
    # Extract the first row as the labels
    labels = df.iloc[0]

    # Set the labels as the new index
    df = df[1:]   
    df.columns = labels
    df_date = pd.to_datetime(df['Date']).dt.date
    df = df.drop(['Site Group', 'Site', 'Monitoring Point', 'Type',
                  'Unit of Measurement', 'Date'], axis = 1)
    return df, df_date


Blackett_heating_data, B_h_d_date = set_first_row_as_labels(Blackett_heating_data)
Blackett_electricity_data, B_e_d_date = set_first_row_as_labels(Blackett_electricity_data)
Huxley_heating_data, H_h_d_date = set_first_row_as_labels(Huxley_heating_data)
Huxley_electricity_data, H_e__d_date = set_first_row_as_labels(Huxley_electricity_data) 



#%%%
#To get an idea of the outliers, I will plot every value linearly.
#To do this we will need to assign every point in the data frame to a time value

#Creating the time Data Frame, set first entry as zero time

     


def time_frame(num_rows, num_columns, sample_size):
    df = pd.DataFrame(index = range(num_rows), columns = range(num_columns))
    for i in range(0, num_rows):
        for j in range(0, num_columns):
            df.at[i,j] = (j*sample_size + i*1440)/(60*24)
    return df

num_rows, num_columns = Blackett_electricity_data.shape
thirty_min_sample = time_frame(num_rows, num_columns, 30)
            

B_h_d_column_name = Blackett_heating_data.columns
B_e_d_column_name = Blackett_electricity_data.columns
H_h_d_column_name = Huxley_heating_data.columns
H_e_d_column_name = Huxley_electricity_data.columns

time_column_name = thirty_min_sample.columns 

print(Blackett_heating_data[0])

#%%

fig,ax=plt.subplots()

for x_col, y_col in zip( time_column_name,B_h_d_column_name):
    
    ax.plot(thirty_min_sample[x_col],Blackett_heating_data[y_col], 'o',
             label=f'{x_col} vs {y_col}', color = "green")
ax.set_title('Blackett Heating', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)', fontsize = 13)
ax.set_ylabel('Total Heat Energy (kwH)', fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(B_h_d_date, Blackett_heating_data[0],'o', color = 'green')
tick_positions = [date for date in B_h_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)
#

plt.show()

#%%
fig,ax=plt.subplots()
for x_col, y_col in zip(time_column_name,B_e_d_column_name):
    ax.plot(thirty_min_sample[x_col],Blackett_electricity_data[y_col], 'o',
             label=f'{x_col} vs {y_col}', color = 'orange')
ax.set_title('Blackett Electricity', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)',fontsize = 13)
ax.set_ylabel('Total Electrical Energy (kwH)', fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(B_e_d_date, Blackett_electricity_data[0], 'o', color = 'orange')
tick_positions = [date for date in B_e_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)

plt.show()
#%%

    
for x_col, y_col in zip( time_column_name, H_h_d_column_name):
    plt.plot(thirty_min_sample[x_col],Huxley_heating_data[y_col], 'o',
             label=f'{x_col} vs {y_col}')
plt.title('Huxley Hheating', fontsize = 15)
plt.xlabel('Time elapsed from start (Days)')
plt.ylabel('Total Heat Energy (kwH)')
plt.show()
    
for x_col, y_col in zip(time_column_name, H_e_d_column_name):
    plt.plot(thirty_min_sample[x_col],Huxley_electricity_data[y_col], 'o',
             label=f'{x_col} vs {y_col}')
plt.title('Huxley Electricity', fontsize = 15)
plt.xlabel('Time elapsed from start (Days)')
plt.ylabel('Total Electrical Energy (kwH)')
plt.show()





