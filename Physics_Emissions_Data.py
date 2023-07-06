# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:54:32 2023

@author: feher
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#Loading Data into a Pandas Data Frames

Data_file_path = r"C:\Users\feher\OneDrive - Imperial College London\UROP 2023\Blackett_Huxley_Data.xlsx"



Blackett_heating_data_original = pd.read_excel(Data_file_path, sheet_name= 'BLKT(h)')
Blackett_electricity_data_original = pd.read_excel(Data_file_path, sheet_name= 
                                          'BLKT(e)')
Huxley_heating_data_original = pd.read_excel(Data_file_path, sheet_name= 'HXLY(h)')
Huxley_electricity_data_original = pd.read_excel(Data_file_path, sheet_name= 'HXLY(e)')





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


Blackett_heating_data, B_h_d_date = set_first_row_as_labels(Blackett_heating_data_original)
Blackett_electricity_data, B_e_d_date = set_first_row_as_labels(Blackett_electricity_data_original)
Huxley_heating_data, H_h_d_date = set_first_row_as_labels(Huxley_heating_data_original)
Huxley_electricity_data, H_e_d_date = set_first_row_as_labels(Huxley_electricity_data_original) 

#%%


#Creating Data drames for the years seperately:

#find 2018 range:




def set_first_row_as_labels_with_years(df):
    """
    

    Parameters
    ----------
    df : DataFrame
        Insert the data frame you want to modify.

    Returns
    -------
    df : DataFrame
        Returns a DataFrame that only contains trhe numerical data with the date

    """
    # Extract the first row as the labels
    labels = df.iloc[0]

    # Set the labels as the new index
    df = df[1:]   
    df.columns = labels
    
    df = df.drop(['Site Group', 'Site', 'Monitoring Point', 'Type',
                  'Unit of Measurement'], axis = 1, errors='ignore')
    #df = pd.to_datetime(df['Date']).dt.date
    return df

def separating_years(df):
    """
    

    Parameters
    ----------
    df : DataFrame
        The Dataframe you want to modify.

    Returns
    -------
    separate_dfs : list
        Returns a list of DataFrames each corresponding to a calender year.

    """
    B_h_d_wyears = set_first_row_as_labels_with_years(df)
    
    B_h_d_wyears['Date'] = pd.to_datetime(B_h_d_wyears['Date'])
    
    
    # Group the DataFrame based on the 'Condition' column
    desired_years = [2018,2019, 2020, 2021,2022]
    
    # Create separate DataFrames for each group
    separate_dfs = []
    for year in desired_years:
        year_df = B_h_d_wyears[B_h_d_wyears['Date'].dt.year == year]
        
     
        separate_dfs.append(year_df.drop(['Date'], axis =1))
    return separate_dfs

B_h_d_separated = separating_years(Blackett_heating_data_original)
B_e_d_separated = separating_years(Blackett_electricity_data_original)
H_e_d_separated = separating_years(Huxley_electricity_data_original)
H_h_d_separated = separating_years(Huxley_heating_data_original)





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

def time_frame_separate_years(num_columns, sample_size, desired_years, length_of_years):
    separate_dfs = []
    for i, year in enumerate(desired_years):
        length = length_of_years[i] 
        df = pd.DataFrame(index = range(length), columns = range(num_columns))
        for i in range(0, length):
            for j in range(0, num_columns):
                df.at[i,j] = (j*sample_size + i*1440)/(60*24)
       
        
        #year_df.drop(['Date'], axis = 1)
        separate_dfs.append(df)
    return separate_dfs

def dates_by_year(date_list, desired_years):
    year_lists = {}
    for date in date_list:
        year = date.year
        if year not in year_lists:
            year_lists[year] = []
        year_lists[year].append(date)
   
    return year_lists
        
separated_dates = dates_by_year(B_h_d_date, [2018,2019, 2020, 2021,2022])




num_rows, num_columns = Blackett_electricity_data.shape
thirty_min_sample = time_frame(num_rows, num_columns, 30)

length_of_years = [365,365,366, 365,365]
desired_years = [2018,2019, 2020, 2021,2022]
thirty_min_seperate_y = time_frame_separate_years(num_columns, 30, desired_years, length_of_years)
         

B_h_d_column_name = Blackett_heating_data.columns
B_e_d_column_name = Blackett_electricity_data.columns
H_h_d_column_name = Huxley_heating_data.columns
H_e_d_column_name = Huxley_electricity_data.columns
#To write functions to plot, and as the time stamps are the same create a general variable
time_stamps = H_e_d_column_name

time_column_name = thirty_min_sample.columns 

max_B_h_d = Blackett_heating_data.max()



#%%

#Plotting all four datasets for all four years
fig,ax=plt.subplots()

for x_col, y_col in zip( time_column_name,B_h_d_column_name):
    
    ax.plot(thirty_min_sample[x_col],Blackett_heating_data[y_col], 'o',
             label=f'{x_col} vs {y_col}', color = "green")
ax.set_title('Blackett Heating', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)', fontsize = 13)
ax.set_ylabel('Total Heat Energy (kWh)', fontsize = 13)
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
ax.set_ylabel('Total Electrical Energy (kWh)', fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(B_e_d_date, Blackett_electricity_data[0], 'o', color = 'orange')
tick_positions = [date for date in B_e_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)

plt.show()
#%%

fig,ax=plt.subplots()   
for x_col, y_col in zip( time_column_name, H_h_d_column_name):
    ax.plot(thirty_min_sample[x_col],Huxley_heating_data[y_col], 'o',
             label=f'{x_col} vs {y_col}', color = 'Blue')
ax.set_title('Huxley Heating', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)', fontsize = 13)
ax.set_ylabel('Total Heat Energy (kWh)', fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(H_h_d_date, Huxley_heating_data[0], 'o', color = 'blue')
tick_positions = [date for date in H_h_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)


plt.show()
#%%
fig,ax=plt.subplots()
    
for x_col, y_col in zip(time_column_name, H_e_d_column_name):
    ax.plot(thirty_min_sample[x_col],Huxley_electricity_data[y_col], '.',
             label=f'{x_col} vs {y_col}', color = 'purple')
ax.set_title('Huxley Electricity', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)',  fontsize = 13 )
ax.set_ylabel('Total Electrical Energy (kWh)',  fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(H_h_d_date, Huxley_electricity_data[0], '.', color = 'purple')
tick_positions = [date for date in H_e_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)
plt.show()


#%%
#The aim of this next section is to do all these plots for each individual year


def data_ploting_separate_years(list_of_data, date_data, title, y_axis_name, color):
    for i in range(1, len(list_of_data)):
        fig, ax  = plt.subplots()
        df = list_of_data[i]        
        df2 = thirty_min_seperate_y[i]
        dates = separated_dates[2018+i]       
        
        for x_col, y_col in zip(time_column_name, time_stamps):
            ax.plot(df2[x_col],df[y_col], '.',
                     label=f'{x_col} vs {y_col}', color = color)
        ax.set_title(title +" " + str(2018+i), fontsize = 15)
        ax.set_xlabel('Time elapsed from start (Days)',  fontsize = 13 )
        ax.set_ylabel(y_axis_name,  fontsize = 13)
        ax2 = ax.twiny()
        ax2.tick_params(axis='x', which='both', bottom=False, top=True)
        ax2.plot(dates, df[0], '.', color = color)       
        ax2.set_xlabel('Date', fontsize = 13)
        plt.show()

#%%
#PLotting the data sets individually
        
data_ploting_separate_years(B_h_d_separated, B_h_d_date,
                            'Blackett Heating', 'Total Heating(kWh)', 'green')
#%%
data_ploting_separate_years(B_e_d_separated, B_e_d_date,
                            'Blackett Electricity', 'Total Electricity(kWh)', 'orange')
#%%

data_ploting_separate_years(H_h_d_separated, H_h_d_date,
                            'Huxley Heating', 'Total Heating(kWh)', 'blue')

#%%
data_ploting_separate_years(H_e_d_separated, H_e_d_date,
                            'Huxley Heating', 'Total Electricity(kWh)', 'purple')
        



