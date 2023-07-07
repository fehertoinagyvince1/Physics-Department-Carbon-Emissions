# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:54:32 2023

@author: feher"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Loading Data into a Pandas Data Frames

Data_file_path = r"C:\Users\feher\OneDrive - Imperial College London\UROP 2023\Blackett_Huxley_Data.xlsx"



Blackett_heating_data_original = pd.read_excel(Data_file_path, 
                                               sheet_name= 'BLKT(h)')
Blackett_electricity_data_original = pd.read_excel(Data_file_path, sheet_name= 
                                          'BLKT(e)')
Huxley_heating_data_original = pd.read_excel(Data_file_path, 
                                             sheet_name= 'HXLY(h)')
Huxley_electricity_data_original = pd.read_excel(Data_file_path, sheet_name= 'HXLY(e)')



def set_first_row_as_labels(df):
    """ This function was created to make the column labels the time stamps,
    and drop non-numerical data, while keeping series of date-stamps

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


#Creating the DataFrames that have the desired column labels

Blackett_heating_data, B_h_d_date = set_first_row_as_labels(Blackett_heating_data_original)
Blackett_electricity_data, B_e_d_date = set_first_row_as_labels(Blackett_electricity_data_original)
Huxley_heating_data, H_h_d_date = set_first_row_as_labels(Huxley_heating_data_original)
Huxley_electricity_data, H_e_d_date = set_first_row_as_labels(Huxley_electricity_data_original) 

#%%


def set_first_row_as_labels_with_years(df):
    """This function is the same as the set_rows_as_labels, without getting
    rid of the dates. This is required for the next function, separating_years,
    as this will be the id that separates the original DataFrame into the
    separate years. The date column is then dropped, as it is not useful 
    afterwards.
    

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


"""Creating DataFrames for the years seperately"""

def separating_years(df):
    """This function was created to separate the DataFrames into years.
    I believe this makes accessing only the data for that year a lot simpler 
    and quicker.     

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
    
#Changing the Date given into datetime format, which enables a lot of functionality:
    B_h_d_wyears['Date'] = pd.to_datetime(B_h_d_wyears['Date'])    
    
    # Group the DataFrame based on the 'Condition' column
    desired_years = [2018,2019, 2020, 2021,2022]
    
    # Create separate DataFrames for each group
    separate_dfs = []
    for year in desired_years:
        year_df = B_h_d_wyears[B_h_d_wyears['Date'].dt.year == year]
        
     
        separate_dfs.append(year_df.drop(['Date'], axis =1))
    return separate_dfs

#Creating the DataFrames separated into years

B_h_d_separated = separating_years(Blackett_heating_data_original)
B_e_d_separated = separating_years(Blackett_electricity_data_original)
H_e_d_separated = separating_years(Huxley_electricity_data_original)
H_h_d_separated = separating_years(Huxley_heating_data_original)





#%%%
"""To get an idea of the outliers, I will plot every value against the time 
elapsed in days.To do this we will need to assign every point in the data frame 
to a day"""

#Creating a Data Frame that represents the time elapsed, set first entry as zero time    



def time_frame(num_rows, num_columns, sample_size):
    """
    

    Parameters
    ----------
    num_rows : int
        The number of rows to be created.  
    num_columns : int
        The number of samples taken in a day. (Alternatively could also calculate
                                               from sample_size)
    sample_size : float
        The number of minutes passed between subsequent measurements.

    Returns
    -------
    df : DataFrame
        A DataFrame that contains information about the time elapsed from the 
        start of the data collection.

    """
    df = pd.DataFrame(index = range(num_rows), columns = range(num_columns))
    for i in range(0, num_rows):
        for j in range(0, num_columns):
            df.at[i,j] = (j*sample_size + i*1440)/(60*24)
    return df

def time_frame_separate_years(num_columns, sample_size, desired_years, 
                              length_of_years):
    """
    

    Parameters
    ----------
    num_columns :int
            The number of samples taken in a day. (Alternatively could also calculate
                                           from sample_size)
    sample_size : float     
            The number of minutes passed between subsequent measurements.
    desired_years : list
            A list of the years we are examining
    length_of_years : list
            A list of the length of the desired_years

    Returns
    -------
    separate_dfs : list
        A list of DataFrames that has time entries for that specific year in 
        days.

    """
    separate_dfs = []
    for i, year in enumerate(desired_years):
        length = length_of_years[i] 
        df = time_frame(length, num_columns, sample_size)   
        separate_dfs.append(df)
    return separate_dfs

def dates_by_year(date_list):
    """This function was created to extract the years analysed and find the
    length of the year from the data   

    Parameters
    ----------
    date_list : list of datetime.datetime objects
        List of all the dates in the dataset.  

    Returns
    -------
    year_lists : dict containg lists
        Returns a dictionary of lists with the key being the years, and the 
        entries of the list being the dates

    """
    year_lists = {}
    for date in date_list:
        year = date.year
        if year not in year_lists:
            year_lists[year] = []
        year_lists[year].append(date)
   
    return year_lists



#%%

#Creating a DataFrame representing the time for the whole DataFrame
num_rows, num_columns = Blackett_electricity_data.shape
thirty_min_sample = time_frame(num_rows, num_columns, 30)         

B_h_d_column_name = Blackett_heating_data.columns
B_e_d_column_name = Blackett_electricity_data.columns
H_h_d_column_name = Huxley_heating_data.columns
H_e_d_column_name = Huxley_electricity_data.columns
#To write functions to plot, and as the time stamps are the same create a general variable
time_stamps = H_e_d_column_name

time_column_name = thirty_min_sample.columns 
#%%

"""Plotting all four datasets for all four years"""
fig,ax=plt.subplots()

#Creating pairs of columns to iterate over using the zip() function
for x_col, y_col in zip(time_column_name,B_h_d_column_name):    
    ax.plot(thirty_min_sample[x_col],Blackett_heating_data[y_col], 'o',
              color = "green")
ax.set_title('Blackett Heating', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)', fontsize = 13)
ax.set_ylabel('Total Heat Energy (kWh)', fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(B_h_d_date, Blackett_heating_data[0],'o', color = 'green')
tick_positions = [date for date in B_h_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)
ax2.xaxis.set_label_coords(0.95, 1.02)
#


plt.show()

#%%
fig,ax=plt.subplots()
for x_col, y_col in zip(time_column_name,B_e_d_column_name):
    ax.plot(thirty_min_sample[x_col],Blackett_electricity_data[y_col], 'o',
              color = 'orange')
ax.set_title('Blackett Electricity', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)',fontsize = 13)
ax.set_ylabel('Total Electrical Energy (kWh)', fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(B_e_d_date, Blackett_electricity_data[0], 'o', color = 'orange')
tick_positions = [date for date in B_e_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)
ax2.xaxis.set_label_coords(0.95, 1.02)

plt.show()
#%%

fig,ax=plt.subplots()   
for x_col, y_col in zip( time_column_name, H_h_d_column_name):
    ax.plot(thirty_min_sample[x_col],Huxley_heating_data[y_col], 'o',
             color = 'Blue')
ax.set_title('Huxley Heating', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)', fontsize = 13)
ax.set_ylabel('Total Heat Energy (kWh)', fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(H_h_d_date, Huxley_heating_data[0], 'o', color = 'blue')
tick_positions = [date for date in H_h_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)
ax2.xaxis.set_label_coords(0.95, 1.02)


plt.show()
#%%
fig,ax=plt.subplots()
    
for x_col, y_col in zip(time_column_name, H_e_d_column_name):
    ax.plot(thirty_min_sample[x_col],Huxley_electricity_data[y_col], '.',
             color = 'purple')
ax.set_title('Huxley Electricity', fontsize = 15)
ax.set_xlabel('Time elapsed from start (Days)',  fontsize = 13 )
ax.set_ylabel('Total Electrical Energy (kWh)',  fontsize = 13)
ax2 = ax.twiny()
ax2.tick_params(axis='x', which='both', bottom=False, top=True)
ax2.plot(H_h_d_date, Huxley_electricity_data[0], '.', color = 'purple')
tick_positions = [date for date in H_e_d_date if (date.month == 1 and date.day == 1)]
ax2.set_xticks(tick_positions)
ax2.set_xlabel('Date', fontsize = 13)
ax2.xaxis.set_label_coords(0.95, 1.02)
plt.show()


#%%
"""The aim of this next section is to do all these plots for each individual year"""

#Creating the list of dates with the separeted dates, as the dates are the same, used an arbitrary one      
separated_dates = dates_by_year(B_h_d_date)

desired_years = list(separated_dates.keys()) # The years that are present in the data
length_of_years = list(len(lst) for lst in separated_dates.values()) #The length of each year
thirty_min_seperate_y = time_frame_separate_years(
    num_columns, 30, desired_years, length_of_years)

def data_ploting_separate_years(list_of_data, date_data, title, y_axis_name, color= ""):
    """
    This function was created so that the plotting of the data is convenient

    Parameters
    ----------
    list_of_data : list of DataFrames
        The list of all the data that has been sepearted into different years.
    date_data : Series
        A series containing all the dates.
    title : string
        The title of the plot.
    y_axis_name : string
        The axis title
    color : string
        The desired color of the plot

    Returns
    -------
    None.

    """
    if color:
        color = color
    else:
        color = "green"
    for i in range(0, len(list_of_data)):
        fig, ax  = plt.subplots()
        df = list_of_data[i]        
        df2 = thirty_min_seperate_y[i]
        dates = separated_dates[2018+i]       
        
        for x_col, y_col in zip(time_column_name, time_stamps):
            ax.plot(df2[x_col],df[y_col], '.',
                      color = color)
        ax.set_title(title +" " + str(2018+i), fontsize = 15)
        ax.set_xlabel('Time elapsed from start (Days)',  fontsize = 13 )
        ax.set_ylabel(y_axis_name,  fontsize = 13)
        ax2 = ax.twiny()
        ax2.tick_params(axis='x', which='both', bottom=False, top=True)
        ax2.plot(dates, df[0], '.', color = color)       
        ax2.set_xlabel('Date', fontsize = 13)
        plt.show()

#%%
"""PLotting the data sets individually"""
        
data_ploting_separate_years(B_h_d_separated, B_h_d_date,
                            'Blackett Heating', 'Total Heating(kWh)')
#%%
data_ploting_separate_years(B_e_d_separated, B_e_d_date,
                            'Blackett Electricity', 'Total Electricity(kWh)',
                            'orange')
#%%

data_ploting_separate_years(H_h_d_separated, H_h_d_date,
                            'Huxley Heating', 'Total Heating(kWh)', 'blue')

#%%
data_ploting_separate_years(H_e_d_separated, H_e_d_date,
                            'Huxley electricity', 'Total Electricity(kWh)',
                            'purple')


#%%
"""
To clean the data I will find the mean and the standard deviation for every dataset
and delete everything outside 5 sigma. This method might have issues from
seasonal variation but is a good first method"""

def filtering(df, num_of_sigma):
    """
    This function was created to filter data according to the standard 
    deviation from the mean of the inputed DataFrame

    Parameters
    ----------
    df : DataFrame
        The DataFrame that we want to filter
    num_of_sigma : float
        The number of sigmas from which we want to filter from..

    Returns
    -------
    df1 : DataFrame
        Returns a DataFrame where the values which are outside the desired
        range are returned as Nan.

    """
    stackeddf = df.stack()
    mean = np.mean(stackeddf)
    sig = np.std(stackeddf)
    condition1 = df >= mean + num_of_sigma*sig  
    condition2 = df <= mean - num_of_sigma*sig
    df1 = df.mask(condition1)
    df1 = df1.mask(condition2)
    return df1

#%%

def plotting_data_all_years(df, time_data, title, y_title, color= ''):
    if color:
        color = color
    else:
        color = "green" 
    fig,ax=plt.subplots()

    for x_col, y_col in zip( time_column_name,B_h_d_column_name):
        
        ax.plot(time_data[x_col], df[y_col], '.',
                 color = color)
    ax.set_title(title, fontsize = 15)
    ax.set_xlabel('Time elapsed from start (Days)', fontsize = 13)
    ax.set_ylabel(y_title, fontsize = 13)
    ax2 = ax.twiny()
    ax2.tick_params(axis='x', which='both', bottom=False, top=True)
    ax2.plot(B_h_d_date, df[0],'.', color = color)
    tick_positions = [date for date in B_h_d_date if (date.month == 1 and date.day == 1)]
    ax2.set_xticks(tick_positions)
    ax2.set_xlabel('Date', fontsize = 13)
    ax2.xaxis.set_label_coords(0.95, 1.02)
        
#%%

b_h_d_filtered = filtering(Blackett_heating_data, 5)
plotting_data_all_years(b_h_d_filtered, thirty_min_sample, 'Blackett Heating',
                        'Total Heat Energy(kWh)')

#%%
b_e_d_filtered = filtering(Blackett_electricity_data, 5)

plotting_data_all_years(b_e_d_filtered, thirty_min_sample,
                        'Blackett Electricity', 'Total Electrical Energy(kWh)',
                        color = "orange")

#%%
h_h_d_filtered = filtering(Huxley_heating_data, 5)
plotting_data_all_years(h_h_d_filtered, thirty_min_sample,
                        'Huxley Heating', 'Total Heat Energy(kWh)',
                        color = "blue")

#%%
h_e_d_filtered = filtering(Huxley_electricity_data, 5)

plotting_data_all_years(h_e_d_filtered, thirty_min_sample,
                        'Huxley Electricity', 'Total Electrical Energy(kWh)',
                        color = "purple")


#%%





