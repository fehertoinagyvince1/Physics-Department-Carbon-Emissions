# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:54:32 2023

@author: feher"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import Data_mod as mod
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
    separate_dfs_w_years = []
    for year in desired_years:
        year_df = B_h_d_wyears[B_h_d_wyears['Date'].dt.year == year]        
        separate_dfs_w_years.append(year_df)
        separate_dfs.append(year_df.drop(['Date'], axis =1))
    return separate_dfs, separate_dfs_w_years

#Creating the DataFrames separated into years

B_h_d_separated = separating_years(Blackett_heating_data_original)[0]
B_e_d_separated = separating_years(Blackett_electricity_data_original)[0]
H_e_d_separated = separating_years(Huxley_electricity_data_original)[0]
H_h_d_separated = separating_years(Huxley_heating_data_original)[0]

B_h_d_separated_w_years = separating_years(Blackett_heating_data_original)[1]
B_e_d_separated_w_years = separating_years(Blackett_electricity_data_original)[1]
H_e_d_separated_w_years = separating_years(Huxley_electricity_data_original)[1]
H_h_d_separated_w_years = separating_years(Huxley_heating_data_original)[1]




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
        plt.rcParams['axes.xmargin'] = 0
        
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
        plt.rcParams['axes.xmargin'] = 0
        
        
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

"""This section is to plot the filtered data yearly, to see the seasonal
 variation better, to decide if a seasonal or daily filtering would be better."""
 
def filtering_years_separated(list_of_df, num_of_sigma):
    for i in range(0, len(list_of_df)):
        df = list_of_df[i]
        df_filtered = filtering(df, num_of_sigma)
        list_of_df[i] = df_filtered
    return list_of_df

#%%

B_h_f_separated = filtering_years_separated(B_h_d_separated, 5)
B_e_f_separated = filtering_years_separated(B_e_d_separated, 5)
H_h_f_separated = filtering_years_separated(H_h_d_separated, 5)
H_e_f_separated = filtering_years_separated(H_e_d_separated, 5)

#%%
data_ploting_separate_years(B_h_f_separated, B_h_d_date,
                            'Blackett Heating', 'Total Heating Energy(kWh)')
#%%
data_ploting_separate_years(B_e_f_separated, B_e_d_date,
                             'Blackett Electricity', 
                             'Total Electrical Energy(kWh)', 'orange')   

#%%
data_ploting_separate_years(H_e_f_separated, H_e_d_date,
                             'Huxley Electricity', 
                             'Total Electrical Energy(kWh)', 'purple')
#%%
data_ploting_separate_years(H_h_f_separated, H_h_d_date,
                            'Blackett Heating', 'Total Heating Energy(kWh)',
                            'blue')   

#%%%  

"""Splitting the yearly data further down into months"""

def dates_by_year_and_month(date_list):
    """This function was created to extract the years analysed and find the
    length of the year from the data   

    Parameters
    ----------
    date_list : list of datetime.datetime objects
        List of all the dates in the dataset.  

    Returns
    -------
    year_lists : dict containing lists
        Returns a dictionary of dictionaries with the key being the years, the 
        entries of the outer dictionary being the months, and the entries of the
        inner dictionaries being the dates.

    """
    year_month_lists = {}
    for date in date_list:
        year = date.year
        month = date.month
        if year not in year_month_lists:
            year_month_lists[year] = {}
            
        if month not in year_month_lists[year]:
            year_month_lists[year][month] = []
        year_month_lists[year][month].append(date)
   
    return year_month_lists



separated_dates_months = dates_by_year(B_h_d_date)
separated_dates_y_m = dates_by_year_and_month(B_h_d_date)
length_of_months = list(len(lst) for lst in separated_dates_y_m[2018].values())
thirty_min_seperate_y_m = time_frame_separate_years(
    num_columns, 30, desired_years, length_of_months)




#%%




def separate_into_months(separated_w_date, date_series):
    """This function takes in a dataframe taht includes the date stamps
    and creates two dictionaries, each seperating the data into months and
    years, one with the date stamps removed, and one with no date stamps.
    

    Parameters
    ----------
    separated_w_date : DataFrame
        The data with the date stamps.

    Returns
    -------
    df : DataFrame
        The modified DataFrame with no dates.
    df_w_date :  DataFrame
        The modified DataFrame with dates.

    """
    df = {}
    df_w_date = {}  
        
    separated_dates_y_m = dates_by_year_and_month(date_series)
    for i in range(0,len(separated_w_date)):        
        months = list(separated_dates_y_m[2018+i].keys()) # The years that are present in the data
        df_w_date[2018+i] = {} 
        df[2018+i] = {} 
        for month in months:
            month_df = separated_w_date[i].loc[separated_w_date
                                               [i]['Date'].dt.month == month]
            month_no_date = month_df.drop(['Date'], axis=1)
            
            df_w_date[2018+i][month] = []
            df[2018+i][month] = []
            
            df_w_date[2018+i][month].append(month_df)
            df[2018+i][month].append(month_no_date)
              
    return  df, df_w_date
    



#%%
def filtering_years_months_separated(dict_of_df, num_of_sigma):
    """
    This function filters the data for every month seperately

    Parameters
    ----------
    dict_of_df : dict
        The dictionary that includes the data that is broken down into years 
        and months
    num_of_sigma : float
        The statistical error margain

    Returns
    -------
    dict_of_df_2 : dict
        The filtered dictionary

    """
    dict_of_df_2 = dict_of_df
    for j in range(0, len(dict_of_df)):        
        for i in range(1, len(dict_of_df[2018+j])):
            df = dict_of_df[2018+j][i][0]
            df_filtered = filtering(df, num_of_sigma)
            dict_of_df[2018+j][i][0] = df_filtered
    return  dict_of_df_2




#%%
def total_monthly_usage(dict_of_df_1):
    """
    This function sums the energy usage over a month

    Parameters
    ----------
    dict_of_df_1 : dict
        The dictionary that includes the data that is broken down into years 
        and months       

    Returns
    -------
    result_dict : dict
        A dictionary that has a list associated with every month in every
        year which includes the total energy usage over that month.

    """
    result_dict = {}
    for year, months_dict in dict_of_df_1.items():
        result_dict[year] = {}
        for month, df in months_dict.items():
            total_sum = df[0].sum().sum()  # Calculate the sum of the dataframe
            result_dict[year][month] = total_sum
    return result_dict



def plot_monthly_total_yearly(total_data, title, y_axis):
    for year, months_dict in total_data.items():
        
        
        fig, ax = plt.subplots()
        y_list = list(months_dict)
        usage_list = [total_data[year][month]/1000 for month in y_list]
        
        # Convert month numbers to dates
        month_dates = [mdates.datetime.datetime(year, month, 1) for month in y_list]
        ax.grid()
        ax.scatter(month_dates, usage_list, s=70,zorder=10)
        

        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Displays abbreviated month names
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_ylabel(y_axis, fontsize = 15)
        ax.set_title(title + ' ' + str(year), fontsize = 17)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.show()
        
       
        
        
B_h_d_s_m = separate_into_months(B_h_d_separated_w_years, B_h_d_date)[0]
B_e_d_s_m = separate_into_months(B_e_d_separated_w_years,B_h_d_date)[0]

H_h_d_s_m = separate_into_months(H_h_d_separated_w_years, B_h_d_date)[0]
H_e_d_s_m = separate_into_months(H_e_d_separated_w_years, B_h_d_date)[0]


 

B_h_s_m_f = filtering_years_months_separated(B_h_d_s_m, 5)
B_e_s_m_f = filtering_years_months_separated(B_e_d_s_m, 5)
H_h_s_m_f = filtering_years_months_separated(H_h_d_s_m, 5)
H_e_s_m_f = filtering_years_months_separated(H_e_d_s_m, 5)

B_h_m_t_f = total_monthly_usage(B_h_s_m_f)
B_e_m_t_f = total_monthly_usage(B_e_s_m_f)
H_h_m_t_f = total_monthly_usage(H_h_s_m_f)
H_e_m_t_f = total_monthly_usage(H_e_s_m_f)

#%%
plot_monthly_total_yearly(B_h_m_t_f, 'Blackett Heating', 'Monthly Usage (MWh)')
#%%
plot_monthly_total_yearly(B_e_m_t_f, 'Blackett Electricity', 'Monthly Usage (MWh)')
#%%
plot_monthly_total_yearly(H_h_m_t_f, 'Huxley Heating', 'Monthly Usage (MWh)')
#%%
plot_monthly_total_yearly(H_e_m_t_f, 'Huxley Electricity', 'Monthly Usage (MWh)')




#%%


def plot_monthly_total(total_data, title, y_axis):
    fig, ax = plt.subplots()
    plt.rcParams['axes.xmargin'] = 0
    ax.grid()
    all_month_dates = []
    all_usage_list = []
    
    for year, months_dict in total_data.items():
        y_list = list(months_dict)
        usage_list = [total_data[year][month]/1000 for month in y_list]
        
        # Convert month numbers to dates
        month_dates = [datetime(year, month, 1) for month in y_list]
        
        all_month_dates.extend(month_dates)
        all_usage_list.extend(usage_list)
    
    ax.plot(all_month_dates, all_usage_list, zorder=10, color='navy')

    # Format x-axis as years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Displays the year
    ax.xaxis.set_major_locator(mdates.YearLocator())

    ax.set_ylabel(y_axis, fontsize=15)
    ax.set_title(title, fontsize=17)
    ax.set_xlabel('Date', fontsize = 15)
    ax.xaxis.set_label_coords(0.95,-0.015)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.show()
        
      

#%%
plot_monthly_total(B_h_m_t_f, 'Blackett Heating', 'Monthly Usage (MWh)')
#%%
plot_monthly_total(B_e_m_t_f, 'Blackett Electricity', 'Monthly Usage (MWh)')
#%%
plot_monthly_total(H_h_m_t_f, 'Huxley Heating', 'Monthly Usage (MWh)')
#%%
plot_monthly_total(H_e_m_t_f, 'Huxley Electricity', 'Monthly Usage (MWh)')


        
    


        
        
    
    





