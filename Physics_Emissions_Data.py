# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:54:32 2023

@author: feher"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import datetime
import mod 
import plotting as vis
#%%

#Loading Data into a Pandas Data Frames

Data_file_path = r"C:\Users\feher\OneDrive - Imperial College London\UROP 2023\Blackett_Huxley_Data.xlsx"



Blackett_heating_data_original = pd.read_excel(Data_file_path, 
                                               sheet_name= 'BLKT(h)')
Blackett_electricity_data_original = pd.read_excel(Data_file_path, sheet_name= 
                                          'BLKT(e)')
Huxley_heating_data_original = pd.read_excel(Data_file_path, 
                                             sheet_name= 'HXLY(h)')
Huxley_electricity_data_original = pd.read_excel(Data_file_path, sheet_name= 'HXLY(e)')

#%%

Blackett_heating = vis.plotting(Blackett_heating_data_original)
#%%
Blackett_electricity = vis.plotting(Blackett_electricity_data_original)
Huxley_heating = vis.plotting(Huxley_heating_data_original)
Huxley_electricity = vis.plotting(Huxley_electricity_data_original)


#%%
"""Plotting the raw data for all years"""

Blackett_heating.plottingdata('Blackett Heating', 
             'Total Heat Energy (kWh)', 'Time elapsed from start', 'green')
#%%

Blackett_electricity.plottingdata('Blackett Electricity', 
             'Total Electrical Energy (kWh)', 'Time elapsed from start', 'orange')
#%%
Huxley_heating.plottingdata('Huxley Heating', 
             'Total Heat Energy (kWh)', 'Time elapsed from start', 'blue')
#%%

Huxley_electricity.plottingdata('Huxley Electricity', 
             'Total Electrical Energy (kWh)', 'Time elapsed from start', 'purple')

#%%
"""In this section I am plotting the unfiltered data for the years separetely"""

Blackett_heating.data_ploting_separate_years('Blackett Heating',
                                             'Total Heating Energy(kWh)')
#%%
Blackett_electricity.data_ploting_separate_years('Blackett Electricity ',
                                                 'Total Electrical Energy(kWh)')
#%%
Huxley_heating.data_ploting_separate_years('Huxley Heating',
                                             'Total Heating Energy(kWh)')
#%%
Huxley_electricity.data_ploting_separate_years('Huxley Electricity ',
                                                 'Total Electrical Energy(kWh)')


#%%
b_h_d_filtered = mod.filtering(Blackett_heating_data, 5)
b_e_d_filtered = mod.filtering(Blackett_electricity_data, 5)
h_h_d_filtered = mod.filtering(Huxley_heating_data, 5)
h_e_d_filtered = mod.filtering(Huxley_electricity_data, 5)

#%%
Blackett_heating.plotting_data_all_years(b_h_d_filtered, 'Blackett Heating',
                        'Total Heat Energy(kWh)')

     

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

b_h_d_filtered = mod.filtering(Blackett_heating_data, 5)
b_e_d_filtered = mod.filtering(Blackett_electricity_data, 5)
h_h_d_filtered = mod.filtering(Huxley_heating_data, 5)
h_e_d_filtered = mod.filtering(Huxley_electricity_data, 5)
#%%

plotting_data_all_years(b_h_d_filtered, thirty_min_sample, 'Blackett Heating',
                        'Total Heat Energy(kWh)')

#%%

plotting_data_all_years(b_e_d_filtered, thirty_min_sample,
                        'Blackett Electricity', 'Total Electrical Energy(kWh)',
                        color = "orange")

#%%

plotting_data_all_years(h_h_d_filtered, thirty_min_sample,
                        'Huxley Heating', 'Total Heat Energy(kWh)',
                        color = "blue")

#%%


plotting_data_all_years(h_e_d_filtered, thirty_min_sample,
                        'Huxley Electricity', 'Total Electrical Energy(kWh)',
                        color = "purple")



#%%

B_h_f_separated = mod.filtering_years_separated(B_h_d_separated, 5)
B_e_f_separated = mod.filtering_years_separated(B_e_d_separated, 5)
H_h_f_separated = mod.filtering_years_separated(H_h_d_separated, 5)
H_e_f_separated = mod.filtering_years_separated(H_e_d_separated, 5)

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
#%% 

separated_dates_months = mod.dates_by_year(B_h_d_date)
separated_dates_y_m = mod.dates_by_year_and_month(B_h_d_date)
length_of_months = list(len(lst) for lst in separated_dates_y_m[2018].values())
thirty_min_seperate_y_m = mod.time_frame_separate_years(
    num_columns, 30, desired_years, length_of_months) 
#%%


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
        
       
 #%%       
        
B_h_d_s_m = mod.separate_into_months(B_h_d_separated_w_years, B_h_d_date)[0]
B_e_d_s_m = mod.separate_into_months(B_e_d_separated_w_years,B_h_d_date)[0]

H_h_d_s_m = mod.separate_into_months(H_h_d_separated_w_years, B_h_d_date)[0]
H_e_d_s_m = mod.separate_into_months(H_e_d_separated_w_years, B_h_d_date)[0]


 

B_h_s_m_f = mod.filtering_years_months_separated(B_h_d_s_m, 5)
B_e_s_m_f = mod.filtering_years_months_separated(B_e_d_s_m, 5)
H_h_s_m_f = mod.filtering_years_months_separated(H_h_d_s_m, 5)
H_e_s_m_f = mod.filtering_years_months_separated(H_e_d_s_m, 5)

B_h_m_t_f = mod.total_monthly_usage(B_h_s_m_f)
B_e_m_t_f = mod.total_monthly_usage(B_e_s_m_f)
H_h_m_t_f = mod.total_monthly_usage(H_h_s_m_f)
H_e_m_t_f = mod.total_monthly_usage(H_e_s_m_f)

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
        month_dates = [datetime.datetime(year, month, 1) for month in y_list]
        
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


        
    


        
        
    
    





