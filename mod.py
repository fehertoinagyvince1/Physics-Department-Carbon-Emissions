# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:13:52 2023

@author: feher
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

class modifier:
    def __init__(self, original_data):
        self.original_data = original_data
        self.data, self.df_date = self.set_first_row_as_labels()
        self.data_w_dates = self.set_first_row_as_labels_with_years()
        self.num_rows, self.num_columns = self.data.shape 
        
    
    def set_first_row_as_labels(self):
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
        df = self.original_data
        labels = df.iloc[0]
    
        # Set the labels as the new index
        df = df[1:]   
        df.columns = labels
        df_date = pd.to_datetime(df['Date']).dt.date
        df = df.drop(['Site Group', 'Site', 'Monitoring Point', 'Type',
                      'Unit of Measurement', 'Date'], axis = 1)
        """Could modify this to be customisable, or require this data structure"""
        return df, df_date
    
    
    def set_first_row_as_labels_with_years(self):
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
        df_w_dates = self.original_data
        labels = df_w_dates .iloc[0]
    
        # Set the labels as the new index
        df_w_dates  = df_w_dates[1:]   
        df_w_dates.columns = labels
        
        df_w_dates = df_w_dates.drop(['Site Group', 'Site', 'Monitoring Point', 'Type',
                      'Unit of Measurement'], axis = 1, errors='ignore')
        #df = pd.to_datetime(df['Date']).dt.date
        return df_w_dates 
    
    
    """Creating DataFrames for the years seperately"""
    
    def finding_dates(self):       
    
        self.separated_dates = self.dates_by_year()
        self.desired_years = list(self.separated_dates.keys())
        self.length_of_years = list(len(lst) for lst in self.separated_dates.values())
        self.thirty_min_separate_y = self.time_frame_separate_years(30)
        return self.separated_dates, self.desired_years, self.length_of_years, self.thirty_min_separate_y
    
    def separating_years(self):
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
        
        
        #Changing the Date given into datetime format, which enables a lot of functionality:
        self.data_w_dates['Date'] = pd.to_datetime(self.data_w_dates['Date']) 
    # Group the DataFrame based on the 'Condition' column
        
        dates_info = self.finding_dates()  # Store the result of finding_dates        
        desired_years = self.desired_years
        
        
        # Create separate DataFrames for each group
        separate_dfs = []
        separate_dfs_w_years = []
        for year in desired_years:
                     
            year_df = self.data_w_dates[self.data_w_dates['Date'].dt.year == year]        
            separate_dfs_w_years.append(year_df)
            separate_dfs.append(year_df.drop(['Date'], axis =1))
        self.separate_dfs = separate_dfs
        self.separate_dfs_w_years = separate_dfs_w_years
        return separate_dfs, separate_dfs_w_years
    
    def time_frame(self, sample_size, length = None):
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
        if length is None:
            length = self.num_rows
        
        df = pd.DataFrame(index = range(length), columns = range(self.num_columns))
        for i in range(0, length):
            for j in range(0, self.num_columns):
                df.at[i,j] = (j*sample_size + i*1440)/(60*24)
        return df
    
    def time_frame_separate_years(self, sample_size):
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
        for i, year in enumerate(self.desired_years):
            self.length = self.length_of_years[i] 
            df = self.time_frame(30,self.length)   
            separate_dfs.append(df)
        return separate_dfs
    
    def dates_by_year(self):
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
        for date in self.df_date:
            year = date.year
            if year not in year_lists:
                year_lists[year] = []
            year_lists[year].append(date)            
       
        return year_lists
    
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
    
    
    def filtering_years_separated(self, num_of_sigma):
        list_of_df = self.separate_dfs
        for i in range(0, len(list_of_df)):
            df = list_of_df[i]
            df_filtered = self.filtering(df, num_of_sigma)
            list_of_df[i] = df_filtered
        return list_of_df
    
    
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
    
    
    def separate_into_months(self,):
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
        date_series = self.df_date
        separated_w_date = self.separate_dfs_w_years
        df = {}
        df_w_date = {}  
            
        separated_dates_y_m = self.dates_by_year_and_month(date_series)
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
    
    
    def filtering_years_months_separated(self,dict_of_df, num_of_sigma):
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
                df_filtered = self.filtering(df, num_of_sigma)
                dict_of_df[2018+j][i][0] = df_filtered
        return  dict_of_df_2