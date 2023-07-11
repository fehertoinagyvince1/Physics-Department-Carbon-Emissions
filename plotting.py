# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:14:36 2023

@author: feher
"""

"""This module includes everything that plots data"""

import mod
import matplotlib.pyplot as plt

class plotting:
    def __init__(self, data_original):
    
        self.data, self.date_data = mod.set_first_row_as_labels(data_original)
        self.num_rows, self.num_columns = self.data.shape
        self.thirty_min_sample = mod.time_frame(self.num_rows, self.num_columns, 30) 
        self.d_column_name = self.data.columns
        self.time_stamps = self.d_column_name
        self.time_column_name = self.thirty_min_sample.columns 
        self.list_of_data = mod.separating_years(data_original,self.date_data)[0] 

    def plottingdata(self, title, y_axis_name, x_axis_name, color):
        if color:
            color = color
        else:
            color = "green"           
        fig,ax=plt.subplots()    
        time_column_name = self.thirty_min_sample.columns     
        #Creating pairs of columns to iterate over using the zip() function
        for x_col, y_col in zip(self.time_column_name,self.d_column_name):        
            ax.plot(self.thirty_min_sample[x_col],self.data[y_col], 'o',
                      color = color)
        ax.set_title(title, fontsize = 15)
        ax.set_xlabel(y_axis_name, fontsize = 13)
        ax.set_ylabel(x_axis_name, fontsize = 13)
        ax2 = ax.twiny()
        ax2.tick_params(axis='x', which='both', bottom=False, top=True)
        ax2.plot(self.date_data, self.data[0],'o', color = color)
        tick_positions = [date for date in self.date_data if (date.month == 1 and date.day == 1)]
        ax2.set_xticks(tick_positions)
        ax2.set_xlabel('Date', fontsize = 13)
        ax2.xaxis.set_label_coords(0.95, 1.02)
        plt.show()
        
    def data_ploting_separate_years(self, title, y_axis_name, color= ""):
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
         
        for i in range(0, len(self.list_of_data)):
            plt.rcParams['axes.xmargin'] = 0            
            fig, ax  = plt.subplots()
            thirty_min_seperate_y = mod.finding_dates(self.list_of_data[i], self.date_data)[3]
            separated_dates = mod.finding_dates(self.list_of_data[i], self.date_data)[0]
            first_year = mod.finding_dates(self.list_of_data[i], self.date_data)[1][0]
            df = self.list_of_data[i]        
            df2 = thirty_min_seperate_y[i]
            dates = separated_dates[first_year+i]            
            
            for x_col, y_col in zip(self.time_column_name, self.time_stamps):
                ax.plot(df2[x_col],df[y_col], '.',
                          color = color)            
            ax.set_title(title +" " + str(2018+i), fontsize = 15)
            ax.set_xlabel('Time elapsed from start (Days)',  fontsize = 13 )
            ax.set_ylabel(y_axis_name,  fontsize = 13)           
            ax2 = ax.twiny()
            ax2.tick_params(axis='x', which='both', bottom=False, top=True)             
           #ax2.plot(dates, df[0], '.', color = color)       
            ax2.set_xlabel('Date', fontsize = 13)            
            plt.show()
            
    def plotting_data_all_years(self, title, y_title, color= ''):
        if color:
            color = color
        else:
            color = "green" 
        fig,ax=plt.subplots()

        for x_col, y_col in zip( self.time_column_name,self.d_column_name):
            plt.rcParams['axes.xmargin'] = 0
            
            
            ax.plot(self.thirty_min_sample[x_col], self.data[y_col], '.',
                     color = color)
        ax.set_title(title, fontsize = 15)
        ax.set_xlabel('Time elapsed from start (Days)', fontsize = 13)
        ax.set_ylabel(y_title, fontsize = 13)
        ax2 = ax.twiny()
        ax2.tick_params(axis='x', which='both', bottom=False, top=True)
        
        tick_positions = [date for date in self.date_data if (date.month == 1 and date.day == 1)]
        ax2.set_xticks(tick_positions)
        ax2.set_xlabel('Date', fontsize = 13)
        ax2.xaxis.set_label_coords(0.95, 1.02)
            