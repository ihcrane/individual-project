#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from prepare import remove_outliers, reduce_mem_usage

from scipy import stats

pd.set_option('display.max_columns', None)


# In[3]:


def missing_values(df):
    
    '''
    This function determins how many missing rows are in each column and what percentage it is.
    '''
    
    # determing how many missing rows there are
    missing_df = pd.DataFrame(df.isna().sum(), columns=['num_rows_missing'])
    
    # determining what percentage of the column is missing
    missing_df['pct_rows_missing'] = missing_df['num_rows_missing'] / len(df)
    
    return missing_df


# In[4]:


def handle_missing_values(df, prop_required_col, prop_required_row):
    
    '''
    This function drops columns and rows if it is contains a certain percentage of nulls determined by the user.
    '''
    
    # determine how many nulls are needed to meet the threshold and then dropping the column if it meets it
    drop_cols = round(prop_required_col * len(df))
    df.dropna(thresh=drop_cols, axis=1, inplace=True)

    # determine how many nulls are needed to meet the threshold and then dropping the row if it meets it    
    drop_rows = round(prop_required_row * len(df.columns))
    df.dropna(thresh=drop_rows, axis=0, inplace=True)
    
    return df


# In[63]:


def drop_rename_cols(df):
    
    '''
    This function drops columns that do not have useful data and renames other columns to for easier readability.
    '''
    
    # list of columns to be dropped
    drop_cols = ['vin', 'dealer_zip','description',
                 'torque','transmission_display','trimId',
                 'trim_name','wheel_system_display', 'fleet',
                 'height','isCab','latitude','longitude','listing_id',
                 'main_picture_url','major_options','franchise_make',
                 'model_name','power','salvage','listing_color', 
                 'savings_amount','sp_id', 'sp_name','theft_title', 'engine_type', 
                 'frame_damaged','exterior_color','interior_color']
    
    df.drop(columns=drop_cols, inplace=True)
    
    # list of columns to be renamed
    cols_rename = {'city_fuel_economy':'city_mpg','engine_cylinders':'cyl',
                   'engine_displacement':'displ','franchise_dealer':'dealer',
                   'fuel_tank_volume':'tank_size','has_accidents':'accidents',
                   'highway_fuel_economy':'hwy_mpg','is_new':'new','make_name':'model',
                   'maximum_seating':'seats','owner_count':'owners',
                   'transmission':'tran','wheel_system':'drive_type'}
    
    df.rename(columns=cols_rename, inplace=True)
    
    return df


# In[6]:


def formatting_cols(df):

    '''
    This function formats the data frame to allow of data manipulation in the future. As well as preparing
    it for exploration phase and modeling phase.
    '''
    
    # removing values when null. Then splittin the values and dropping the 'in' and converting to float
    df = df[df['back_legroom'] != '--']
    df['back_legroom'] = df['back_legroom'].str.split(' ',expand=True).drop(columns=[1])
    df['back_legroom'].astype('float64')
    
    # formatting body type for easier readibility
    df['body_type'] = df['body_type'].map({'SUV / Crossover':'SUV', 'Sedan':'Sedan',
                                           'Pickup Truck':'Pickup','Coupe':'Coupe',
                                           'Minivan':'Minivan', 'Wagon':'Wagon','Van':'Van',
                                           'Convertible':'Convertible'})
    
    # dropping the word 'seats' from the data and converting to int
    df = df[df['seats'] != '--']
    df['seats'] = df['seats'].str.split(' ', expand=True).drop(columns=[1])
    df['seats'].astype('float64')
    
    # dropping unnecessary words and keeping on the the number and style of cylinders
    df['cyl'] = df['cyl'].str.split(' ',expand=True).drop(columns=[1,2,3])
    
    # dropping the 'in' and converting to float
    df = df[df['wheelbase'] != '--']
    df['wheelbase'] = df['wheelbase'].str.split(' ',expand=True).drop(columns=[1])
    df['wheelbase'].astype('float64')
    
    # dropping the 'in' and converting to float
    df = df[df['width'] != '--']
    df['width'] = df['width'].str.split(' ', expand=True).drop(columns=[1])
    df['width'].astype('float64')
    
    # dropping the 'in' and converting to float
    df = df[df['front_legroom'] != '--']
    df['front_legroom'] = df['front_legroom'].str.split(' ',expand=True).drop(columns=[1])
    df['front_legroom'].astype('float64')
    
    # dropping the null values, dropping 'gal' and converting to float
    df = df[df['tank_size'] != '--']
    df['tank_size'] = df['tank_size'].str.split(' ',expand=True).drop(columns=[1])
    df['tank_size'].astype('float64')

    # dropping the 'in' and converting to float
    df = df[df['length'] != '--']
    df['length'] = df['length'].str.split(' ', expand=True).drop(columns=[1])
    df['length'].astype('float64')
    
    return df


# In[ ]:


with pd.read_csv("used_cars_data.csv", chunksize=5000) as reader:
    reader
    i = 0
    for chunk in reader:
        df = pd.DataFrame(chunk)
        
        df = drop_rename_cols(df)
        df = handle_missing_values(df, .4, .4)
        df = formatting_cols(df)
        
        cols = df.columns.to_list()
        
        if i == 0:
            df.to_csv('formatted_data.csv', mode='a')
        else:
            df.to_csv('formatted_data.csv', mode='a', header=cols)
        i += 1
        


# In[54]:


def col_conversion(df):
    
    
    df = df[df['daysonmarket']!='daysonmarket']
    
    df = df[(df['horsepower']!='False') & (df['horsepower']!='True')]
    
    df = df[(df['seller_rating']!='A') & (df['seller_rating']!='M') & 
        (df['seller_rating']!='CVT') & (df['seller_rating']!='Dual Clutch')]
    
    num_cols = ['daysonmarket','displ','hwy_mpg',
            'horsepower','seats','owners','price','year',
            'back_legroom','city_mpg','front_legroom','tank_size',
            'length','mileage','seller_rating','wheelbase','width']
    
    df[num_cols] = df[num_cols].astype('float64')
    
    return df


# In[64]:


def wrangle_cars():
    
    df = pd.read_csv('formatted_data.csv')
    
    df = col_conversion(df)
    
    df, var_fences = remove_outliers(df, num=50, k=3)
    
    df = df.reset_index(drop=True)
    
    df = df.drop(columns='Unnamed: 0')
    
    df = df.dropna()
    
    df = reduce_mem_usage(df)
    
    return df


# In[ ]:





# In[ ]:




