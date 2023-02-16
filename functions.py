#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from scipy import stats

from wrangle import wrangle_cars
from prepare import x_y_split, rmse, select_kbest, rfe

from xgboost import XGBRegressor
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# In[2]:


def pearson_test(df, col1):
    corr, p = stats.pearsonr(df[col1], df['price'])
    
    print(f'The correlation between {col1} and price is: {corr:.2f}')


# In[4]:


def horsepower_plot(df, col1, col2):
    sns.lmplot(x='horsepower', y='price',data=df.sample(3000), line_kws={'color':'red'}, size=6)
    plt.title('Horsepower to Price Graph')
    plt.xlabel('Horsepower')
    plt.ylabel('Price')
    plt.show()


# In[5]:


def mileage_plot(df, col1, col2):
    sns.lmplot(x='mileage', y='price',data=df.sample(3000), line_kws={'color':'red'}, size=6,)
    plt.title('Mileage to Price Graph')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.xlim((-1000,190000))
    plt.show()


# In[6]:


def width_plot(df, col1, col2):
    sns.lmplot(x='width', y='price',data=df.sample(3000), line_kws={'color':'red'}, size=6)
    plt.title('Width to Price Graph')
    plt.xlabel('Width')
    plt.ylabel('Price')
    plt.show()


# In[7]:


def length_plot(df, col1, col2):
    sns.lmplot(x='length', y='price',data=df.sample(3000), line_kws={'color':'red'}, size=6)
    plt.title('Length to Price Graph')
    plt.xlabel('Length')
    plt.ylabel('Price')
    plt.show()


# In[8]:


def dealer_plot(df, col1, col2):
    fig,ax = plt.subplots(figsize=(10,9))
    bplot = sns.barplot(x='dealer', y='price',data=df)
    plt.title('Does whether the car is sold by a dealer affect price?')
    plt.xlabel('Sold by a Dealer')
    plt.ylabel('Price')
    ax.bar_label(bplot.containers[0], padding=9)
    plt.axhline(df['price'].mean(), label='Average Price')
    plt.legend(loc='upper left')
    plt.show()


# In[9]:


def ttest_samp(df, col):
    
    sold_dealer = df[df['dealer']=='True']['price']

    t, p = stats.ttest_1samp(sold_dealer, df['price'].mean())
    
    alpha = .05
    
    if p/2 > alpha:
        print("We fail to reject null")
    elif t < 0:
        print("We fail to reject null")
    else:
        print("We reject null")


# In[11]:


def split_scale(df):
    
    df = pd.get_dummies(df, columns=['dealer', 'owners'])
    
    X_train, y_train, X_val, y_val, X_test, y_test = x_y_split(df, 'price')
    
    mms = MinMaxScaler()
    
    X_train[['back_legroom','city_mpg','daysonmarket',
         'displ','front_legroom','tank_size','hwy_mpg',
         'horsepower','length','seats','mileage',
         'seller_rating','wheelbase','width']] = mms.fit_transform(X_train[['back_legroom','city_mpg','daysonmarket',
                                                                            'displ','front_legroom','tank_size','hwy_mpg',
                                                                            'horsepower','length','seats','mileage',
                                                                            'seller_rating','wheelbase','width']])
    X_val[['back_legroom','city_mpg','daysonmarket',
         'displ','front_legroom','tank_size','hwy_mpg',
         'horsepower','length','seats','mileage',
         'seller_rating','wheelbase','width']] = mms.fit_transform(X_val[['back_legroom','city_mpg','daysonmarket',
                                                                            'displ','front_legroom','tank_size','hwy_mpg',
                                                                            'horsepower','length','seats','mileage',
                                                                            'seller_rating','wheelbase','width']])
    X_test[['back_legroom','city_mpg','daysonmarket',
         'displ','front_legroom','tank_size','hwy_mpg',
         'horsepower','length','seats','mileage',
         'seller_rating','wheelbase','width']] = mms.fit_transform(X_test[['back_legroom','city_mpg','daysonmarket',
                                                                            'displ','front_legroom','tank_size','hwy_mpg',
                                                                            'horsepower','length','seats','mileage',
                                                                            'seller_rating','wheelbase','width']])
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# In[ ]:


def preds_table(y_train):
    preds = pd.DataFrame({'actual':y_train,
                          'baseline':y_train.mean()})
    
    baseline_rmse = rmse(preds, 'baseline')
    
    return preds, baseline_rmse


# In[14]:


def linear_reg(X_train, y_train, preds):
    
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    preds['lm_preds'] = lm.predict(X_train)
    
    lm_rmse = rmse(preds, 'lm_preds')
    
    print(f'RMSE: {lm_rmse}')
    
    return preds, lm_rmse


# In[16]:


def lasso(X_train, y_train, preds):
    
    lasso = LassoLars(alpha=0)

    lasso.fit(X_train, y_train)

    preds['lasso_preds'] = lasso.predict(X_train)
    
    lasso_rmse = rmse(preds, 'lasso_preds')
    
    print(f'RMSE: {lasso_rmse}')
        
    return preds, lasso_rmse


# In[17]:


def lm_poly(X_train, y_train, preds):
    
    pf = PolynomialFeatures(degree=2)

    pf.fit(X_train, y_train)
    X_polynomial = pf.transform(X_train)
    
    lmtwo = LinearRegression()
    lmtwo.fit(X_polynomial, y_train)
    
    preds['poly_preds'] = lmtwo.predict(X_polynomial)
    
    poly_rmse = rmse(preds, 'poly_preds')
    
    print(f'RMSE: {poly_rmse}')
    
    return preds, poly_rmse


# In[27]:


def lasso_poly(X_train, y_train, preds):
    
    pf = PolynomialFeatures(degree=2)

    pf.fit(X_train, y_train)
    X_polynomial = pf.transform(X_train)
    
    lassotwo = LassoLars(alpha=0)

    lassotwo.fit(X_polynomial, y_train)
    
    preds['lasso_poly'] = lassotwo.predict(X_polynomial)
    
    lassopoly_rmse = rmse(preds, 'lasso_poly')
    
    print(f'RMSE: {lassopoly_rmse}')
    
    return preds, lassopoly_rmse


# In[19]:


def xgb_model(X_train, y_train, preds):
    
    xgb = XGBRegressor(objective='reg:squarederror',n_estimators=20, max_depth=4, 
                   subsample=0.5, colsample_bytree=0.7, seed=42)
    
    xgb.fit(X_train, y_train)
    
    preds['xgb'] = xgb.predict(X_train)
    
    xgb_rmse = rmse(preds, 'xgb')
    
    print(f'RMSE: {xgb_rmse}')
    
    return preds, xgb_rmse


# In[ ]:


def rmse_table(baseline_rmse, lm_rmse, lasso_rmse, poly_rmse, lassopoly_rmse, xgb_rmse):
    
    rmse_df = pd.DataFrame({'model':['baseline','linear', 'lasso','linear_poly', 'lasso_poly', 'xgb'],
            'rmse':[baseline_rmse, lm_rmse, lasso_rmse, poly_rmse, lassopoly_rmse, xgb_rmse]})
    
    return rmse_df


# In[25]:


def rmse_graph(rmse_df):
    fig, ax = plt.subplots(figsize=(10,7))
    bplot = sns.barplot(x='model',y='rmse', data=rmse_df.sort_values('rmse'))
    plt.ylabel('RMSE')
    plt.xlabel('Model')
    plt.title('RMSE for Each Tested Model')
    plt.ylim(0, 12000)
    ax.bar_label(bplot.containers[0], padding= 6)
    plt.show()


# In[21]:


def val_tests(X_train, y_train, X_val, y_val):
    
    pf = PolynomialFeatures(degree=2)
    pf.fit(X_train, y_train)
    
    X_polynomial = pf.transform(X_train)
    X_val_polynomial = pf.transform(X_val)
    
    lmtwo = LinearRegression()
    lmtwo.fit(X_polynomial, y_train)
    
    lasso = LassoLars(alpha=0)
    lasso.fit(X_train, y_train)
    
    xgb = XGBRegressor(objective='reg:squarederror',n_estimators=20, max_depth=4, 
                   subsample=0.5, colsample_bytree=0.7, seed=42)
    xgb.fit(X_train, y_train)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    
    val_preds = pd.DataFrame({'actual':y_val,
                              'baseline':y_train.mean()})
    
    val_preds['lasso_preds'] = lasso.predict(X_val)

    val_preds['poly_preds'] = lmtwo.predict(X_val_polynomial)

    val_preds['linear_preds'] = lm.predict(X_val)

    val_preds['xgb_preds'] = xgb.predict(X_val)
    
    return val_preds


# In[23]:


def val_rmse(val_preds):
    baseline_rmse = rmse(val_preds, 'baseline')

    lasso_rmse = rmse(val_preds, 'lasso_preds')

    poly_rmse = rmse(val_preds, 'poly_preds')

    linear_rmse = rmse(val_preds, 'linear_preds')

    xgb_rmse = rmse(val_preds, 'xgb_preds')
    
    val_rmse_df = pd.DataFrame({'model':['baseline', 'lasso','poly', 'linear', 'xgb'],
              'rmse':[baseline_rmse, lasso_rmse, poly_rmse, linear_rmse, xgb_rmse]})
    
    return val_rmse_df


# In[24]:


def val_plot(val_rmse_df):
    fig, ax = plt.subplots(figsize=(10,7))
    bplot = sns.barplot(x='model',y='rmse', data=val_rmse_df.sort_values('rmse'))
    plt.ylabel('RMSE')
    plt.xlabel('Model')
    plt.title('RMSE for Each Tested Model')
    plt.ylim(0, 12000)
    ax.bar_label(bplot.containers[0], padding= 6)
    plt.show()


# In[26]:


def test_set(X_train, y_train, X_test, y_test):
    
    xgb = XGBRegressor(objective='reg:squarederror',n_estimators=20, max_depth=4, 
                   subsample=0.5, colsample_bytree=0.7, seed=42)
    
    xgb.fit(X_train, y_train)
    
    test_preds = pd.DataFrame({'actual':y_test,
                          'test_pred':xgb.predict(X_test)})
    
    test_score = round(rmse(test_preds, 'test_pred'), 2)
    
    print(f'The final test RMSE is: {test_score}')


# In[ ]:




