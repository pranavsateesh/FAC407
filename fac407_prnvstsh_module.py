import pandas as pd
import datetime 
import matplotlib.pyplot as plt 
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro, anderson
import scipy.stats as stats
import numpy as np
import os
import statistics
import time
from sklearn import linear_model
import statsmodels.api as sm
import math


def HW2_Function():
    data=pd.read_excel('HW5.xlsx',engine='openpyxl')
    lm = linear_model.LinearRegression()
    model = lm.fit(data[['Rmt*']],data[['Rjt*']])
    data.plot(kind='scatter', x= 'Rjt*', y='Rmt*')
    Stockrate= pd.DataFrame(data['Rjt*'])
    Marketrate= pd.DataFrame(data['Rmt*'])
    model=lm.fit(Marketrate,Stockrate)
    print(lm.intercept_)
    print(lm.coef_)
    results = sm.OLS(Stockrate,Marketrate).fit()
    print(results.summary()) 
    Stockrate= pd.DataFrame(data['Rjt*'])
    Marketrate= pd.DataFrame(data['Rmt*'])
    Marketrate = sm.add_constant(Marketrate)
    results = sm.OLS(Stockrate,Marketrate).fit()
    print(results.summary()) 
    print(results.conf_int())


def normal_test_sw():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')

    stat, p = shapiro(abbot["Close"])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Abbot :: Sample looks Gaussian :: Shapiro-Wilk Test')
    else:
        print('Abbot :: Sample does not look Gaussian :: Shapiro-Wilk Test')

    stat, p = shapiro(mrf["Close"])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('MRF :: Sample looks Gaussian :: Shapiro-Wilk Test')
    else:
        print('MRF :: Sample does not look Gaussian :: Shapiro-Wilk Test')

    stat, p = shapiro(shreecem["Close"])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Shree Cements :: Sample looks Gaussian :: Shapiro-Wilk Test')
    else:
        print('Shree Cements :: Sample does not look Gaussian :: Shapiro-Wilk Test')
    time.sleep(5)
def normal_test_ad():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')
    
    result = anderson(abbot["Close"])
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
	    sl, cv = result.significance_level[i], result.critical_values[i]
	    if result.statistic < result.critical_values[i]:
		    print('%.3f: %.3f, data looks normal :: Anderson-Darling Test :: Abbot' % (sl, cv))
	    else:
		    print('%.3f: %.3f, data does not look normal :: Anderson-Darling Test :: Abbot' % (sl, cv))
    
    result = anderson(mrf["Close"])
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
	    sl, cv = result.significance_level[i], result.critical_values[i]
	    if result.statistic < result.critical_values[i]:
		    print('%.3f: %.3f, data looks normal :: Anderson-Darling Test  :: MRF' % (sl, cv))
	    else:
		    print('%.3f: %.3f, data does not look normal :: Anderson-Darling Test :: MRF' % (sl, cv))
    
    result = anderson(shreecem["Close"])
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
	    sl, cv = result.significance_level[i], result.critical_values[i]
	    if result.statistic < result.critical_values[i]:
		    print('%.3f: %.3f, data looks normal :: Anderson-Darling Test :: Shree Cements' % (sl, cv))
	    else:
		    print('%.3f: %.3f, data does not look normal :: Anderson-Darling Test :: Shree Cements' % (sl, cv))
    time.sleep(5)
def distribution_returns_normdist():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    abbot["Return"] = abbot["Close"].pct_change().fillna(method='bfill')
    abbot['Return'].hist(bins=50, ax=ax1)
    plt.title("Abbot :: Distribution")
    plt.show()
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    stats.probplot(abbot['Return'], dist='norm', plot=ax1) 
    plt.title("Abbot :: Normal Plot")
    plt.show()

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    mrf["Return"] = mrf["Close"].pct_change().fillna(method='bfill')
    mrf['Return'].hist(bins=50, ax=ax1)
    plt.title("MRF :: Distribution")
    plt.show()
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    stats.probplot(mrf['Return'], dist='norm', plot=ax1) 
    plt.title("MRF :: Normal Plot")
    plt.show()

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    shreecem["Return"] = shreecem["Close"].pct_change().fillna(method='bfill')
    shreecem['Return'].hist(bins=50, ax=ax1)
    plt.title("Shree Cements :: Distribution")
    plt.show()
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    stats.probplot(shreecem['Return'], dist='norm', plot=ax1) 
    plt.title("Shree Cements :: Normal Plot")
    plt.show()

def plot_the_closing():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')
    
    plt.plot(abbot["Close"])
    plt.plot(mrf["Close"])
    plt.plot(shreecem["Close"])
    plt.legend(["Abbot","MRF","Shree Cements"])
    plt.title("Closing price vs days from 1/1/2009")
    plt.show()

def plot_the_net_returns():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')

    
    abbot["Return"] = abbot["Close"].pct_change().fillna(method='bfill')
    mrf["Return"] = mrf["Close"].pct_change().fillna(method='bfill')
    shreecem["Return"] = shreecem["Close"].pct_change().fillna(method='bfill')
    plt.plot(abbot["Return"])
    plt.show()
    plt.plot(mrf["Return"])
    plt.show()
    plt.plot(shreecem["Return"])
    #plt.legend(["Abbot","MRF","Shree Cements"])
    #plt.title("Net-returns vs days from 1/1/2009")
    plt.show()

def plot_the_gross_returns():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')

    abbot["Return"] = abbot["Close"].pct_change().fillna(method='bfill')
    mrf["Return"] = mrf["Close"].pct_change().fillna(method='bfill')
    shreecem["Return"] = shreecem["Close"].pct_change().fillna(method='bfill')
    abbot["Gross"] = abbot["Return"] + 1
    mrf["Gross"] = mrf["Return"] + 1
    shreecem["Gross"] = shreecem["Return"] + 1
    plt.plot(abbot["Gross"])
    plt.plot(mrf["Gross"])
    plt.plot(shreecem["Gross"])
    plt.legend(["Abbot","MRF","Shree Cements"])
    plt.title("Gross-returns vs days from 1/1/2009")
    plt.show()

def plot_the_logret():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')

    abbot["Return"] = abbot["Close"].pct_change().fillna(method='bfill')
    mrf["Return"] = mrf["Close"].pct_change().fillna(method='bfill')
    shreecem["Return"] = shreecem["Close"].pct_change().fillna(method='bfill')

    abbot["Log"] = np.log(abbot["Return"])
    mrf["Log"] = np.log(mrf["Return"])
    shreecem["Log"] = np.log(shreecem["Return"])
    plt.plot(abbot["Log"])
    plt.plot(mrf["Log"])
    plt.plot(shreecem["Log"])
    plt.legend(["Abbot","MRF","Shree Cements"])
    plt.title("Log-returns vs days from 1/1/2009")
    plt.show()

def plot_the_adjclosing():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')

    plt.plot(abbot["Adj Close"])
    plt.plot(mrf["Adj Close"])
    plt.plot(shreecem["Adj Close"])
    plt.legend(["Abbot","MRF","Shree Cements"])
    plt.title("Adjusted-Closing price vs days from 1/1/2009")
    plt.show()

def get_mean_median_mode_logret():    
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')
    abbot["Return"] = abbot["Close"].pct_change().fillna(method='bfill')
    mrf["Return"] = mrf["Close"].pct_change().fillna(method='bfill')
    shreecem["Return"] = shreecem["Close"].pct_change().fillna(method='bfill')

    abbot["Log"] = np.log(abbot["Return"])
    mrf["Log"] = np.log(mrf["Return"])
    shreecem["Log"] = np.log(shreecem["Return"])
    
    abbot_r = list(abbot["Return"])
    mrf_r = list(mrf["Return"])
    shreecem_r = list(shreecem["Return"])
    abbot_l = [] 
    mrf_l = []
    shreecem_l = []
    count_abbot = 0
    count_mrf = 0 
    count_shreecem = 0 
    for i in abbot_r:
        abbot_l.append(np.log(i))
    for i in mrf_r:
        mrf_l.append(np.log(i))
    for i in shreecem_r:
        shreecem_l.append(np.log(i))

    abbot_mean = statistics.mean(abbot_l)
    abbot_median = statistics.median(abbot_l)
    abbot_mode = statistics.mode(abbot_l)
    
    mrf_mean = statistics.mean(mrf_l)
    mrf_median = statistics.median(mrf_l)
    mrf_mode = statistics.mode(mrf_l)

    shreecem_mean = statistics.mean(shreecem_l)
    shreecem_median = statistics.median(shreecem_l)
    shreecem_mode = statistics.mode(shreecem_l)

    print('Abbot :: mean=%.3f, median=%.3f, mode=%.3f' % (abbot_mean, abbot_median, abbot_mode))
    print('MRF :: mean=%.3f, median=%.3f, mode=%.3f' % (mrf_mean, mrf_median, mrf_mode))
    print('Shree Cements :: mean=%.3f, median=%.3f, mode=%.3f' % (shreecem_mean, shreecem_median, shreecem_mode))

def correlation():
    abbot = pd.read_csv('ABBOTINDIA.NS.csv')
    mrf = pd.read_csv('MRF.NS.csv')
    shreecem = pd.read_csv('SHREECEM.NS.csv')
    abbot["Return"] = abbot["Close"].pct_change().fillna(method='bfill')
    mrf["Return"] = mrf["Close"].pct_change().fillna(method='bfill')
    shreecem["Return"] = shreecem["Close"].pct_change().fillna(method='bfill')

    abbot_mrf = abbot["Return"].corr(mrf["Return"])
    abbot_shreecem = abbot["Return"].corr(shreecem["Return"])
    mrf_shreecem = mrf["Return"].corr(shreecem["Return"]) 


    abbot["Log"] = np.log(abbot["Return"])
    mrf["Log"] = np.log(mrf["Return"])
    shreecem["Log"] = np.log(shreecem["Return"])

    abbot_mrf_log = abbot["Log"].corr(mrf["Log"])
    abbot_shreecem_log = abbot["Log"].corr(shreecem["Log"])
    mrf_shreecem_log = mrf["Log"].corr(shreecem["Log"])
    print("abbot:mrf:")
    print(abbot_mrf)
    print("abbot:shreecem:")
    print(abbot_shreecem)
    print("mrf:shreecem:")
    print(mrf_shreecem)
    print("abbot:mrf_log:")
    print(abbot_mrf_log)
    print("abbot:shreecem_log:")
    print(abbot_shreecem_log)
    print("mrf:shreecem_log:")
    print(mrf_shreecem_log)
    time.sleep(10)


HW2_Function()