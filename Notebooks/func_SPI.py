#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import xarray as xr
import numpy as np
import sys
import scipy.stats as stats


def calc_SPI_gs_doy(df_gs, doy):
    '''
    Calculates SPI for a given gridcell on a specific day of year
    '''
    df_gs_m = df_gs[df_gs.index.dayofyear == doy]
    # note floc avoids loc being fitted, see Stagge et al. 2015
    params = stats.gamma.fit(df_gs_m.values, loc=0) # fit parameters of gamma distribution to SPI data 
    rv = stats.gamma(*params) # Continuous random variable class, can sample randomly from the gamma                distribution we just fitted  
    
    # Account for zero values (cfr.Stagge et al. 2015))  
    indices_nonzero = np.nonzero(df_gs_m)[0]
    nyears_zero = len(df_gs_m) - np.count_nonzero(df_gs_m)
    ratio_zeros = nyears_zero / len(df_gs_m)

    p_zero_mean = (nyears_zero + 1) / (2 * (len(df_gs_m) + 1))           
    prob_gamma = (df_gs_m * 0 ) + p_zero_mean
    # probability of values, given the Gamma distribution and excluding zeros
    prob_gamma[indices_nonzero] = ratio_zeros+((1-ratio_zeros)*rv.cdf(df_gs_m[indices_nonzero]))
    
    # Transform Gamma probabilities to standard normal probabilities
    prob_std = stats.norm.ppf(prob_gamma)                                   
    prob_std[prob_std>3] = 3
    prob_std[prob_std<-3] = -3 
    return prob_std

def calc_SPI_gs_pentad(df_gs, p):
    '''
    For a given dataframe, the centered single date for the pentad is taken and passed to      calc_SPI_gs_doy(df_gs, doy).
    There 73 pentads of 5 in 365 days, hence p should be in range (0,73)
    '''
    i = np.arange(0,365, 5)[p]
    dates = df_gs.index
    idx_pentad = np.arange(i, dates.size, 365, dtype=int)
    doy = int(np.mean(dates[idx_pentad].dayofyear)) # central dayofyear for the given month {m}
    prob_std = calc_SPI_gs_doy(df_gs, doy)
    return prob_std

def calc_SPI_gs_month(df_gs, month):
    '''
    For a given dataframe,  the centered single date for the {month} is taken and passed to calc_SPI_gs_doy(df_gs, doy).
    There 73 pentads of 5 in 365 days, hence p should be in range (0,73)
    '''
    # to select the SPI for month m (with 1 = jan and 12 = Dec):
    m = month
    dates = df_gs.index
    doy = int(np.mean(dates.dayofyear[dates.month == m])) # central dayofyear for the given month {m}
    prob_std = calc_SPI_gs_doy(df_gs, doy)
    return prob_std

def calc_SPI_from_daily(xarray, SPI_aggr, freq='months'):
    '''
    Function extracts the values of xarray that are not nans 
    and will pass them to calc_SPI_gs_. 
    One can choose to calculate SPI for all days, or for each month or pentad (5-day bins).
    Advisably, stick to monthly or pentads, for computational reasons.
    '''

    # SPI monthly windowed aggregation
    months = [1, 2, 3, 6, 9, 12]
    days = [31, 61, 91, 183, 274, 365]
    SPI_to_days = {months[i]:days[i] for i in range(len(months))}
    SPI_window = SPI_to_days[SPI_aggr]
    dates = pd.to_datetime(xarray.time.values)
    n_yrs = np.unique(dates.year)

    # Finding only the non-NaN gridcells 
    np_pr = np.reshape(xarray.values, (dates.size, -1))
    n_gs_total = np_pr.shape[1]
    mask_NanIsFalse   = ~np.isnan(np_pr) # where are values not NaN 
    # Select only the non-NaN gridcells, and cast to dataframe.
    pr_no_nans  = np_pr[mask_NanIsFalse].reshape( dates.size, -1) 
    df_no_nans = pd.DataFrame(pr_no_nans, index=dates) # make a pandas dataframe
    # Calculate a sum of window size {SPI_window}, using equal weights. 
    df_no_nans = df_no_nans.rolling(window=SPI_window, min_periods=1, center=True, axis=0).sum()
    n_gs_no_nans = int(df_no_nans.columns.size)

    for i, gs in enumerate(df_no_nans.columns):
        if freq == 'daily':
            new_dates = dates
            if i == 0:
                np_tofill = np.zeros( dates.dayofyear.size,  n_gs_no_nans)
            for doy in np.unique(dates.dayofyear):
                mask_doy = dates.dayofyear == doy
                np_tofill[mask_doy,i] = calc_SPI_gs_doy(df_no_nans[gs], doy)
                
        if freq == 'pentads':
            np_tofill = np.zeros( (int(dates.size/5),  n_gs_no_nans)) 
            new_dates = []
            for p, d in enumerate(range(0,365,5)):
                x = np.arange(d, dates.size, 365, dtype=int)
                new_dates.append(dates[x])
                if i == 0:
                    mask_NanIsFalse = mask_NanIsFalse[:new_dates.size]
                    idx_fill = np.arange(p, int(dates.size/5), 73)
                np_tofill[idx_fill,i] = calc_SPI_gs_pentad(df_no_nans[gs], p)
            new_dates = pd.to_datetime(sorted(np.concatenate(new_dates)))
            
        if freq == 'months':
            new_dates = xarray.resample(time='M', 
                                        label='left').mean().time.values 
            new_dates = pd.to_datetime(new_dates + pd.Timedelta('1D'))
            if i == 0:
                mask_NanIsFalse = mask_NanIsFalse[:new_dates.size]
                np_tofill = np.zeros( (new_dates.size,  n_gs_no_nans)) 
            for m in range(1,13):
                idx_fill = np.arange(m-1, new_dates.size, 12)
                np_tofill[idx_fill,i] = calc_SPI_gs_month(df_no_nans[gs], m)

            

    # return values in there original position
    np_time_space = np.zeros( (new_dates.size, n_gs_total) )
    np_time_space[mask_NanIsFalse] = np_tofill.ravel()
    np_latlon = np.reshape(np_time_space, (new_dates.size, 
                                           xarray.latitude.size, 
                                           xarray.longitude.size))

    # return xarray with same dimensions:
    dates = pd.to_datetime(new_dates)
    SPI_xr = xr.DataArray(np_latlon, coords=[dates, 
                                             xarray.latitude, xarray.longitude],
                            dims=['time', 'latitude', 'longitude'])
    mask_nans = mask_NanIsFalse[0].reshape(xarray.latitude.size, 
                                           xarray.longitude.size)
    mask = (('latitude', 'longitude'), mask_nans)
    SPI_xr['mask'] = mask
    SPI_xr.attrs['units'] = '[std]'
    return SPI_xr

