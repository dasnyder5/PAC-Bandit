#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:48:00 2021

@author: dasnyder
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import leastsq as LS


def lsFunExp(x, tt, yy): 
    """
    Function to be called by scipy.optimize.least_squares

    Parameters
    ----------
    x : np.array (float)
        Parameters of the regression problem (here, constant + coefficient)
    tt : np.array (float)
        independent variable
    yy : np.array (float)
        Dependent (target) variable

    Returns
    -------
    Residuals of regression; array with length equal to len(tt) = len(yy)

    """
    y_est = x[0] + x[1]*tt
    rr = yy-y_est
    
    return rr


def lsFunLin(x, tt, yy): 
    """
    Function to be called by scipy.optimize.least_squares

    Parameters
    ----------
    x : np.array (float)
        Parameters of the regression problem (here, constant + coefficient)
    tt : np.array (float)
        independent variable
    yy : np.array (float)
        Dependent (target) variable

    Returns
    -------
    Residuals of regression; array with length equal to len(tt) = len(yy)

    """
    y_est = x[0]*tt**x[1]
    rr = yy-y_est
    
    return rr

def curveFitUtil(file_name, optVal, sP=1, plotFlag=True, verbose=False):
    """
    

    Parameters
    ----------
    filename : string
        Name of the file to read into the pandas dataframe (must be .csv)
    optVal : float 
        Optimal hindsight value (e.g. mean of the best arm for MAB). Should be 
        bounded in [0, 1] (certainly for the finite problem)
    sP : int > 0, optional
        Starting point in the array used for regression. This can be chosen in 
        order to avoid early fluctuations and better estimate the true 
        convergence parameter b. The default is 1 (the whole array).
    plotFlag : boolean, optional
        Choose whether or not to display the function approximation result. The 
        default is True. 
    verbose : boolean, optional
        Choose whether or not to print out the domain features. 

    Returns
    -------
    xs : np.array (float)
        The vector of optimal parameters [a, b] where the convergence is of the 
        form Loss_avg = a*t^b [desire b -> -1 for logarithmic regret]
    """
    data = pd.read_csv(file_name)
    # BanditMax = 0.98335892
    # sP = 20 # Point in the array at which to start the curve fit
    X0 = data['Step'].values[sP:]
    Y0 = data['Value'].values[sP:]
    if np.min(X0) < 2: 
        c = 2 - np.min(X0)
    else: 
        c = 0
    
    X1 = X0 + c
    Y1 = optVal - Y0
    X2 = np.log(X1)
    Y2 = np.log(Y1)
    # (This transforms ax^b into log(a) + b*log(x))
    # Run linear regression and recover b directly and exponentiate estimate to get a*
    # This might be biased in the estimates, so we will run a separate version 
    # that analyzes ax^b directly
    
    xs1, mincost1 = LS(lsFunLin, ((0.5, -0.95)), args=(X1, Y1))
    xs2, mincost2 = LS(lsFunExp, ((0.1, -0.9)), args=(X2, Y2))

    if verbose: 
        print(data.shape)
        print(data.head())
        print(np.min(X1))
        print(np.max(X1))
        print(np.min(Y1))
        print(np.max(Y1))
        print(xs1)
        print(xs2)
        breakpoint()

    XX = np.exp(X2)
    # YY = np.exp(Y2)
    YYest1 = xs1[0]*X1**xs1[1]
    YYest2 = np.exp(xs2[0])*(XX**xs2[1])
    XX = XX - c
    
    # Plot true values and estimate
    plt.figure()
    plt.plot(X0, Y1, 'k')
    plt.plot(X0, YYest2, 'r--')
    plt.plot(X0, YYest1, 'b--')
    plt.title('Function approximation of regret metric')
    plt.xlabel('Step')
    plt.ylabel('Difference to optimal')
    plt.legend(('True function', 'Exp-Est function', 'Lin-Est function'))
    plt.show()

    return

