#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:37:33 2021

@author: dasnyder
"""

import numpy as np
from matplotlib import pyplot as plt
import time


def KLUCB_Newton(N, p, t, c=1, q=None, plotFlag = False): 
    """
    Function using Newton iterations (second-order method) to converge to the 
    maximal q for the KL-UCB optimization. 

    Parameters
    ----------
    N : Array of integers
        Number of times each arm in {0, 1, ..., k-1} has been pulled thus far
    p : Array of floats
        Empirical mean rewards for each arm through time t-1
    t : int
        Current iteration of the bandit game
    c : float
        Constant term for the log(log(t)) regularization in the optimization
    q : Array of floats
        Array of previous confidence bounds from earlier bandit iterations t'<t
    plotFlag : boolean
        Determines whether a plot of the convergence rate is to be undertaken
        within the optimization routine (debugging purposes)

    Returns
    -------
    q_ind : int
        Arm index of the highest UCB reward thus far [ie choice of action]
        
    q : Array of floats
        Current Upper Confidence Bounds for each arm, to be input as the 
        initial guess for each future iteration. 
        
    oneFlag : boolean
        True if the UCB on any of the arms is at least 1

    """
    oneFlag = False
    k = len(N)
    if q is None:
        q = p # + 0.5*(np.ones(k)-p)
    
    # Add in if statement to check for empirical rewards of 1
    if np.max(q) >= 1: 
        oneFlag = True
        q_inds = np.argmax(q)
        if np.size(q_inds) > 1: 
            q_tmp = int(np.random.randint(np.size(q_inds)))
            q_ind = int(q_inds[q_tmp])
        else: 
            q_ind = int(q_inds)
        
        # print('Still at empirical reward of 1!')
        return q_ind, q, oneFlag

    # (else: )    
    L0 = (np.log(t) + c*np.log(np.log(t)))/N
    # print('NO ARM REMAINS PERFECT!')
    # Initialize termination condition variables
    DQ = np.ones(k)
    it_num = 0
    # Set termination tolerances
    DQ_tol = 0.0001
    max_iter = 100
    # Observe convergence behavior
    DQ_norm = np.zeros(max_iter+1)+0.0000001
    DQ_norm[0] = np.max(DQ)             # This is in effect an infinity-norm
    
    # While convergence is False...
    conv = False
    
    while conv is False: 
        for i in range(k): 
            if np.abs(DQ[i]) > DQ_tol:
                C0p = KL(p[i], q[i])-L0[i]
                C1 = (q[i]-p[i])/(q[i]*(1-q[i]))
                C2 = (q[i]**2 + p[i] - 2*p[i]*q[i])/(q[i]**2 * (1-q[i])**2)
                
                disc_tmp = C1**2 - 4*C2*C0p
                if disc_tmp >= 0:
                    dq = (-C1 + np.sqrt(disc_tmp))/(2 * C2)
                else: 
                    dq = -C1/(2 * C2)

                if (q[i] + dq >= 1): 
                    #dq = (-C1 - np.sqrt(C1**2 - 4*C2*C0p))/(2 * C2)
                    dq = 0.9*(1-q[i])
                    
                DQ[i] = dq
                q[i] = q[i] + dq
        
        if np.max(np.abs(DQ)) <= 2*DQ_tol: 
            conv = True
        elif it_num >= max_iter-1: 
            conv = True
        it_num += 1
        DQ_norm[it_num] = np.max(np.abs(DQ))

    if plotFlag: 
        plt.figure(0)
        plt.semilogy(np.arange(max_iter+1), DQ_norm)
        plt.title('Norm of changes in q vs iterations')
        plt.xlabel('Iteration Number')
        plt.ylabel('Inf-Norm of Update Size')
        plt.show()
        # time.sleep(2)

    q_inds = np.argmax(q)
    if np.size(q_inds) > 1: 
        q_tmp = int(np.random.randint(np.size(q_inds)))
        q_ind = int(q_inds[q_tmp])
    else: 
        q_ind = int(q_inds)

    return q_ind, q, oneFlag       # q_index of highest UCB reward


def KL(p,q): 
    if q <= 0.00000001 or q >= 0.99999999: 
        print(q)
        raise ValueError('q outside range (0, 1)')
    else:
        if p <= 0.00000001: 
            return (1-p)*np.log((1-p)/(1-q))
        elif p >= 0.99999999:
            return p*np.log(p/q)
        else:
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q)) 
