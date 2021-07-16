#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:37:33 2021

@author: dasnyder
"""

import numpy as np
from matplotlib import pyplot as plt
import time


def KLUCB_Newton(N, p, t, c=1, q=None, plotFlag=False):
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
    SetRewardsLessThanOne = True

    k = len(N)
    if SetRewardsLessThanOne:
        if q is None:
            for i in range(k): 
                if p[i] >= 0.9995: 
                    p[i] = 0.9995
    
            q = p + 0.5*(np.ones(k)-p)
    
    # Add in if statement to check for empirical rewards of 1
    #
    # if SetRewardsLessThanOne is True, then this will be skipped (TODO: check the preceding statement)
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
    #print('NO ARM REMAINS PERFECT! - KL-UCB')
    #print('Time is: ', t)
    # Initialize termination condition variables
    DQ = np.ones(k)
    it_num = 0
    # Set termination tolerances
    DQ_tol = 0.000005
    max_iter = 100
    # Observe convergence behavior
    DQ_norm = np.zeros(max_iter+1)+0.0000001
    DQ_norm[0] = np.max(DQ)             # This is in effect an infinity-norm

    # While convergence is False...
    conv = False

    while conv is False:
        for i in range(k):
            if np.abs(DQ[i]) > DQ_tol:
                print()
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
    
    for i in range(k): 
        if q[i] >= 0.9995: 
            q[i] = 0.9995

    if t%200 == 0: 
        print('Number of iterations: ', it_num)
        print('q: ', q)
        breakpoint()

    q_inds = np.argmax(q)
    if np.size(q_inds) > 1:
        q_tmp = int(np.random.randint(np.size(q_inds)))
        q_ind = int(q_inds[q_tmp])
    else:
        q_ind = int(q_inds)

    return q_ind, q, oneFlag       # q_index of highest UCB reward


def KL(p, q):
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


def pacOPT(n, rhat, t, K, delta=0.001):
    """
    Optimize the PAC-Bayes policy based on the necessary parameters. 

    Parameters
    ----------
    n : np.array of size K, integer elements
        Vector of length K encoding the number of times each arm has been pulled.
    rhat : np.array of size K, float elements
        Vector of length K encoding the empirical reward so far of each arm.
    t : int 
        Iteration number of the MAB problem
    K : int
        Number of arms in the MAB problem.
    delta : float in (0, 1)
        Confidence parameter for the PAC-Bayes optimization.

    Returns
    -------
    alpha : np.array of size K, float elements
        Vector encoding the randomized optimal policy, assigning probability 
        mass to each arm

    """

    # define the nprime vector as elementwise (1/n_i)
    if np.min(n) < 1:
        raise ValueError('Invalid n-vector; some element is less than 1')

    nprime = np.zeros(K)
    for k in range(K):
        nprime[k] = 1./n[k]

    # Solve using KKT conditions
    #
    # The key condition is np[k]*(log(C) - log(alpha * np)) + rhat[k] = v* - L*[k]
    #
    # So, v* is the max over k of the LHS (because L*[k] >= 0), and is equal to
    # the overall upper confidence bound across all arms. Therefore, we choose
    # that arm (and break ties randomly)
    #

    tmp = 0
    tmp_ind = -1
    sol = []  # List of all solutions with equal upper confidence bounds

    '''
    # First check for any empirically perfect arms
    if np.max(rhat) >= 1: 
        q_inds = np.argmax(rhat)
        if np.size(q_inds) > 1:
            q_tmp = int(np.random.randint(np.size(q_inds)))
            q_ind = int(q_inds[q_tmp])
        else:
            q_ind = int(q_inds)

        alpha = np.zeros(K)
        alpha[q_ind] = 1
        # print('Still at empirical reward of 1!')
        return alpha 
    
    #print('No arm remains Perfect - PAC-UCB')
    #print('Round number: ', t)
    '''

    # Iterate through to find the solution of maximal upper conf. bound
    # List is appended if the UCBs are equal (to allow for tie-breaking later)
    # Assumes that no arm remains empirically perfect
    for k in range(K):
        alpha = np.zeros(K)
        alpha[k] = 1
        lhs = nprime[k]*(np.log(2*K/delta) - 1 - np.log(np.dot(alpha, nprime))) + rhat[k]
        
        if lhs > tmp:
            tmp = lhs
            tmp_ind = k
            # Reset sol list if it is longer than one element
            sol = []
            sol.append(tmp_ind)
        elif lhs == tmp:
            # Append additional solution to sol list
            sol.append(k)

    # Tie-breaking scheme
    # Assign equal weight to all arms having maximal UCB
    kk = len(sol)
    #print('pacOPT solution length: ', kk)
    #breakpoint()
    alpha = np.zeros(K)
    for k in range(K):
        tmp_log = False
        for i in range(kk):
            if sol[i] == k:
                tmp_log = True

        if tmp_log:
            alpha[k] = 1./kk
            
    # return the randomized policy alpha (that will generally be deterministic)
    return alpha
