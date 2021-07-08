#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:08:52 2021

@author: dasnyder
"""
import numpy as np
from algs.utils import KL, KLUCB_Newton

class BandAlg: 
    
    def __init__(self, T): 
        self.T = T
        self.t = int(0)


class FBandAlg(BandAlg): 

    def __init__(self, k, T): 
        super().__init__(T)

        self.k = k                  # Number of arms
        self.n = np.zeros(k)        # Number of pulls of each arm
        self.R = np.zeros(k)        # Cumulative reward of each arm 
        self.rhat = np.zeros(k)     # Empirical mean reward for each arm
        self.Rcum = 0               # Cumulative observed reward
    
    def choose_arm(self, t): 
        raise NotImplementedError()

    def update(self, reward_t): 
        raise NotImplementedError()


class Random(FBandAlg): 

    def __init__(self, k, T=1000): 
        super().__init__(k, T)
        
    def choose_arm(self, t):
        if t < self.k: 
            return int(t)
        else:
            return int(np.random.randint(self.k))

    def update(self, a, t, reward_t):
        self.t = t

        self.n[a] = self.n[a] + 1

        self.R[a] = self.R[a] + reward_t
        self.Rcum = self.Rcum + reward_t

        if t >= self.k: 
            self.rhat = self.R/self.n

        return True


class UCB(FBandAlg): 
    
    def __init__(self, k, T=1000, c=1, delta=0.005): 
        super().__init__(k, T)

        self.c = c
        self.delta = delta
        self.iota = np.log(k*T/delta)
        self.bonus = self.c*np.sqrt(self.iota/np.ones(k))

    def choose_arm(self, t): 
        if t < self.k: 
            a = int(t)
        else: 
            armchoice = np.argmax(self.bonus + self.rhat)
            if np.size(armchoice) > 1: 
                # Account for multiple solutions by randomly breaking ties
                a_ind = int(np.random.randint(np.size(armchoice)))
                a = int(armchoice[a_ind])
            else: 
                a = int(armchoice)

        return a

    def update(self, a, t, reward_t): 

        self.t = t

        self.n[a] = self.n[a] + 1
        self.bonus[a] = self.c*np.sqrt(self.iota/(self.n[a]))

        self.R[a] = self.R[a] + reward_t
        self.Rcum = self.Rcum + reward_t

        if t >= self.k: 
            self.rhat = self.R/self.n

        return True


class KLUCB(FBandAlg): 
    
    def __init__(self, k, T=1000, c=1): 
        super().__init__(k, T)

        self.c = c
        self.q_tmp = np.ones(self.k)
        self.useQFlag = False
        
    # def make_bonus(self): 
    #     self.bonus = self.c * np.sqrt(self.iota/self.n)
        
    def choose_arm(self, t): 
        if t < self.k: 
            a = int(t)
        else: 
            if self.useQFlag: 
                a, q_tmp, oneFlag = KLUCB_Newton(self.n, self.rhat, t, c=self.c, q=self.q_tmp) #, plotFlag=True)
            else: 
                a, q_tmp, oneFlag = KLUCB_Newton(self.n, self.rhat, t, c=self.c) #, plotFlag=True)

            self.q_tmp = q_tmp
        return a

    def update(self, a, t, reward_t): 
        
        if t%10 == 0: 
            print('Current time iteration: ', t)
        self.t = t
        
        self.n[a] = self.n[a] + 1
        
        self.R[a] = self.R[a] + reward_t
        self.Rcum = self.Rcum + reward_t
        
        if t >= self.k: 
            self.rhat = self.R/self.n
            if np.max(self.rhat < 1): 
                self.useQFlag = True
                if np.max(self.q_tmp) >= 1: 
                    self.q_tmp = self.rhat + 0.5*(np.ones(self.k)-self.rhat)
        
        return True
    
class pacUCB(FBandAlg): 
    
    def __init__(self, k, T=1000, c=1, pr=None): 
        super().__init__(k, T)

        self.c = c
        if pr is None:
            self.pr = (1./k)*np.ones(k)
        else: 
            if (np.sum(pr) != 1 or np.min(pr) < 0):
                raise ValueError('Invalid pmf')
            self.pr = pr

        self.q_tmp = np.ones(self.k)
        self.useQFlag = False
        
    # def make_bonus(self): 
    #     self.bonus = self.c * np.sqrt(self.iota/self.n)
        
    def choose_arm(self, t): 
        if t < self.k: 
            a = int(t)
        else: 
            if self.useQFlag: 
                a, q_tmp, oneFlag = KLUCB_Newton(self.n, self.rhat, t, c=self.c, q=self.q_tmp) #, plotFlag=True)
            else: 
                a, q_tmp, oneFlag = KLUCB_Newton(self.n, self.rhat, t, c=self.c) #, plotFlag=True)

            self.q_tmp = q_tmp
        return a

    def update(self, a, t, reward_t): 
        
        if t%10 == 0: 
            print('Current time iteration: ', t)
        self.t = t
        
        self.n[a] = self.n[a] + 1
        
        self.R[a] = self.R[a] + reward_t
        self.Rcum = self.Rcum + reward_t

        if t >= self.k: 
            self.rhat = self.R/self.n
            if np.max(self.rhat < 1): 
                self.useQFlag = True
                if np.max(self.q_tmp) >= 1: 
                    self.q_tmp = self.rhat + 0.5*(np.ones(self.k)-self.rhat)
        
        return True