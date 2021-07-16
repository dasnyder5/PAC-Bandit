#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:06:15 2021

@author: dasnyder
"""
import numpy as np



class Bandit: 
    """
    Most basic and general bandit environment object. Encompasses potential
    for the stochastic or adversarial setting (we will focus on stochastic 
    setting here). Similarly allows generalization to continuous bandits with 
    a generating function accessed via assign_rewards.
    
    Methods: 
        __init__
        step 
        assign_rewards
    """
    
    
    def __init__(self, T): 
        self.T = T

#    def step(self, a): 
#        return None
#
#    def assign_rewards(self):
#        return None
        

class FiniteBandit(Bandit):

    def __init__(self, k, T): 
        super().__init__(T)
        
        # Public Information for the Player
        self.k = k                  # Number of arms (finite!)
        self.t = int(0)             # Current timestep (reset to 0 in init)
        self.n = np.zeros(k)        # Number of pulls of each arm
        self.R = np.zeros(k)        # Cumulative reward of each arm 
        self.rhat = np.zeros(k)     # Empirical mean reward for each arm
        self.Rcum = 0               # Cumulative observed reward
        
        # Private Information Unobservable By the Player
        self.Rhind = np.zeros((k, self.T)) # Cumulative reward for each arm (unobserved)

    def step(self, a): 
        raise NotImplementedError()
    
    def assign_rewards(self):
        raise NotImplementedError()


class SFBandit(FiniteBandit): 
    
    def __init__(self, k, T=1000, r=None, dist='Bernoulli', seed=None, repeatable=True, printFlag=False): 
        super().__init__(k, T)
        
        self.rt = np.zeros((k, T))
        
        # Assign repeatable but random reward means based on r and seed: 
        if r is not None: # Assign the user-given rewards + check for issues
            self.r = r
            if (np.max(r) > 1 or np.min(r) < 0): 
                raise ValueError('Invalid Reward - Outside [0, 1] Bounds')
            if (len(r) != k): 
                raise ValueError('Reward vector length does not match' +
                                 ' the number of arms')
        else:  # Assign random rewards
            if seed is not None: # Use user-given seed
                np.random.seed(seed)
                self.r = np.random.rand(k)
            else: 
                if repeatable:
                    np.random.seed(123456)
                else: 
                    np.random.seed()

                self.r = 0.5 + 0.5*np.random.rand(k)
        
        # Store the distribution type
        if dist == 'Bernoulli': 
            self.rdiststr = 'Bernoulli'
        else: 
            raise ValueError('No valid reward distribution specified!')
            
        if printFlag: 
            print('Reward vector: ', self.r)
            print('Optimal arm index: ', np.argmax(self.r))


    def step(self, a): 
        # Check that the action is valid, and if so, proceed accordingly
        if (isinstance(a, int) and 0 <= a and self.k > a): 
            # Do all of the logic of the stepping
            if self.t < self.k: 
                a = self.t 
            
            # self.rt[a] = is the reward assigned to arm a by assign_rewards()
            reward_t = self.rt[a, self.t]
            self.n[a] = self.n[a] + 1
            self.R[a] = self.R[a] + reward_t
            self.Rcum = self.Rcum + reward_t
            self.t = self.t + 1
            
            if self.t >= self.k: 
                self.rhat = self.R/self.n

        else: 
            raise ValueError('Action must be an integer in the set {0, 1, ..., k-1}')
        
        return reward_t
    
    def assign_rewards(self):
        if self.rdiststr == 'Bernoulli': 
            for k in range(self.k): 
                self.rt[k,:] = np.random.binomial(1,self.r[k], self.T)
                self.Rhind[k,:] = np.cumsum(self.rt[k,:])
        else: 
            raise NotImplementedError('Have not incorporated other distributions')
    
        return True


class AFBandit(FiniteBandit): 
    
    def __init__(self, k, T): 
        super().__init__(k, T)

    def step(self, a): 
        raise NotImplementedError()
    
    def assign_rewards(self):
        raise NotImplementedError()


class InfBandit(Bandit): 
    
    def __init__(self, T=1000, r=None): 
        super().__init__(T)
        
        # Public Information for the Player
        self.k = k                  # Number of arms (finite!)
        self.t = int(0)             # Current timestep (reset to 0 in init)
        self.n = np.zeros(k)        # Number of pulls of each arm
        self.R = np.zeros(k)        # Cumulative reward of each arm 
        self.rhat = np.zeros(k)     # Empirical mean reward for each arm
        self.Rcum = 0               # Cumulative observed reward
        
        # Private Information Unobservable By the Player
        self.Rhind = np.zeros((k, self.T)) # Cumulative reward for each arm (unobserved)
        return

    def step(self, a): 
        raise NotImplementedError()
    
    def generate_reward(self):
        raise NotImplementedError()


class SIBandit(InfBandit):
    
    def __init__(self, k, T): 
        super().__init__(k, T)

    def step(self, a): 
        raise NotImplementedError()
    
    def assign_rewards(self):
        raise NotImplementedError()
    
    
    
        
        
        
        
