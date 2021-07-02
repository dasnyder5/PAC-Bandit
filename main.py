#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:01:39 2021

@author: dasnyder
"""
from env.bandit_env import SFBandit
from algs.band_algs import Random, UCB, KLUCB
import numpy as np
from matplotlib import pyplot as plt
import time

nRew = 40
sleepFlag = False
nAlgs = 3
k = 10
T = 1000
REW = np.zeros((nAlgs, nRew))
iterNum = 0

SFB1 = SFBandit(k, T=T)
SFB1.assign_rewards()
SFB2 = SFBandit(k, T=T)
SFB2.assign_rewards()
SFB3 = SFBandit(k, T=T, printFlag=True)
SFB3.assign_rewards()
ALG1 = UCB(k, T=T)
ALG2 = Random(k, T=T)
ALG3 = KLUCB(k, T=T)

for t in range(T): 
    a1 = ALG1.choose_arm(t)
    a2 = ALG2.choose_arm(t)
    a3 = ALG3.choose_arm(t)
    r1 = SFB1.step(a1)
    r2 = SFB2.step(a2)
    r3 = SFB3.step(a3)
    ALG1.update(a1, t+1, r1)
    ALG2.update(a2, t+1, r2)
    ALG3.update(a3, t+1, r3)
    
    if (t % (T/nRew) == ((T/nRew) - 1)): 
        print(t)
        print('Current empirical reward for Random Alg: ', ALG2.Rcum/t)
        print('Current empirical reward for UCB Alg: ', ALG1.Rcum/t)
        print('Current empirical reward for KLUCB Alg: ', ALG3.Rcum/t)
        print('True optimal achievable reward in hindsight: ', np.max(SFB2.Rhind[:,t])/t)
        print('True optimal achievable reward [checksum]: ', np.max(SFB1.Rhind[:,t])/t)
        print('True optimal achievable reward [checksum 2]: ', np.max(SFB3.Rhind[:,t])/t)
        REW[0,iterNum] = ALG1.Rcum/t
        REW[1,iterNum] = ALG2.Rcum/t
        REW[2,iterNum] = ALG3.Rcum/t
        iterNum += 1
        if sleepFlag: 
            time.sleep(1)

xlimits = np.zeros(2)
xlimits[1] += (nRew - 1)*(T/nRew)
 
plt.figure(1)
plt.plot(np.arange(nRew)*(T/nRew), REW[0,:], 'r')
plt.plot(np.arange(nRew)*(T/nRew), REW[1,:], 'b')
plt.plot(np.arange(nRew)*(T/nRew), REW[2,:], 'k')
plt.title('Cumulative Reward of Each Bandit Algorithm')
plt.xlabel('Number of Iterations')
plt.ylabel('Average Cumulative Reward')
plt.plot(xlimits, np.max(SFB1.r)*np.ones(2), 'g--')
plt.ylim([0.6, 1])
plt.legend(('UCB', 'Random', 'KL-UCB', 'Best Arm'))
plt.show()
