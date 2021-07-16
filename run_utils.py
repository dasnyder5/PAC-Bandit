#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:33:37 2021

@author: dasnyder
"""
from utils import curveFitUtil

FN = 'KLUCB_convergence.csv'
FN2 = 'PAC-Bandit_convergence.csv'
optVal = 0.98335892
sP = 20

curveFitUtil(FN, optVal, sP=sP, verbose=True)
breakpoint()
curveFitUtil(FN2, optVal, sP=sP, verbose=True)