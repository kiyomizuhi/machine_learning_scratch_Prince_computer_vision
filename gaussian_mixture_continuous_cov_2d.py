#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 03:31:30 2017

@author: hiroyukiinoue
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def generate_nice_covariance(k):
    ''' this function outputs an covariance matrices avoiding generating too much squashed ellipses'''
    sg = np.zeros((k, 2, 2))
    for i in range(k):
        eig = 2*np.random.rand(2) + 0.5 # adding 0.2 and normalize to avoid generating smashed ellipses
        ang = np.pi * np.random.rand(1) # in radian
        c, s = np.cos(ang), np.sin(ang)
        rot = np.matrix(np.asarray([[c, s], [-s, c]]))
        pax = np.eye(2) * eig # axes of ellipes
        sgrt = np.dot(rot, pax)
        sg[i] = np.dot(sgrt, sgrt.T)
    return sg

def contour_ellipse(mu, sg):
    '''this function outputs an elliptic contour of a Gaussian based on its mu and sg.'''
    theta = np.linspace(0,2*np.pi, 100)
    eg, egv = np.linalg.eig(sg) # eigen values/vectors of the covariant matrices
    elp_orig = np.vstack((eg[0] * np.cos(theta), eg[1] * np.sin(theta))) 
    elp_rot = np.dot(egv, elp_orig) # rotate the ellipse
    elp = mu[:,np.newaxis] + elp_rot # translate the ellipse
    return elp

def studentT(x, mu, cov, nu):
    
def e_step:
def m_step:
def exp_logp_xht:
def exp_logp_ht:
def exp_logh:
def exp_h:
def eval_nu:

