#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 03:24:43 2017

@author: hiroyukiinoue
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_nice_covariance_for_ellipse(k):
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

dim = 2    # dimension of the problem. Suppose we work on a 2D problem.
nm = 3     # number of mixed gaussian. Here, we set it to 3.
ns = 300   # number of samples

mu = 10 * np.random.rand(nm, dim)             # mean
sg = generate_nice_covariance_for_ellipse(nm)  # this is to ensure the positive-definiteness of the sigma
lm = np.random.rand(nm) + 0.1           # lambda : ratio of the mixture. 0.1 is to make sure it won't generate too tiny weight
lm /= lm.sum()
lm_ind = np.round(np.cumsum(lm) * ns).astype(int)
lm_ind = np.insert(lm_ind, 0, 0)

smp = np.zeros((dim, ns))
gs_true = np.zeros(ns)
clrs1 = [(0.6, 0, 0), (0, 0.6, 0), (0, 0, 0.6)]
clrs2 = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.figure(1)
plt.clf()
for k in range(nm):
    this_smp = np.random.multivariate_normal(mu[k], sg[k], lm_ind[k+1] - lm_ind[k]).T
    smp[:, lm_ind[k]:lm_ind[k + 1]] = this_smp
    gs_true[lm_ind[k]:lm_ind[k + 1]] = multivariate_normal(mu[k], sg[k]).pdf(this_smp.T)
    plt.plot(this_smp[0], this_smp[1], '.', markersize=3, color=clrs1[k])
    elp = contour_ellipse(mu[k], sg[k])
    plt.plot(elp[0], elp[1], color=clrs2[k], ls='-')

L_true = np.log(gs_true).sum() / ns # average log likelihood
print('L_true', L_true)

mue = 10 * np.random.rand(nm, dim)             # initial mean (to be Estimated)
sge = generate_nice_covariance_for_ellipse(nm) # initial std  (to be Estimated)
lme = np.random.rand(nm)
lme /= lme.sum()                      # initial mixture ratio (to be Estimated)

Nrep = 16
plt.rcParams['figure.figsize'] = (16.0, 20.0) # set default size of plots
clr = 0.6
clrs = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]

plt.figure(2)
plt.clf()
# usually the repetition is terminated once the log likelihood saturates.
# For the sake of plotting every step, I use a for loop instead of a while loop.
for rep in range(Nrep):
    
    # E-step
    gs = np.zeros((nm, ns))
    for k in range(nm):
        gs[k, :] = lme[k] * multivariate_normal(mue[k], sge[k]).pdf(smp.T)
    r = gs / gs.sum(axis=0) # sum over classes
    # plot the update
    ax = plt.subplot(Nrep/2, 6, 1 + 3 * rep)
    for i in range(ns):
        cl = (clr*r[0, i], clr*r[1, i], clr*r[2, i])
        plt.plot(smp[0, i], smp[1, i], '.', color=cl, markersize=2)
    for i, c in zip(range(nm), clrs):
        elp = contour_ellipse(mue[i], sge[i])
        plt.plot(elp[0], elp[1], color=c, ls='-')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title('E-step : %d' % (rep + 1))
        
    # M-step
    lme = r.sum(axis=1)/r.sum()
    for k in range(nm):
        dlts = smp - mue[k][:, np.newaxis]
        sge[i] = np.dot((r[k] * dlts), dlts.T) / r[k].sum()
        mue[i] = (r[k] * smp).sum(axis=1) / r[k].sum()
    # plot the update
    ax = plt.subplot(Nrep/2, 6, 1 + 3 * rep + 1)
    for i in range(ns):
        cl = (clr*r[0, i], clr*r[1, i], clr*r[2, i])
        plt.plot(smp[0, i], smp[1, i], '.', color=cl, markersize=2)
    for i, c in zip(range(nm), clrs):
        elp = contour_ellipse(mue[i], sge[i])
        plt.plot(elp[0], elp[1], color=c, ls='-')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title('M-step : %d' % (rep + 1))
        
    # plot the ground truth for a comparison
    ax = plt.subplot(Nrep/2, 6, 1 + 3 * rep + 2)
    for i, c in zip(range(nm), clrs):
        this_smp = smp[:, lm_ind[i]:lm_ind[i+1]]
        plt.plot(this_smp[0], this_smp[1], '.', markersize=2, color=clrs1[i])
        elp = contour_ellipse(mu[i], sg[i])
        plt.plot(elp[0], elp[1], color=clrs2[i], ls='-')
        ax.set_title('ground truth')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # update the average log likelihood
    L_infer = np.log(gs.sum(axis=0)).sum() / ns # average log likelihood
    print('L_infer = %2.6f , L_true = %2.6f' % (L_infer, L_true))