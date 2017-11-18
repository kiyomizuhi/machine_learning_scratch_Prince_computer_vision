#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:52:17 2017

@author: hiroyukiinoue
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

nm = 3
ns = 300

mu = 15 * np.random.rand(nm)
sg = 3 * (np.random.rand(nm) + 0.1)
lm = (np.random.rand(nm) + 0.1)
lm /= lm.sum()
lm_ind = np.round(np.cumsum(lm) * ns).astype(int)
lm_ind = np.insert(lm_ind, 0, 0)

smp = np.zeros(ns)
gs_true = np.zeros(ns)
x = np.linspace(-20,20,400)

clrs1 = [(0.6, 0, 0), (0, 0.6, 0), (0, 0, 0.6)]
clrs2 = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]
plt.figure(1, figsize=(5,4))
plt.clf()
for k in range(nm):
    this_smp = np.random.normal(mu[k], sg[k], lm_ind[k+1] - lm_ind[k])
    smp[lm_ind[k]:lm_ind[k + 1]] = this_smp
    gs_true[lm_ind[k]:lm_ind[k + 1]] = sp.stats.norm(mu[k], sg[k]).pdf(this_smp)
    plt.hist(this_smp, 100, normed=1, color=clrs1[k])
    plt.plot(x, sp.stats.norm(mu[k], sg[k]).pdf(x), color=clrs2[k])


L_true = np.log(gs_true).sum() / ns # average log likelihood
print('L_true', L_true)


mue = 10 * np.random.rand(nm)
sge = 3 * (np.random.rand(nm) + 0.1)
lme = (np.random.rand(nm) + 0.1)
lme /= lm.sum()

plt.figure(2, figsize=(10,8))
plt.clf()
clr = 0.6
clrs = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]

Nrep = 8
for rep in range(Nrep):
    gs = np.zeros((nm, ns))
    for k in range(nm):
        gs[k,:] = lme[k] * sp.stats.norm(mue[k], sge[k]).pdf(smp)
    
    r = gs/gs.sum(axis=0)
    ax = plt.subplot(Nrep/2, 6, 1 + 3 * rep + 0)
    plt.hist(smp, 100, normed=1, color=(0.7, 0.7, 0.7))
    for i in range(ns):
        cl = (clr*r[0, i], clr*r[1, i], clr*r[2, i])
        plt.plot(smp[i], -0.05, '.', color=cl, markersize=2)
    for k, c in zip(range(nm), clrs):
        plt.plot(x, lme[k] * sp.stats.norm(mue[k], sge[k]).pdf(x), color=clrs[k])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim((-0.1, 0.3))
        ax.set_title('E-step : %d' % (rep + 1))
        
    lme = r.sum(axis=1)/r.sum()
    sge = np.sqrt((r*((np.tile(smp, (nm,1)) - mue[:,np.newaxis])**2)).sum(axis=1)/r.sum(axis=1))
    mue = (r * smp).sum(axis=1)/r.sum(axis=1)
    ax = plt.subplot(Nrep/2, 6, 1 + 3 * rep + 1)
    plt.hist(smp, 100, normed=1, color=(0.7, 0.7, 0.7))
    for i in range(ns):
        cl = (clr*r[0, i], clr*r[1, i], clr*r[2, i])
        plt.plot(smp[i], -0.05, '.', color=cl, markersize=2)
    for k, c in zip(range(nm), clrs):
        plt.plot(x, lme[k] * sp.stats.norm(mue[k], sge[k]).pdf(x), color=clrs[k])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim((-0.1, 0.3))
        ax.set_title('M-step : %d' % (rep + 1))
    
    ax = plt.subplot(Nrep/2, 6, 1 + 3 * rep + 2)
    plt.hist(smp, 100, normed=1, color=(0.7, 0.7, 0.7))    
    for k in range(nm):
        plt.plot(smp[lm_ind[k]:lm_ind[k + 1]], -0.05*np.ones(lm_ind[k + 1] - lm_ind[k]), '.', color=clrs[k], markersize=2)
        plt.plot(x, lm[k] * sp.stats.norm(mu[k], sg[k]).pdf(x), color=clrs2[k])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim((-0.1, 0.3))
        ax.set_title('truth : %d' % (rep + 1))
    
    # update the average log likelihood
    L_infer = np.log(gs.sum(axis=0)).sum() / ns # average log likelihood
    print('L_infer = %2.6f , L_true = %2.6f' % (L_infer, L_true))