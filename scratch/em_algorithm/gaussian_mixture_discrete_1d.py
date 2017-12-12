#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt

def generate_synthetic_data(nm, ns):
    """synthesize a hypothetical data set
        
    # Arguements
        nm: number of mixed gaussians.
        ns: number of samples.
            
    # Returns
        mu:     mean.
        sg:     std.
        lm:     abbr of lambda, meaning mixture ratio.
        lm_ind: tail index of each mix.
        smp:    samples.
        L_true: average log likelihood.
    """
    mu = 5 * np.random.rand(nm)
    mu += np.array([-7.5, 0, 7.5])
    sg = 3 * (np.random.rand(nm) + 0.1)
    lm = (np.random.rand(nm) + 0.1)
    lm /= lm.sum()
    lm_ind = np.round(np.cumsum(lm) * ns).astype(int)
    lm_ind = np.insert(lm_ind, 0, 0)          
    
    smp = np.zeros(ns)
    gs_true = np.zeros(ns)
    for k in range(nm):
        this_smp = np.random.normal(mu[k], sg[k], lm_ind[k+1] - lm_ind[k])
        smp[lm_ind[k]:lm_ind[k + 1]] = this_smp
        gs_true[lm_ind[k]:lm_ind[k + 1]] = spst.norm(mu[k], sg[k]).pdf(this_smp)
    
    L_true = np.log(gs_true).sum() / ns
    return mu, sg, lm, lm_ind, smp, L_true


def plot_synthetic_data(smp, mu, sg, lm, lm_ind, nm, ns):
    """plot the hypothetical data set
        
    # Arguements
        smp:    samples.
        mu:     mean.
        sg:     std.
        lm:     abbr of lambda, meaning mixture ratio.
        lm_ind: tail index of each mix.
        nm:     number of mixed gaussians.
        ns:     number of samples.
            
    # Returns
        None. plot the data.
    """
    x = np.linspace(-20, 20, 400)
    clrs1 = [(0.6, 0, 0), (0, 0.6, 0), (0, 0, 0.6)]
    clrs2 = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]
    plt.hist(smp, 100, normed=1, color=(0.7, 0.7, 0.7))
    for k in range(nm):
        this_smp = smp[lm_ind[k]:lm_ind[k + 1]]
        plt.plot(x, lm[k] * spst.norm(mu[k], sg[k]).pdf(x), color=clrs2[k])
        plt.plot(this_smp, -0.05*np.ones(this_smp.shape), '.',\
                 color=clrs1[k], markersize=2)


def generate_initial_state(nm, ns):
    """this will be the initial parameters for the EM algorithm
        
    # Arguements
        nm:     number of mixed gaussians.
        ns:     number of samples.
            
    # Returns
        mue:     estimated mean.
        sge:     estimated std (inferred would be a better term, perhasp).
        lme:     abbr of lambda, meaning estimated mixture ratio estimated.
    """
    mue = 15 * np.random.rand(nm)  # mean
    sge = 3 * (np.random.rand(nm) + 0.1)# std
    lme = (np.random.rand(nm) + 0.1) # mixture ratio (lambda)
    lme /= lme.sum()
    return mue, sge, lme


def e_step(smp, mue, sge, lme, nm, ns):
    """E-step.
        
    # Arguements
        smp:    samples.
        mue:    estimated mean.
        sge:    estimated std.
        lme:    abbr of lambda, meaning estimated mixture ratio estimated.  
        nm:     number of mixed gaussians.
        ns:     number of samples.
            
    # Returns
        r:      responsibility.
        L_infer: inferred avrage log likelihood at this step.
    """
    gs = np.zeros((nm, ns)) # pdf of gaussians
    for k in range(nm):
        gs[k] = lme[k] * spst.norm(mue[k], sge[k]).pdf(smp)
    r = gs/gs.sum(axis=0)
    L_infer = np.log(gs.sum(axis=0)).sum() / ns # average log likelihood
    return r, L_infer


def m_step(smp, r, nm, ns):
    """M-step.
        
    # Arguements
        smp:    samples.
        r:      responsibility.
        nm:     number of mixed gaussians.
        ns:     number of samples.
            
    # Returns
        mue:    estimated mean.
        sge:    estimated std.
        lme:    abbr of lambda, meaning estimated mixture ratio estimated.  
    """
    lme = r.sum(axis=1)/r.sum()
    mue = (r * smp).sum(axis=1)/r.sum(axis=1)
    sge = np.sqrt((r*((np.tile(smp, (nm,1)) - mue[:,np.newaxis])**2)).sum(axis=1)/r.sum(axis=1))
    return mue, sge, lme


def plot_em_steps(smp, r, mue, sge, lme, nm, ns):
    """plot the each step of EM algorithm
        
    # Arguements
        smp:    samples.
        mu:     mean.
        sg:     std.
        lm:     abbr of lambda, meaning mixture ratio.
        lm_ind: tail index of each mix.
        nm:     number of mixed gaussians.
        ns:     number of samples.
            
    # Returns
        None. plot the data.
    """
    x = np.linspace(-20, 20, 400)
    clr = 0.6
    clrs = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]
    plt.hist(smp, 100, normed=1, color=(0.7, 0.7, 0.7))
    for i in range(ns):
        cl = (clr*r[0, i], clr*r[1, i], clr*r[2, i])
        plt.plot(smp[i], -0.05, '.', color=cl, markersize=2)
    for k, c in zip(range(nm), clrs):
        plt.plot(x, lme[k] * spst.norm(mue[k], sge[k]).pdf(x), color=clrs[k])


def main():
    """usually the repetition of EM steps continues till the log likelihood 
       saturates. For the sake of plotting every step, I use a for loop 
       instead of a while loop.
        
    # Arguements
        smp:    samples.
        mu:     mean.
        sg:     std.
        lm:     abbr of lambda, meaning mixture ratio.
        lm_ind: tail index of each mix.
        nm:     number of mixed gaussians.
        ns:     number of samples.
            
    # Returns
        None. plot the data.
    """
    Nrep = 8  # number of repetition of EM steps
    nm = 3    # number of mixed gaussians.
    ns = 300  # number of samples.
    
    mu, sg, lm, lm_ind, smp, L_true = generate_synthetic_data(nm, ns)
    plt.figure(1, figsize=(5,4))
    plt.clf()
    plot_synthetic_data(smp, mu, sg, lm, lm_ind, nm, ns)
    
    mue, sge, lme = generate_initial_state(nm, ns)
    axi = 0 # subplot number
    plt.figure(2, figsize=(12,9))
    plt.clf()
    for rep in range(Nrep):
        # E-step
        r, L_infer = e_step(smp, mue, sge, lme, nm, ns)
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        plot_em_steps(smp, r, mue, sge, lme, nm, ns)
        ax.set_title('E-step : %d' % (rep + 1))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim((-0.1, 0.3))

        # M-step
        mue, sge, lme = m_step(smp, r, nm, ns)
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        plot_em_steps(smp, r, mue, sge, lme, nm, ns)
        ax.set_title('M-step : %d' % (rep + 1))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim((-0.1, 0.3))

        # plot the ground truth for comparison
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        plot_synthetic_data(smp, mu, sg, lm, lm_ind, nm, ns)
        ax.set_title('grn_truth')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim((-0.1, 0.3))

        print('L_infer = %2.6f , L_true = %2.6f' % (L_infer, L_true))

if __name__ == '__main__':
    main()
