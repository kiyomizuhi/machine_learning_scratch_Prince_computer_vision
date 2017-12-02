#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_nice_covariance_for_ellipse(k):
    """this function outputs an covariance matrices
       avoiding generating too much squashed ellipses
        
    # Arguements
        k: number of covariance matrices. shape = (k, 2, 2).
            
    # Returns
        sg:     covariance matrix.
    """
    sg = np.zeros((k, 2, 2))
    for i in range(k):
        # adding 0.2 and normalize to avoid generating smashed ellipses
        eig = 2 * np.random.rand(2)
        eig += 0.5
        ang = np.pi * np.random.rand(1) # in radian
        c, s = np.cos(ang), np.sin(ang)
        rot = np.matrix(np.asarray([[c, s], [-s, c]]))
        pax = np.eye(2) * eig # axes of ellipes
        sgrt = np.dot(rot, pax)
        sg[i] = np.dot(sgrt, sgrt.T)
    return sg


def contour_ellipse(mu, sg):
    """this function outputs an elliptic contour 
       of a Gaussian based on its mu and sg.
        
    # Arguements
        mu:     mean.
        sg:     covariance matrix.

    # Returns
        elp:    contour of ellipse.
    """
    theta = np.linspace(0,2*np.pi, 100)
    eg, egv = np.linalg.eig(sg) # eigen values/vectors of the covariant matrices
    elp_orig = np.vstack((eg[0] * np.cos(theta), eg[1] * np.sin(theta))) 
    elp_rot = np.dot(egv, elp_orig) # rotate the ellipse
    elp = mu[:,np.newaxis] + elp_rot # translate the ellipse
    return elp


def generate_synthetic_data(nm, ns, dim):
    """synthesize a hypothetical data set
        
    # Arguements
        nm:  number of mixed gaussians.
        ns:  number of samples.
        dim: dimension of the problem.
            
    # Returns
        mu:     mean.
        sg:     std.
        lm:     abbr of lambda, meaning mixture ratio.
        lm_ind: tail index of each mix.
        smp:    samples.
        L_true: average log likelihood.
    """
    mu = 3.0 * np.random.rand(nm, dim) +\
         7.5*np.array([[1., 0.],[0., 1.],[1., 1.]])
    # this is to ensure the positive-definiteness of the sigma
    sg = generate_nice_covariance_for_ellipse(nm)
    # lambda : ratio of the mixture.
    # 0.1 is to make sure it won't generate too tiny weight
    lm = np.random.rand(nm) + 0.1
    lm /= lm.sum()
    lm_ind = np.round(np.cumsum(lm) * ns).astype(int)
    lm_ind = np.insert(lm_ind, 0, 0)

    smp = np.zeros((dim, ns))
    gs_true = np.zeros(ns)
    for k in range(nm):
        this_smp = np.random.multivariate_normal\
                            (mu[k], sg[k], lm_ind[k+1] - lm_ind[k]).T
        smp[:, lm_ind[k]:lm_ind[k + 1]] = this_smp
        gs_true[lm_ind[k]:lm_ind[k + 1]] = multivariate_normal\
                                          (mu[k], sg[k]).pdf(this_smp.T)
        
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
    clrs1 = [(0.6, 0, 0), (0, 0.6, 0), (0, 0, 0.6)]
    clrs2 = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]
    for k in range(nm):
       this_smp = smp[:, lm_ind[k]:lm_ind[k + 1]]
       plt.plot(this_smp[0], this_smp[1], '.', markersize=3, color=clrs1[k])
       elp = contour_ellipse(mu[k], sg[k])
       plt.plot(elp[0], elp[1], color=clrs2[k], ls='-')


def generate_initial_state(nm, ns, dim):
    """this will be the initial parameters for the EM algorithm
        
    # Arguements
        nm:  number of mixed gaussians.
        ns:  number of samples.
        dim: dimension of the problem.
            
    # Returns
        mue:     estimated mean.
        sge:     estimated std (inferred would be a better term, perhasp).
        lme:     abbr of lambda, meaning estimated mixture ratio estimated.
    """
    mue = 10 * np.random.rand(nm, dim)
    sge = generate_nice_covariance_for_ellipse(nm)
    lme = np.random.rand(nm)
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
    gs = np.zeros((nm, ns))
    for k in range(nm):
        gs[k, :] = lme[k] * multivariate_normal(mue[k], sge[k]).pdf(smp.T)
    r = gs / gs.sum(axis=0) # sum over classes
    L_infer = np.log(gs.sum(axis=0)).sum() / ns # average log likelihood
    return r, L_infer


def m_step(smp, mue, sge, lme, r, nm, ns):
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
    for k in range(nm):
        mue[k] = (r[k] * smp).sum(axis=1) / r[k].sum()
        dlts = smp - mue[k][:, np.newaxis]
        sge[k] = np.dot((r[k] * dlts), dlts.T) / r[k].sum()
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
    clr = 0.6
    clrs = [(1, 0.2, 0), (0, 1, 0), (0, 0.6, 1)]
    for i in range(ns):
        cl = (clr*r[0, i], clr*r[1, i], clr*r[2, i])
        plt.plot(smp[0, i], smp[1, i], '.', color=cl, markersize=2)
    for i, c in zip(range(nm), clrs):
        elp = contour_ellipse(mue[i], sge[i])
        plt.plot(elp[0], elp[1], color=c, ls='-')


def main():
    """usually the repetition of EM steps continues till the log likelihood 
       saturates. For the sake of plotting every step, I use a for loop 
       instead of a while loop.
        
    # Arguements
            
    # Returns
    """
    dim = 2  # dimension of the problem. Suppose we work on a 2D problem.
    Nrep = 16 # number of repetition of EM steps
    nm = 3    # number of mixed gaussian. Here, we set it to 3.
    ns= 300  # number of samples
    
    mu, sg, lm, lm_ind, smp, L_true = generate_synthetic_data(nm, ns, dim)
    plt.figure(1, figsize=(5,4))
    plt.clf()
    plot_synthetic_data(smp, mu, sg, lm, lm_ind, nm, ns)
    
    mue, sge, lme = generate_initial_state(nm, ns, dim)
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

        # M-step
        mue, sge, lme = m_step(smp, mue, sge, lme, r, nm, ns)
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        plot_em_steps(smp, r, mue, sge, lme, nm, ns)
        ax.set_title('M-step : %d' % (rep + 1))
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # plot the ground truth for comparison
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        plot_synthetic_data(smp, mu, sg, lm, lm_ind, nm, ns)
        ax.set_title('grn_truth')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        print('L_infer = %2.6f , L_true = %2.6f' % (L_infer, L_true))

    
if __name__ == '__main__':
    main()
