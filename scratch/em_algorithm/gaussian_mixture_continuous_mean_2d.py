#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst


def generate_nice_params():
    """this function outputs an covariance matrices
       avoiding generating too much squashed ellipses
        
    # Arguements
        None.
            
    # Returns
        mu:  mean.
        sg:  covariance matrix.
        phi: factor.
    """
    sg = np.diag(2*np.random.rand(2) + 0.2)
    mu = 6*np.random.rand(2) - 5.
    mu = mu[:, np.newaxis]
    ang = np.pi * np.random.rand(1) # in radian
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([c, s, -s, c]).reshape((2, 2))
    phi = np.array([[2*np.random.rand() + 0.2, 0.]])
    # adding 0.2 and normalize to avoid generating smashed ellipses
    phi = np.dot(rot, phi.T)
    return sg, mu, phi


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
    eg, egv = np.linalg.eig(sg)
    # eigen values/vectors of the covariant matrices
    elp_orig = np.vstack((eg[0] * np.cos(theta), eg[1] * np.sin(theta))) 
    elp_rot = np.dot(egv, elp_orig) # rotate the ellipse
    elp = mu + elp_rot # translate the ellipse
    return elp


def draw_samples(sg, mu, phi, h, ns):
    """synthesize a hypothetical data set
        
    # Arguements
        mu:  mean.
        sg:  covariance matrix.
        phi: factor.
        h:   latent variable.
        ns:  number of samples.
            
    # Returns
        smp:    samples.
    """
    smp = np.zeros((2, ns))
    mu_phi = mu + np.dot(phi, h)
    for i in range(ns):
        smp[:, i]  = np.random.multivariate_normal(mu_phi[:, i], sg, 1)
    return smp


def plot_samples(smp, mu, phi, elp=None):
    """plot the hypothetical data set
        
    # Arguements
        mu:  mean.
        sg:  covariance matrix.
        phi: factor.
        elp: contour of ellipse.
            
    # Returns
        None. plot the data.
    """
    plt.plot(smp[0], smp[1], 'b.', markersize=0.5)
    h_edge = np.array([[-100, 100]])
    phi_line = mu + np.dot(phi, h_edge)
    plt.plot(phi_line[0], phi_line[1], ls='-', color=(0., 0.4, 1.))
    if elp is not None:
        plt.plot(elp[0], elp[1], color=(0.8, 0., 0.), ls='-')
    plt.axis('square')
    plt.xlim((-10., 10.))
    plt.ylim((-10., 10.))


def e_step(delta, sge, phie):
    """E-step.
        
    # Arguements
        delta:  smp - mue.
        sge:    estimated std.
        phie:   estimated factor.
            
    # Returns
        exph:   expectation value of latent variable (h) for all the samples.
        exphh:  expectation value of h_i**2 for all the samples.
    """
    B = np.dot(np.dot(phie.T, np.linalg.inv(sge)), delta) # K x ns, here, K = 1
    C = np.eye(1) + np.dot(np.dot(phie.T, np.linalg.inv(sge)), phie) # K x K
    exph = B/C              # K x ns (K = 1)
    exphh = 1/C + exph ** 2 # K x ns
    return exph, exphh

    
def m_step(delta, exph, exphh):
    """M-step.
        
    # Arguements
        delta:  smp - mue.
        exph:   expectation value of latent variable (h) for all the samples.
        exphh:  expectation value of h_i**2 for all the samples.
    
            
    # Returns
        sge:    estimated std.
        phie:   estimated factor.
    """
    phie = np.dot(delta, exph.T) / exphh.sum() # K x 1
    sge  = np.dot(delta, delta.T)              # D x 1, 1 x D 
    sge -= np.dot(np.dot(phie, exph), delta.T) # D x K, K x ns, ns x D
    sge /= delta.shape[1]
    sge  = np.diag(np.diag(sge))
    return phie, sge


def normalize_exph(exph, thrh):
    """put a threshold and normalize exph for plottig.
       this will be used for color attribute.
        
    # Arguements
        exph:   expectation value of latent variable (h) for all the samples.
        thrh:   threshold.

    # Returns
        clr:    color scheme for exph
    """
    clr = np.squeeze(exph) + 0.5 * thrh
    clr[np.where(clr > thrh)] = thrh
    clr[np.where(clr < 0)] = 0
    clr /= thrh
    return clr


def eval_loglikelihood(smp, mu, phi, sg, ns):
    """evaluate average log likelihood of the samples based on the params.
        
    # Arguements
        smp:    samples.
        mu:     mean.
        sge:    estimated std.
        phie:   estimated factor.
            
    # Returns
        aveloglike:  average log likelihood.
    """
    sg_full = sg + np.dot(phi, phi.T)
    gs = spst.multivariate_normal(mean=np.squeeze(mu), cov=sg_full).pdf(smp.T)
    aveloglike = np.log(gs).sum()/ns
    return aveloglike


def generate_synthetic_data(ns):
    """synthesize a hypothetical data set
        
    # Arguements
        ns:  number of samples.
            
    # Returns
        mu:     mean.
        sg:     std.
        lm:     abbr of lambda, meaning mixture ratio.
        lm_ind: tail index of each mix.
        smp:    samples.
        L_true: average log likelihood.
    """
    ns = 300
    sg, mu, phi = generate_nice_params()
    h = np.random.normal(0, 1, ns)[:, np.newaxis].T
    smp = draw_samples(sg, mu, phi, h, ns)
    sg_full = sg + np.dot(phi, phi.T)
    elp = contour_ellipse(mu, sg_full)
    return smp, mu, phi, sg, elp


def plot_decomposition(smp, mu, phi, sg):
    """plot decompositions of the gaussian by selecting representative
       factors.
       subplot(1, 3, 1)
           The full gaussian.
       subplot(1, 3, 2)
           Selecting h = np.arange(-20, 20, 5).
           But exaggerating their contributions just for visualization.
       subplot(1, 3, 3)
           Selecting h = np.arange(-20, 20, 1).
           With correct weighting.
           
    # Arguements
        smp:    samples.
        mu:     mean.
        sge:    estimated std.
        phie:   estimated factor.
            
    # Returns
        None. plot the data.
    """
    x, y = np.mgrid[-10.:10.025:.025, -10.:10.025:.025]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x;
    pos[:, :, 1] = y
    sg_full = sg + np.dot(phi, phi.T)
    gs = spst.multivariate_normal(mean=np.squeeze(mu), cov=sg_full).pdf(pos).T
    gs = np.flipud(gs)

    # plot gaussian
    plt.figure(234, figsize=(12.0, 4.0))
    plt.subplot(1,3,1)
    plt.imshow(gs, extent=[-10, 10, -10, 10])
    plt.title('marginalized covariance')

    # Illustrate the cotributions of many gaussians with 
    # identical diagonal covariance along pax
    # The weight should decay as standard normal. Here, exaggerating it.
    h_rep = np.arange(-20, 20, 5)[np.newaxis, :] # h representatives
    mu_rep = mu + np.dot(phi, h_rep)
    gs_rep = np.zeros(gs.shape)
    for i in range(h_rep.shape[1]):
        gs_add  = spst.multivariate_normal\
                      (mean=np.squeeze(mu_rep[:, i]), cov=sg).pdf(pos).T
        gs_add *= 1.0 - 0.06 * np.abs(h_rep[0,i])
        gs_rep += gs_add
    gs_rep = np.flipud(gs_rep)
    plt.subplot(1,3,2)
    plt.imshow(gs_rep, extent=[-10, 10, -10, 10])
    plt.title('exaggerated decaying contributions')

    # Illustrate adding discrete gaussians weighted by 
    # standard normal with respect to h
    h_rep = np.arange(-20, 20, 1)[np.newaxis, :] 
    mu_rep = mu + np.dot(phi, h_rep)
    gs_rep = np.zeros(gs.shape)
    for i in range(h_rep.shape[1]):
        gs_add  = spst.multivariate_normal\
                      (mean=np.squeeze(mu_rep[:, i]), cov=sg).pdf(pos).T
        gs_add *= spst.norm(0, 1).pdf(h_rep[0,i])
        gs_rep += gs_add
    gs_rep = np.flipud(gs_rep)
    plt.subplot(1,3,3)
    plt.imshow(gs_rep, extent=[-10, 10, -10, 10])
    plt.title('approaching the marginalized cov')


def main():
    """usually the repetition of EM steps continues till the log likelihood 
       saturates. For the sake of plotting every step, I use a for loop 
       instead of a while loop.
        
    # Arguements
            
    # Returns
    """
    Nrep = 8 # number of repetition of EM steps
    ns= 300  # number of samples
    axi = 0 # subplot number

    smp, mu, phi, sg, elp = generate_synthetic_data(ns)
    L_true = eval_loglikelihood(smp, mu, phi, sg, ns)
    plt.figure(1, figsize=(4.0, 4.0))
    plot_samples(smp, mu, phi, elp)
    exph, exphh = e_step(smp - mu, sg, phi)
    h_edge = np.array([[-100, 100]])
    phi_line = mu + np.dot(phi, h_edge)
    clrt = normalize_exph(exph, 4)

    sge, mue, phie = generate_nice_params()
    mue = smp.mean(axis=1)[:, np.newaxis]
    delta = smp - mue # D x ns
    plt.rcParams['figure.figsize'] = (16.0, 10.0) # set default size of plots
    plt.figure()
    for rep in range(Nrep):
        # E-step
        exph, exphh = e_step(delta, sge, phie)
        elpe = contour_ellipse(mue,  sge + np.dot(phie, phie.T))
        phie_line = mue + np.dot(phie, np.array([[-100, 100]]))
        clr = normalize_exph(exph, 4)    
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        for i in range(ns):
            plt.plot(smp[0, i], smp[1, i], '.', color=(1.-clr[i], 0., clr[i]))
            plt.plot(mue[0], mue[1], 'y.', markersize=5)
        plt.plot(elpe[0], elpe[1], color=(0.0, 0.8, 0.), ls='-')
        plt.plot(phie_line[0], phie_line[1], ls='-', color=(0., 0.6, 0.))
        ax.set_title('E-step : %d' % (rep + 1))
        ax.set_xlim((-10., 10.))
        ax.set_ylim((-10., 10.))
        ax.set_xticklabels([])
        ax.set_yticklabels([])    
    
        # M-step
        phie, sge = m_step(delta, exph, exphh)
        elpe = contour_ellipse(mue,  sge + np.dot(phie, phie.T))
        phie_line = mue + np.dot(phie, np.array([[-100, 100]]))
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        for i in range(ns):
            plt.plot(smp[0, i], smp[1, i], '.', color=(1.-clr[i], 0., clr[i]))
        plt.plot(mue[0], mue[1], 'y.', markersize=5)
        plt.plot(elpe[0], elpe[1], color=(0.0, 0.8, 0.), ls='-')
        plt.plot(phie_line[0], phie_line[1], ls='-', color=(0., 0.6, 0.))
        ax.set_title('M-step : %d' % (rep + 1))
        ax.set_xlim((-10., 10.))
        ax.set_ylim((-10., 10.))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
        # plot the ground truth for comparison
        axi += 1 
        ax = plt.subplot(Nrep/2, 6, axi)
        for i in range(ns):
            plt.plot(smp[0, i], smp[1, i], '.', color=(1.-clrt[i], 0., clrt[i]))
        plt.plot(mu[0], mu[1], 'y.', markersize=5)
        plt.plot(phi_line[0], phi_line[1], ls='-', color=(0., 0.6, 0.))
        plt.plot(elp[0], elp[1], color=(0.0, 0.8, 0.), ls='-')
        ax.set_title('grd truth')
        ax.set_xlim((-10., 10.))
        ax.set_ylim((-10., 10.))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # update the average log likelihood
        L_infer = eval_loglikelihood(smp, mue, phie, sge, ns)
        print('L_infer = %2.6f , L_true = %2.6f' % (L_infer, L_true))
    
if __name__ == '__main__':
    main()