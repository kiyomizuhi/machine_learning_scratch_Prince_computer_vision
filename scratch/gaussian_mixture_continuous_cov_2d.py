#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spsp
import scipy.stats as spst

def pf_t(cov, nu, nD):
    pf  = spsp.gamma((nu + nD)/2.)
    pf /= np.power(np.pi * nu, nD/2.)
    pf /= np.power(np.linalg.det(cov), 0.5)
    pf /= spsp.gamma(nu / 2.)
    return pf

def pf_g(cov, nD):
    pf  = 1
    pf /= np.power(2 * np.pi, nD/2.)
    pf /= np.power(np.linalg.det(cov), 0.5)
    return pf

def studentT(x, mu, cov, nu, nD):
    # compute pdf of student's t distribution for the given inputs
    pf = pf_t(cov, nu, nD)
    dlts = x - mu[:, np.newaxis]
    sg = (np.dot(dlts.T, np.linalg.inv(cov)).T* dlts).sum(axis=0)
    mf  = np.power(1. + sg/nu, -(nD + nu)/2.)
    return pf * mf

def random_student_t(mu, cov, nu, N, nD):
    # this function draws samples from multivariate student's t distribution
    # as it does not exit in the library, I generate it as below.
    # using for loop is not ideal. maybe able to be vectorized with einstein summations.
    hg = np.random.gamma(nu/2, scale=2/nu, size=(N,1))
    smp = np.zeros((N, nD))
    for i in range(N):
        smp[i]  = np.random.multivariate_normal(mu, cov/hg[i], 1)
    return smp.T




def gamma_dist(x, a, b):
    return b * spst.gamma(b*x, a)

def e_step(x, mu, cov, nu, nD):
    dlts = x - mu[:, np.newaxis]
    sg = (np.dot(dlts.T, np.linalg.inv(cov)).T* dlts).sum(axis=0)
    return gamma_dist(x, nu/2 + nD/2, sg/2 + nu/2)
    
#def m_step(x, mu, cov, nu, nD):
def exp_logp_xht(x, mu, cov, nu, nD):
    rsl  = nD * exp_logh(x, mu, cov, nu, nD)
    rsl -= nD * np.log(2* np.pi)
    rsl -= np.log(np.linalg.det(cov))
    dlts = x - mu[:, np.newaxis]
    sg = (np.dot(dlts.T, np.linalg.inv(cov)).T* dlts).sum(axis=0)
    rsl -= sg[:, np.newaxis] * exp_h(x, mu, cov, nu, nD)
    rsl /= 2
    return rsl

def exp_logp_xht_br(x, mu, cov, nu, nD):
    rsl = np.zeros((x.shape[1], nu.shape[0]))
    for i in range(x.shape[1]):
        for j in range(nu.shape[0]):
            explogh = exp_logh(x, mu, cov, nu, nD)
            exph = exp_h(x, mu, cov, nu, nD)
            rsl[i, j]  = nD * explogh[i, j]
            rsl[i, j] -= nD * np.log(2 * np.pi)
            rsl[i, j] -= np.log(np.linalg.det(cov))
            dlt = x[:,i] - mu
            sg = np.dot(np.dot(dlt.T, np.linalg.inv(cov)), dlt)
            rsl[i, j] -= sg * exph[i, j]
            rsl[i, j] /= 2
    return rsl
    
def exp_logp_ht(x, mu, cov, nu, nD):
    rsl  = np.tile((nu/2) * np.log(nu/2), (x.shape[1],1))
    rsl -= np.tile(np.log(spsp.gamma(nu/2)), (x.shape[1],1))
    rsl += (nu/2 - 1) * exp_logh(x, mu, cov, nu, nD)
    rsl -= (nu/2) * exp_h(x, mu, cov, nu, nD)
    return rsl

def exp_logp_ht_br(x, mu, cov, nu, nD):
    rsl = np.zeros((x.shape[1], nu.shape[0]))
    for i in range(x.shape[1]):
        for j in range(nu.shape[0]):
            explogh = exp_logh(x, mu, cov, nu, nD)
            exph = exp_h(x, mu, cov, nu, nD)
            rsl[i, j] = (nu[j]/2) * np.log(nu[j]/2)
            rsl[i, j] -= np.log(spsp.gamma(nu[j]/2))
            rsl[i, j] += (nu[j]/2 - 1) * explogh[i, j]
            rsl[i, j] -= (nu[j]/2) * exph[i, j]
    return rsl
    
def exp_logh(x, mu, cov, nu, nD):
    diG = spsp.digamma((nu + nD)/2)
    dlts = x - mu[:, np.newaxis]
    sg = (np.dot(dlts.T, np.linalg.inv(cov)).T* dlts).sum(axis=0)
    diGG, sgg = np.meshgrid(diG, sg)
    nuu, sgg = np.meshgrid(nu, sg)
    return diGG - np.log((nuu + sgg)/2)

def exp_logh_br(x, mu, cov, nu, nD):
    rsl = np.zeros((x.shape[1], nu.shape[0]))
    for i in range(x.shape[1]):
        for j in range(nu.shape[0]):
            diG = spsp.digamma((nu[j] + nD)/2)
            dlt = x[:,i] - mu
            sg = np.dot(np.dot(dlt.T, np.linalg.inv(cov)), dlt)
            rsl[i, j] = diG - np.log((nu[j]+sg)/2)
    return rsl
    
def exp_h(x, mu, cov, nu, nD):
    dlts = x - mu[:, np.newaxis]
    sg = (np.dot(dlts.T, np.linalg.inv(cov)).T* dlts).sum(axis=0)
    nuu, sgg = np.meshgrid(nu, sg)
    return (nuu + nD)/(nuu + sgg)

def exp_h_br(x, mu, cov, nu, nD):
    rsl = np.zeros((x.shape[1], nu.shape[0]))
    for i in range(x.shape[1]):
        for j in range(nu.shape[0]):
            dlt = x[:,i] - mu
            sg = np.dot(np.dot(dlt.T, np.linalg.inv(cov)), dlt)
            rsl[i, j] = (nu[j] + nD)/(nu[j] + sg)
    return rsl
            
            

def eval_nu(x, mu, cov, nu, nD):
    evl = exp_logp_xht(x, mu, cov, nu, nD) + exp_logp_ht(x, mu, cov, nu, nD)
    nus = evl.sum(axis=0)
    nu_min = np.where(nus == nus.min())[0][0]
    return nus, nu_min


#x, y = np.mgrid[-5:5.1:.1, -5:5.1:.1]
N = 100
mu = np.zeros(2)
cov = np.eye(2)
nu = 6
nD = 2
smp = random_student_t(mu, cov, nu, N, nD)

nu = np.arange(1,50)
exph = exp_h(smp, mu, cov, nu, nD)
explogh = exp_logh(smp, mu, cov, nu, nD)
explogpht = exp_logp_ht(smp, mu, cov, nu, nD)
explogpxht = exp_logp_xht(smp, mu, cov, nu, nD)

aa1 = exp_h(smp, mu, cov, nu, nD)
#bb1 = exp_h_br(smp, mu, cov, nu, nD)
#print('1 : ', np.array_equal(aa1, bb1))

aa2 = exp_logh(smp, mu, cov, nu, nD)
#bb2 = exp_logh_br(smp, mu, cov, nu, nD)
#print('2 : ', np.array_equal(aa2, bb2))

aa3 = exp_logp_ht(smp, mu, cov, nu, nD)
#bb3 = exp_logp_ht_br(smp, mu, cov, nu, nD)
#print('3 : ', np.array_equal(aa2, bb2))

aa4 = exp_logp_xht(smp, mu, cov, nu, nD)
#bb4 = exp_logp_xht_br(smp, mu, cov, nu, nD)
#print('4 : ', np.array_equal(aa2, bb2))

nus, nu_min = eval_nu(smp, mu, cov, nu, nD)

plt.figure(1)
plt.clf()
plt.subplot(1,2,1)
cc1 = nu/2 * np.log(nu/2)
cc2 = np.log(spsp.gamma(nu/2))
cc3 = (nu/2 - 1) * aa2.mean(axis=0)
cc4 = (nu/2) * aa1.mean(axis=0)
plt.plot(nu,  cc1, '.-')
plt.plot(nu, -cc2, '.-')
plt.plot(nu,  cc3, '.-')
plt.plot(nu, -cc4, '.-')

plt.subplot(1,2,2)
plt.plot(nu, cc1-cc2+cc3-cc4, '.-')

plt.figure(2)
plt.clf()
plt.subplot(1,2,1)
plt.plot(nu, aa3.mean(axis=0), 'r.-')
plt.plot(nu, aa4.mean(axis=0), 'b.-')
plt.subplot(1,2,2)
plt.plot(nu, aa3.mean(axis=0)+aa4.mean(axis=0), '.-')