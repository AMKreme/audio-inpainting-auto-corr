#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:45:43 2023

@author: marina
"""

import numpy as np
from scipy.fft import fft, ifft
from useful_functions import compute_pseudo_inverse

def least_square_autocorr(toep_visible, hank_visible, meas):
    """
    

    Parameters
    ----------
    toep_visible : nd-arrray
        Toeplitz matrix
    
    hank_visible : nd-array
        Hankel matrix
        
    meas :  nd-array
        
        measurements
    Returns
    -------
    x_lsqr : nd-array
        
        x_lsqr = (toep_visible + hank_visible)^{\dagger} meas
    """
    
    A = toep_visible + hank_visible
    pinvA = compute_pseudo_inverse(A)
    x_lsqr = np.dot(pinvA,meas)
    #x_lsqr = np.dot(np.linalg.pinv(A), meas)
           
    return x_lsqr  

def wiener_like_regularisation2(toep_visible, hank_visible,meas, b_est, 
                               sig_visible, Phi_u, Phi_v, lamda, eps=1e-8):
    
    #identique a WL.
                                    
    # TH
    A = toep_visible+ hank_visible
    AtA = np.dot(A.T, A)

    b_est[b_est==0] = eps
    
    Atb = np.dot(A.T, meas)
    # take the inverse of b     
    b_inv = 1/(b_est)
    b_inv_square = b_inv**2
    diag_b = np.diag(b_inv_square)
    
    #phivv 
    Phiv_v = np.dot(Phi_v, sig_visible)
    
    Phi_uH= np.conj(Phi_u).T
    
    B = np.dot(Phi_uH, np.dot(diag_b,Phi_u))
    C = np.dot(Phi_uH, np.dot(diag_b,Phiv_v))
    
    F1 = AtA +lamda*B
    F1_inv = np.linalg.inv(F1)
    
    F2 = Atb - lamda*C
    
    x_wiener_reg = np.dot(F1_inv,F2)
    
    
    return   x_wiener_reg
    
                                
    

def wiener_like_regularisation(toep_visible, hank_visible,meas, b_est, 
                               sig_visible, Phi_u, Phi_v, lamda, eps=1e-8):
    """
    

    Parameters
    ----------
    toep_visible : nd-array
       Toeplitz matrix
       
    hank_visible : nd-array
        hankel matrix
        
    meas : nd-array
        measurements
        
    b_est : nd-array
        spectrum estimation
        
    sig_visible : nd-array
        known (observed) signal
        
    Phi_u : nd-array
        Fourier matrix
        
    Phi_v : nd-array
        Fourier matrix
        
    lamda : float
        

    Returns
    -------
    x_wiener_reg : nd-array
        Estimation of the signal using a wiener type regularization

    """
    
    # TH
    A = toep_visible+ hank_visible
    b_est[b_est==0] = eps
        
    # take the inverse of b     
    b_inv = 1/(b_est)
    diag_b = np.diag(b_inv)
    R = np.sqrt(lamda)*np.dot(diag_b, Phi_u)
        
    M = np.concatenate([A,R]) 
        
    # phiv times known signal
        
    Phiv_v = np.dot(Phi_v, sig_visible)
    c =  - np.sqrt(lamda)*np.dot(diag_b,Phiv_v)
    obs = np.concatenate([meas,c])
    pinvM = compute_pseudo_inverse(M) 
    
    x_wiener_reg = np.dot(pinvM,obs)
    
    loss = np.linalg.norm(np.dot(M,x_wiener_reg) - obs)
        
    return x_wiener_reg, loss

def wiener(b_est, sig_visible, Phi_u, Phi_v,eps=1e-8):
    
    
    b_est[b_est==0] = eps
    phiv_v = np.dot(Phi_v, sig_visible) #phiv v 
    
    b_square= b_est**2
    diag_phivv = (1/b_square)*phiv_v
    Phi_uH= np.conj(Phi_u).T

    
    t = np.dot(Phi_uH,diag_phivv)
    N =np.dot(Phi_uH,np.dot(np.diag(1/b_square),Phi_u))
    N_inv = np.linalg.inv(N)
   
    x_wiener = - np.dot(N_inv,t)
    
    return x_wiener


def admm_lasso_parameters(rho,epsilon,max_iter):
    return dict(rho =rho, epsilon=epsilon, max_iter=max_iter)
    

def admm_lasso(toep_visible, hank_visible, b,lamda,sig_visible, admm_params):
    """
    solves the following problem via ADMM:
        *minimize \|Au-b\|_2^2 + \lambda\|y\|_1 s.t y = \Phi_u u + \Phi_v v
        where u is the missing part of the signal and v is the known part. 
        \Phi_u and \Pho_v are resecptiveley the corresponding Fourier matrices

    Parameters
    ----------
    toep_visible : TYPE
        DESCRIPTION.
    hank_visible : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    lamda : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    sig_visible : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.
    max_iter : TYPE
        DESCRIPTION.

    Returns
    -------
    u : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    err : TYPE
        DESCRIPTION.

    """
    

    A = toep_visible+hank_visible
    # Data preprocessing
    M,D = A.shape
    
    #save a matrix-vector multiply
    Atb = np.dot(A.T,b)
    Atb = 2*Atb
    AtA = np.dot(A.T,A)
    
    
    I = np.eye(D)
    
    A1 = 2*AtA + admm_params['rho']*I

    A1_inv = np.linalg.inv(A1)
    
    N = len(sig_visible) + D
    
    # ADMM initialisation 
    u  = np.zeros(D)       
    y = np.zeros(N, dtype = complex)
    z = np.zeros(N, dtype = complex)
    
    x = np.concatenate([u, sig_visible])
    # hides the Cholesky factorization (to accelerate matrix factorisation)
    
    #L,U = factor(A,rho)
    
    # error initialisation
    fft_x = fft(x)
    error = np.linalg.norm(fft_x - y)
    err = []
    
    thresh = lamda/admm_params['rho']
    
    loss1 = []
    loss2 = []
    loss = []
    
    it = 0
    while (error > admm_params['epsilon'] and it < admm_params['max_iter']) : 
        
        
     
        err.append(error)
        
        # u update
    
        q = Atb + ifft((-z+admm_params['rho']*y))[0:D]
        u = np.dot(A1_inv, q)
      
        # y update
    
        x[0:D]  = np.real(u)
        x[D:] = sig_visible
        fft_x = fft(x)
        
             
        u1 = fft_x + z/admm_params['rho']
        y = soft_threshold(u1, thresh)
        
       
        
        #compute each term of loss fucntion
        
        Au = np.dot(A, np.real(u))
        loss1.append(np.linalg.norm(Au -b))
        loss2.append(np.linalg.norm(y,1))
        loss.append(np.linalg.norm(Au-b)+lamda*np.linalg.norm(y,1))
        # z update
        
        z=z + admm_params['rho']*(fft_x - y)
        
      
        #udpate error
        
        error = np.linalg.norm(fft_x - y)
        it= it+1
            
   
    return u,x, y, z, err, loss1, loss2 , loss


def tikhonov_like_regularisation(toep_visible, hank_visible, b, lamda):
    A = toep_visible+ hank_visible
    M,D = A.shape
    AtA = np.dot(A.T,A)
    Atb = np.dot(A.T, b)
    A1 = AtA + lamda*np.eye(D)
    A1_inv=  np.linalg.inv(A1) 
    
    x_tikhonov = np.dot(A1_inv, Atb)
    
    
    Au = np.dot(A,x_tikhonov)
    loss = np.linalg.norm(Au-b) +  lamda*np.linalg.norm(x_tikhonov) 
    return x_tikhonov, loss   

                         
def soft_threshold(u, thresh):
    #https://stats.stackexchange.com/questions/357339/soft-thresholding
    #-for-the-lasso-with-complex-valued-data
    if np.iscomplexobj(u):
        u1 = np.maximum(np.abs(u) - thresh,0)*np.exp(1j*np.angle(u))
    else:
        u1 = np.maximum(np.abs(u) - thresh,0)*np.sign(u)
        
    return u1


