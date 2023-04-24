#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:43:27 2022

@author: marina
"""



import numpy as np
import copy


class CreateInpaintingProblem:
    
    def __init__(self, signal, inpain_params):
       
            
        self.signal = signal # all signal without any gap
        self.nb_miss_samples = inpain_params['nb_missing_samples']
        self.sig_len = inpain_params['sig_len']
        


    def split_signal(self):
        """
        function that splits the signal

        Returns
        -------
        nb_samples_around_miss_area : int
            number of reliable samples on both sides of the gap to be filled
            
        x_moins : nd-array
           reliable signal of length `sig_len` to the left of the gap
           
        x_t : nd-array
            signal of length L (to be reconstructed) containing the missing samples 
            
        x_plus : nd-array
            reliable signal of length L to the right of the gap
            
        x_true : nd-array
            original signal of length L without missing samples 
            
        sig_miss : nd-array
            the complete signal with its context containing missing data (2L +D)
            
        sig : nd-array
            signal with context ithout missing samples

        """
        
        
        nb_samples_around_miss_area = int(np.ceil((self.sig_len-self.nb_miss_samples)/2))
        sig = self.signal[0:2*self.sig_len+self.nb_miss_samples] # signal with contexte
        sig_miss= copy.deepcopy(sig) 
        
        # the complete signal with its context containing missing data (2L +D)
        sig_miss[self.sig_len:self.sig_len+self.nb_miss_samples] =0 
    
        # reliable signal of length `sig_len` to the left of the gap
        x_moins = sig_miss[0:self.sig_len]
        
        # signal of length L (to be reconstructed) containing the missing samples 
        x_t = sig_miss[self.sig_len- nb_samples_around_miss_area: 2*self.sig_len-nb_samples_around_miss_area]
       
        #reliable signal of length L to the right of the gap
        x_plus = sig_miss[self.sig_len +self.nb_miss_samples :2*self.sig_len+self.nb_miss_samples]
       
        # original signal of length L without missing samples 
        x_true = sig[self.sig_len-nb_samples_around_miss_area:2*self.sig_len-nb_samples_around_miss_area]
    
        return nb_samples_around_miss_area, x_moins ,x_t, x_plus,x_true, sig_miss, sig
    
    
    
    def  generate_mask(self,nb_samples_around_miss_area):
        """
        

        Parameters
        ----------
        nb_samples_around_miss_area : int
            number of reliable samples on both sides of the gap to be filled

        Returns
        -------
        mask : nd-array
           binary mask of length `inpain_params[sig_len]`.

        """
        
        mask = np.ones(self.sig_len, dtype=bool)
        mask[nb_samples_around_miss_area:nb_samples_around_miss_area+self.nb_miss_samples] = 0
    
        return mask
      

def split_fourier_matrix(inpain_params, mask, Phi):
        
    """
        

        Parameters
        ----------
        inpain_params : inpain_params[sig_len] \times `inpain_params[sig_len]` 
        mask : nd-array
            binary mask of length `inpain_params[sig_len]`.
            
        Phi : nd-array
            Fourier matrix

        Returns
        -------
        Phi_u : nd-array
            Fourier matrix corresponds to the unknown part of the signal
            
        Phi_v : nd-array
             Fourier matrix corresponds to the known part of the signal
            

    """
        
        
    miss_index=np.where(mask==0)[0]
    n_shift= inpain_params['sig_len']- miss_index[0]
    Phi_shift = np.roll(Phi, n_shift, axis=1)
        
    Phi_u = Phi_shift[:, 0:inpain_params['nb_missing_samples']]
    Phi_v = Phi_shift[:, inpain_params['nb_missing_samples']:]
        
    return Phi_u, Phi_v
    
     
    
def generate_inpainting_data(signal, inpain_params):
    """
    

    Parameters
    ----------
    signal : TYPE
        DESCRIPTION.
    inpain_params : TYPE
        DESCRIPTION.

    Returns
    -------
    x_moins : TYPE
        DESCRIPTION.
    x_gap : TYPE
        DESCRIPTION.
    x_plus : TYPE
        DESCRIPTION.
    x_true : TYPE
        DESCRIPTION.
    sig_miss : TYPE
        DESCRIPTION.
    sig : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.

    """
    ipt = CreateInpaintingProblem(signal,inpain_params)
    nb_samples_around_miss_area, x_moins ,x_gap, x_plus,x_true, sig_miss, sig =ipt.split_signal()
    mask = ipt.generate_mask(nb_samples_around_miss_area)
    return  x_moins ,x_gap, x_plus,x_true, sig_miss, sig, mask 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
       