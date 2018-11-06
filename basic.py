import numpy as np                  
import sympy as sy

from symbols import *
from utils import *

class empty_model():
    def __init__(self, params_dict=None):
        self.default_params_dict = { 
             g: 10.0,
             Q: 1000.0,
             phi:   phi,
             beta_0: 0.001,
             C: sy.sqrt(1e-3),
             n_m: 0.03,
             rho: 1000.0,
             rho_s: 2650.0,
             rho_Delta: 1650.0,
             D_fine:    2.0e-3,
             D_coarse: 20.0e-3,
             L: 1000.0,
             W: 50.0,
             U: 1.0,
             theta: theta,
             beta: beta,
             d: d,
             u: u,
             w: w,
             chi: chi,
             R: R,
             tau: tau,
             u_star: u_star,
             tau_star_fine: tau_star_fine,
             tau_star_coarse: tau_star_coarse,
             nu: 1,
             u_c: 0.0,
             u_i: 5.0,
             d_i: 50.0, #30.0
             epsilon_r:epsilon_r,
             mu: 0.75,
             Omega_r: Omega_r,
             eta: 0.5
        }
        self.pi = sy.pi
        if params_dict is not None:
            self.default_params_dict.update(params_dict)
        for item in self.default_params_dict.items():
            setattr(self,str(item[0]),sy.N(item[1]))
        
        self.figs={}
        
    def set_params(self, params_dict):
        for item in params_dict.items():
            setattr(self,str(item[0]),sy.N(item[1]))
    
    def reset_params(self):
        self.set_params(self.default_params_dict)
    
    def get_params(self,vars=None):
        if vars is not None:
            result = {}
            for var in vars.items(): 
                vdi = {var[0]:getattr(self,str(var[0])).round(var[1])} \
                     if var[1] is not None else {var[0]:getattr(self,str(var[0]))}
                result.update(vdi)
            return result
        else:
            return {
                g:self.g,
                Q:self.Q,
                phi:self.phi, 
                beta_0:self.beta_0, 
                C:self.C, 
                n_m:self.n_m, 
                rho:self.rho, 
                rho_s:self.rho, 
                rho_Delta:self.rho_Delta, 
                D_fine:   self.D_fine,
                D_coarse: self.D_coarse,
                nu:self.nu,
                u_c:self.u_c, 
                mu:self.mu, 
                u_i:self.u_i, d_i:self.d_i,
                eta:self.eta,
                L:self.L, W:self.W, U:self.U
                    }

