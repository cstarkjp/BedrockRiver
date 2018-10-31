import numpy as np                  
import sympy as sy

from symbols import *
from utils import *

class model():
    def __init__(self, params_dict=None):
        self.default_params_dict = { 
             Q: 1000.0,
             theta: sy.pi/3,
             beta_0: 0.001,
             C: sy.sqrt(1e-3),
             w: 100.0,
             R: 1000.0,
             chi: 0.1,
             L: 1000.0,
             W: 50.0,
             U: 1.0,
             d: 2.76,
             u: 3.125,
             nu: 1,
             u_c: 0.0,
             u_i: 5.0,
             d_i: 50.0, #30.0
             epsilon_r: 0.1,
             mu: 0.75,
             Omega_r: 10,
             eta: 0.5
        }
        self.pi = sy.pi
        if params_dict is not None:
            self.default_params_dict.update(params_dict)
        for item in self.default_params_dict.items():
            setattr(self,str(item[0]),sy.N(item[1]))
        
        self.R_eqn = self.define_R_eqn()
        self.epsilon_eqn = self.define_epsilon_eqn()
        self.d_geometric_eqn = self.solve_d_geometric_eqn()
        self.d_dynamic_eqn = self.raw_d_dynamic_eqn()
        self.u_geometric_eqn = self.raw_u_geometric_eqn()
        self.ucubed_dynamic_eqn = self.raw_ucubed_dynamic_eqn()
        self.u_eqn_rect = self.solve_u_eqn_rect()
        self.d_eqn_rect = self.solve_d_eqn_rect()
        self.du_eqn = sy.Eq(self.raw_ucubed_dynamic_eqn().args[1],
                                (self.raw_u_geometric_eqn().args[1])**3)        
        self.d_polynomial_eqn = self.d_eqn()
        self.u_polynomial_eqn = self.u_eqn()
        
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
                Q:self.Q, theta:self.theta, beta_0:self.beta_0, C:self.C, 
                w:self.w, chi:self.chi,
                d:self.d, u:self.u, nu:self.nu,
                u_c:self.u_c, mu:self.mu, Omega_r:self.Omega_r,
                u_i:self.u_i, d_i:self.d_i,
                epsilon_r:self.epsilon_r, R:self.R,
                eta:self.eta,
                L:self.L, W:self.W, U:self.U
                    }

