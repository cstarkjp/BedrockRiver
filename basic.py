import numpy as np
import sympy as sy

# from symbols import *
from utils import *
import symbols as sym

class empty_model():
    def __init__(self, params_dict=None):
        self.reset_params(params_dict=params_dict)
        self.figs={}
        self.pi = sy.pi
        
    def set_params(self, params_dict):
        for item in params_dict.items():
            setattr(self,str(item[0]),sy.N(item[1]))
    
    def reset_params(self, params_dict=None):
        self.default_params_dict = { 
            g: 10.0,
            Q: 1000.0,
            q_bl: q_bl,
            q_sl: q_sl,
            q_c: q_c,
            q_w: q_w,
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
            tau_star_fine: sym.tau_star_fine,
            tau_star_coarse: sym.tau_star_coarse,
            tau_c_star: 0.04,
            nu: 1,
            u_c: 0.0,
            u_i: 5.0,
            d_i: 50.0,
            epsilon_r: epsilon_r,
            mu: 0.75,
            Omega_r: Omega_r,
            eta: 0.5
        }
        if params_dict is not None:
            self.default_params_dict.update(params_dict)
        for item in self.default_params_dict.items():
#             print(  str(item[0]),sy.N(item[1])  )
            setattr(self,str(item[0]),sy.N(item[1]))
#         print(self.default_params_dict)
         
    def get_params(self,vars=None):
        if vars is not None:
            result = {}
            for var in vars.items():
                vdi = {var[0]:getattr(self,str(var[0])).round(var[1])} \
                if var[1] is not None else {str(var[0]):getattr(self,str(var[0]))}
            result.update(vdi)
            return result
        else:
            params_list =  {
                'g',
                'Q',
                'phi', 
                'beta_0', 
                'C', 
                'n_m', 
                'rho', 
                'rho_s', 
                'rho_Delta', 
                'D_fine',
                'D_coarse',
                'u_star',
                'tau_c_star',
                'nu',
                'u_c', 
                'mu', 
                'u_i', 
                'd_i',
                'eta',
                'L', 
                'W', 
                'U'
            }
        rtn_dict = {}
        for param in params_list:
#             print(var[0],type(var[0]))
            try:
                dict_update = {eval(param): getattr(self,str(eval(param)))}
            except:
                print(param, getattr(self,param))
                raise ValueError
            rtn_dict.update(dict_update)
        return rtn_dict
            

