import numpy as np                  
from scipy.interpolate import interp1d, interp2d
import sympy as sy
import scipy.integrate as si
from sympy.utilities.lambdify import lambdastr
from sympy.parsing.sympy_parser import parse_expr
from scipy.optimize import newton, curve_fit
import warnings

from symbols import *
from utils import *

class bedrock_wear_mixin():
    def xi_u(self, u):
        return np.float64( 
                            self.mu*(u**self.nu)
                                        *self.switch_smooth(u**2,self.u_c**2, 1e-30,1 )
                         )
    
    def xiz_for_u(self, u_array):
        return self.xi_u(u_array)
    
    def xiy_for_u_d_epsilon_w(self, u,d,epsilon,w, pm):
#         print(type(u),type(d),type(epsilon),type(w),type(pm))
        # Shouldn't this be xi_u(u*(1+epsilon)) ?
        return np.float64( 
                            self.xi_u(u)
                                       *(1+(epsilon/self.epsilon_r)*pm)
#                                         *(self.Omega_r/self.Omega(w,d)) 
                                       *(d/10)**1.5
                                       *self.eta
                         ) 

