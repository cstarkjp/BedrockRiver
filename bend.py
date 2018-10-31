import numpy as np                  
import sympy as sy
import warnings

from symbols import *
from utils import *

class bend_mixin():
    def define_R_eqn(self):
        return sy.Eq(f_R,2*L*(1+chi)**sy.Rational(3,2)/(13*sy.sqrt(chi)))
    
    def define_epsilon_eqn(self):
        return sy.Eq(f_epsilon, w/(2*self.R_eqn.args[1]))
    
    def recip_R_for_chi(self, chi, L):
#         return (L*(+1))/(4*chi)
#         return (L*(chi+1))/(4*chi)
#         return (L*(chi**1.5+1))/(4*chi)
#         return (L*(chi**2+1))/(4*chi)
        return (6.5*chi**0.5)/(L*(chi+1)**1.5)

    def R_for_chi(self, chi, L):
#         return (L*(+1))/(4*chi)
#         return (L*(chi+1))/(4*chi)
#         return (L*(chi**1.5+1))/(4*chi)
#         return (L*(chi**2+1))/(4*chi)
#         return (L*(chi+1)**1.5)/(6.5*chi**0.5)
#         return (L*(chi+1)**1.25)/(8.5*((chi+1)**0.5-1)**0.5)
        return 1/self.recip_R_for_chi(chi,L)

    def epsilon_for_w_chi_L(self, w,chi,L,d=None):
#         return np.float64( (2*w*chi)/L)
#         return np.float64( (2*w*chi)/(L*(chi+1)) )
#         return np.float64( (2*w*chi)/(L*(chi**1.5+1)) )
#         return np.float64( (2*w*chi)/(L*(chi**2+1)) )
#         return np.float64( (3.25*w*chi**0.5)/(L*(chi+1)**1.5) )
        return np.float64( w*self.recip_R_for_chi(chi,L)*0.5 )
#         return np.float64( 20.0*d*self.recip_R_for_chi(chi,L)*0.5 )
    
    def epsilon_wrap_for_w_chi_meshes(self, w,chi,d=None):
        return self.epsilon_for_w_chi_L(w,chi,self.L,d=d)  
    