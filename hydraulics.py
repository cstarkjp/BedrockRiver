import numpy as np                  
import sympy as sy
from scipy.optimize import newton
import mixin

from symbols import *
from utils import *

class open_channel_flow(trig_mixin):
    def raw_d_dynamic_eqn(self):
        return sy.Eq(d, 
#             ((Q*sy.sin(beta_0))/(C**2*u**3)-w)*(sy.sin(theta)/2) )
            sy.simplify(((Q*sy.sin(beta_0))/(C**2*u**3)-w)*(sy.sin(theta)/2)))

    def raw_d_geometric_eqn(self, d, u,w,Q,theta):
        return sy.Eq((u*sy.cot(theta))*d**2 +u*w*d - Q, 0)
    
    def solve_d_geometric_eqn(self):
        cottheta = sy.symbols('c_theta', real=True, positive=True )
        eq1 = self.raw_d_geometric_eqn(d, u,w,Q,theta).subs(sy.cot(theta),cottheta)
        d_geometric_solns = sy.solveset( eq1, d )
        d_geometric_soln = sy.simplify(d_geometric_solns.args[1].subs(
                                                    cottheta,1/sy.tan(theta)))
        return sy.Eq(d,d_geometric_soln)

    def solve_u_geometric_eqn(self, d_geometric_eqn):
        u_geometric_solns = sy.solveset( d_geometric_eqn, u, domain=sy.S.Reals )
        u_geometric_soln = sy.simplify(u_geometric_solns.args[0].args[0])
        return sy.Eq(u,u_geometric_soln)

    def raw_u_geometric_eqn(self):
        return sy.Eq(u, sy.simplify(Q/(w*d+(d**2)/sy.tan(theta)) ))

    def raw_ucubed_dynamic_eqn(self):
        return sy.Eq(u**3, sy.simplify(Q*(sy.sin(beta_0)/(chi+1)))
                                       /(C**2*(w+2*d/sy.sin(theta)) ))
        
    def d_eqn(self):
        full_d_soln = sy.solveset(self.du_eqn,d)
        return full_d_soln.args[0].args[1]

    def u_eqn(self):
        eqn = self.d_polynomial_eqn.subs(d,self.d_geometric_eqn.args[1])
        full_u_soln = sy.solveset(eqn,u)
        u_soln = sy.simplify(full_u_soln.args[0].args[1]).args[0]/64/Q**2
        u_soln = sy.simplify(sy.expand(u_soln/sy.sqrt(u)))
        return sy.Eq(u_soln)
    
    def solve_u_eqn_rect(self):
        u3_eqn = self.ucubed_dynamic_eqn.subs(theta,sy.pi/2).args[1]
        u2_eqn = sy.simplify(((2*d+w)*(u**2-u3_eqn)).subs(d,Q/(w*u))/u)
        return sy.Eq(u,sy.solve(sy.Eq(sy.numer(u2_eqn)),u)[1])
    
    def solve_d_eqn_rect(self):
        d_eqn_rect = sy.simplify((Q/(u*w)).subs(u,self.u_eqn_rect.args[1]))
        return sy.Eq( d,
            sy.simplify(d_eqn_rect.subs(Q,t**2)).cancel().subs(t,sy.sqrt(Q)) )

    def specify_u_polynomial_constants(self, params_dict_update):
        params_dict = self.get_params()
        params_dict.update({u:u})
        params_dict.update(params_dict_update)
#         for item in params_dict.items():
#             print(item[0],type(item[1]))
        self.u_polynomial_specified = self.u_polynomial_eqn.subs(params_dict)
        return self.u_polynomial_specified

    def specify_d_polynomial_constants(self, params_dict_update):
        params_dict = self.get_params()
        params_dict.update({d:d})
        params_dict.update(params_dict_update)
        self.d_polynomial_specified = self.d_polynomial_eqn.subs(params_dict)
        return self.d_polynomial_specified

    def root_u_polynomial(self, params_dict, u_0=10):
        return np.float64(
            sy.nsolve(self.u_polynomial_specified.subs(params_dict).args[0],u_0))
    def root_d_polynomial(self, params_dict, d_0=50):
        return np.float64(
            sy.nsolve(self.d_polynomial_specified.subs(params_dict).args[0],d_0))

    def root_u_polynomial_for_w_chi(self, w_i,chi_i):
        return self.root_u_polynomial({w:w_i,chi:chi_i})
    def root_d_polynomial_for_w_chi(self, w_i,chi_i):
        return self.root_d_polynomial({w:w_i,chi:chi_i})
    
    def set_ud_lambdas(self):
        from sympy import sqrt
        u_eqn = self.specify_u_polynomial_constants({w:w,chi:chi}).args[0]
        d_eqn = self.specify_d_polynomial_constants({w:w,chi:chi}).args[0]
        self.u_lambda = sy.utilities.lambdify(((u,w,chi),), u_eqn, 'sympy')
        self.d_lambda = sy.utilities.lambdify(((d,w,chi),), d_eqn, "sympy")        
    
    def u_for_w_chi(self, u,w,chi):
        from numpy import sqrt
        u = np.float64(sy.Abs(self.u_lambda([u,w,chi])))
#         if u>20:
#             u /=100000
        return u

    def d_for_w_chi(self, d,w,chi):
        d = self.d_lambda([d,w,chi])
#         print(d)
        return d
    
    def nsolve_u_d_for_w_chi(self, w,chi):
        from sympy import sqrt
#         warnings.simplefilter('ignore')
        self.u_init = np.float64(self.u_i)
        self.d_init = np.float64(self.d_i)
        ux = newton(self.u_for_w_chi,self.u_init, args=[w,chi])
        dx = newton(self.d_for_w_chi,self.d_init, args=[w,chi])
#         if w>100 and w<101 and chi>0.66 and chi<0.67:
#             print(w,chi,' : ',ux,dx)
        return ux,dx  

    def nsolve_u_d_for_w_chi_fast(self, w,chi):
        from sympy import sqrt
#         warnings.simplefilter('ignore')
        ux = newton(self.u_for_w_chi,self.u_init, args=[w,chi])
        dx = newton(self.d_for_w_chi,self.d_init, args=[w,chi])
#         print(w,chi,' : ',ux,dx)
        if ux>2 and ux<3:
            self.u_init = ux
        if dx>3 and dx<8:
            self.d_init = dx
        return ux,dx  

    def nsolve_scipy_u_d_for_w_chi_meshes(self, w_mesh,chi_mesh):
        self.set_ud_lambdas()
        self.u_init = np.float(self.u_i)
        self.d_init = np.float(self.d_i)
        u_array = np.zeros_like(w_mesh,dtype=np.float64)
        d_array = np.zeros_like(w_mesh,dtype=np.float64)
        it = np.nditer([chi_mesh,w_mesh], flags=['multi_index'])
        while not it.finished:
            mi = it.multi_index
            w_i,chi_i = w_mesh[mi],chi_mesh[mi]
            u_array[mi],d_array[mi] = self.nsolve_u_d_for_w_chi(w_i,chi_i)
            it.iternext()
        return u_array,d_array
    
    def nsolve_sympy_u_d_for_w_chi_meshes(self, w_mesh,chi_mesh):
        u_array = np.zeros_like(w_mesh,dtype=np.float64)
        d_array = np.zeros_like(w_mesh,dtype=np.float64)
        it = np.nditer([chi_mesh,w_mesh], flags=['multi_index'])
        while not it.finished:
            mi = it.multi_index
            w_i,chi_i = w_mesh[mi],chi_mesh[mi]
            u_array[mi] = self.root_u_polynomial({w:w_i,chi:chi_i})
            d_array[mi] = self.root_d_polynomial({w:w_i,chi:chi_i})
            it.iternext()
        return u_array,d_array
    
    def u_d_for_w_chi(self, res=50, w_ranges=((0.01,100),(100,1000)), 
                      chi_list=[0,1,2,3]):
        w_vec = np.hstack((np.linspace(*w_ranges[0],res),np.linspace(*w_ranges[1],res)))
        wud_vec_list = [None]*len(chi_list)
        for curve_idx, chi_i in enumerate(chi_list):
            u_array = array([self.root_u_polynomial({w:w_i,chi:chi_i})
                              for w_i in w_vec])
            d_array = array([self.root_d_polynomial({w:w_i,chi:chi_i})
                              for w_i in w_vec])
            wud_array = np.stack((w_vec,u_array,d_array)).T
            wud_array = wud_array[wud_array[:,1]>0]
            wud_vec_list[curve_idx] = wud_array
        return chi_list, wud_vec_list
    
    def Omega(self,w,d):
        return np.float64( ((w+d/np.tan(np.float64(self.theta)))/d) )
#         wx = d/np.tan(np.float64(self.theta)) # if self.theta<self.pi/2 else 0
#         return np.float64( 
#                             ( w + wx )
#                             /( (w*d+ wx)/(w+2*wx) )    
#                         )
    
