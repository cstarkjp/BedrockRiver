import numpy as np                  
from scipy.interpolate import interp1d, interp2d
import sympy as sy
import scipy.integrate as si
from scipy.optimize import newton

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

    def cosec(self, theta):
        return 1.0/sy.sin(theta)
    def cot(self, theta):
        return 1.0/sy.tan(theta)
    def tan(self, theta):
        return sy.tan(theta)
    def cos(self, theta):
        return sy.cos(theta)
    def sin(self, theta):
        return sy.sin(theta)
    def arctan(self, theta):
        return sy.atan(theta)

    def tanphi(self, u,v,theta):
        return ( (u/v)*self.cosec(theta)-self.cot(theta) )
    def cotphi(self, u,v,theta):
        return 1/self.tanphi(u,v,theta)
    def phi(self, u,v,theta):
        return sy.atan(self.tanphi(u,v,theta))
    def phi_deg(self, u,v,theta):
        return np.rad2deg(self.phi(u,v,theta))

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
    
    def create_w_chi_vecs_meshes(self, w_range,chi_range,res=30):
        w_vec = np.linspace(*w_range,res)
        chi_vec = np.linspace(*chi_range,res)
        w_mesh, chi_mesh = np.meshgrid(w_vec,chi_vec)
        return w_vec,chi_vec,w_mesh,chi_mesh
    
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
    
    def interpolate_grid_for_w_chi(self, w_mesh,chi_mesh,grid,kind='cubic'):
        return interp2d(w_mesh,chi_mesh,grid, kind=kind)

    def interpolate_grids_for_w_chi(self, w_mesh,chi_mesh,grid1,grid2,
                                  kinds=('linear','linear')):
        return self.interpolate_grid_for_w_chi(w_mesh,chi_mesh,grid1, kind=kinds[0]), \
               self.interpolate_grid_for_w_chi(w_mesh,chi_mesh,grid2, kind=kinds[1])    

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
    
    def gt_smooth(self, x,x0, k):
        return 0.5*(1+np.tanh((x-np.float64(x0))/k))

    def lt_smooth(self, x,x0, k):
        return self.gt_smooth(-x,-x0,k)

    def switch_smooth(self, x,x0, fn_below, fn_above, k=1.0):
#         return ( self.lt_smooth(x,x0,k)*fn_below + self.gt_smooth(x,x0,k)*fn_above )
#         lt_smooth = self.lt_smooth(x,x0,k)
#         return ( lt_smooth*fn_below + (1-lt_smooth)*fn_above )
        return ( self.lt_smooth(x,x0,k)*(fn_below-fn_above)+fn_above )
    
    def xi_u(self, u):
        return np.float64( 
                            self.mu*(u**self.nu)
                                        *self.switch_smooth(u**2,self.u_c**2, 1e-30,1 )
                         )
    
    def xiz_for_u(self, u_array):
        return self.xi_u(u_array)
    
    def Omega(self,w,d):
        return np.float64( ((w+d/np.tan(np.float64(self.theta)))/d) )
#         wx = d/np.tan(np.float64(self.theta)) # if self.theta<self.pi/2 else 0
#         return np.float64( 
#                             ( w + wx )
#                             /( (w*d+ wx)/(w+2*wx) )    
#                         )
    
    def xiy_for_u_d_epsilon_w(self, u,d,epsilon,w, pm):
#         print(type(u),type(d),type(epsilon),type(w),type(pm))
        return np.float64( 
                            self.xi_u(u)
                                       *(1+(epsilon/self.epsilon_r)*pm)
#                                         *(self.Omega_r/self.Omega(w,d)) 
                                        *(d/10)**1.5
                                       *self.eta
                         ) 

    def tanphi_xiz_xiy(self, xiz,xiy):
        return np.float64( (xiy/xiz - sy.cos(self.theta))
                            /  sy.sin(self.theta) )
    
    def dcdt_for_xiz_xiy(self, xiz,xiy):
        return xiz*self.tanphi_xiz_xiy(xiz,xiy)

    def dwdt_for_dcdt(self, dcdt_plus_array,dcdt_minus_array):
        return dcdt_plus_array+dcdt_minus_array

    def dmdt_for_dcdt(self, dcdt_plus_array,dcdt_minus_array):
        return ((dcdt_plus_array-dcdt_minus_array)/2)#*np.sign(c_plus_array-c_minus_array)

    def dchidt_for_dmdt(self, dmdt_array):
        return np.float64( dmdt_array*(2/self.L) )

#     def dmvdt(self, t,w_chi, dwdt_interp,dchidt_interp):
#         return np.array([dwdt_interp(*wc_chi),dchidt_interp(*wc_chi)]).reshape(2,)
                             
    def dmdt_vec_for_w_chi(self, t, w_chi):
#         if np.any(w_chi!=self.w_chi_init):
#             self.w_chi_init = w_chi
        self.set_ud_lambdas()
        u,d = self.nsolve_u_d_for_w_chi(*w_chi)
        epsilon = self.epsilon_for_w_chi_L(*w_chi,self.L,d=d)
        xiz = self.xiz_for_u(u) 
        dcdt_plus  = self.dcdt_for_xiz_xiy(xiz,
                        self.xiy_for_u_d_epsilon_w(u,d,epsilon,w_chi[0], +1))
        dcdt_minus = self.dcdt_for_xiz_xiy(xiz,
                        self.xiy_for_u_d_epsilon_w(u,d,epsilon,w_chi[0], -1))
        return np.array(
            (self.dwdt_for_dcdt(dcdt_plus,dcdt_minus),
             self.dchidt_for_dmdt(self.dmdt_for_dcdt(dcdt_plus,dcdt_minus)))   )
            
    def ode_integrate_w_chi(self, t_span=[0.0,1e5], initial_state=np.array([50,0.5]),
                              max_step=np.inf, 
                              u_init=None, d_init=None, 
                              do_densify=False,
                              interp_method='quadratic', interp_t_step=100.0,
                              ode_method='LSODA'):
        if u_init is not None:
            self.u_init = u_init
        else:
            self.u_init = np.float64(self.u_i)
        if d_init is not None:
            self.d_init = d_init
        else:
            self.d_init = np.float64(self.d_i)
#         self.w_chi_init = np.array([-1.0,-1.0])
        ode_integration = si.solve_ivp(self.dmdt_vec_for_w_chi, 
                                       t_span, initial_state, max_step=max_step,
                                       method=ode_method,
                                       dense_output=do_densify)
        t_w_chi_vecs = np.vstack((ode_integration['t'], ode_integration['y'])).T
        w_chi_interp_as_t = interp1d(t_w_chi_vecs[:,0], t_w_chi_vecs[:,1:3].T,
                                     kind=interp_method)
        t0, t1 = t_w_chi_vecs[0,0], t_w_chi_vecs[-1,0]
        t_vec = np.arange(t0,t1+interp_t_step,interp_t_step)
        w_chi_vecs = np.vstack((t_vec,w_chi_interp_as_t(t_vec))).T
        return ode_integration, t_w_chi_vecs, w_chi_interp_as_t, w_chi_vecs
    
    def all_for_w_chi_beta0(self, t, w_chi_beta0):
        w_chi = w_chi_beta0[0:2]
        self.beta_0 = w_chi_beta0[2]
        self.set_ud_lambdas()
        u,d = self.nsolve_u_d_for_w_chi(*w_chi)
        epsilon = self.epsilon_for_w_chi_L(*w_chi,self.L,d=d)
        xiz = self.xiz_for_u(u)
        dbeta0dt = (self.U-xiz)/(self.L*self.W*2)
        dcdt_plus  = self.dcdt_for_xiz_xiy(xiz,
                        self.xiy_for_u_d_epsilon_w(u,d,epsilon,w_chi[0], +1))
        dcdt_minus = self.dcdt_for_xiz_xiy(xiz,
                        self.xiy_for_u_d_epsilon_w(u,d,epsilon,w_chi[0], -1))
        dwdt = self.dwdt_for_dcdt(dcdt_plus,dcdt_minus)
        dchidt = self.dchidt_for_dmdt(self.dmdt_for_dcdt(dcdt_plus,dcdt_minus))
#         print('t={0} xiz={1} dbdt={2} b={3} dchdt={4} chi={5}'
#                 .format(  np.float64(t).round(3),
#                           np.float64(xiz).round(3),
#                           np.float64(dbeta0dt).round(8),
#                           np.float64(self.beta_0).round(5),
#                           np.float64(dchidt).round(5),
#                           np.float64(w_chi[1]).round(2) )   )
        return (dwdt, dchidt, dbeta0dt,
                dcdt_plus, dcdt_minus, xiz, epsilon, u,d)

    def dmdt_vec_for_w_chi_beta0(self, t, w_chi_beta0):
        return np.array( self.all_for_w_chi_beta0(t, w_chi_beta0)[0:3]  )

    def ode_integrate_w_chi_beta0(self, t_span=[0.0,1e5], 
                                    initial_state=np.array([50,0.5,0.001]),
                                    max_step=np.inf, 
                                    u_init=None, d_init=None, beta_0_init=None,
                                    do_densify=False,
                                    interp_method='quadratic', interp_t_step=None,
                                    ode_method='LSODA'):
        if interp_t_step is None:
            interp_t_step = t_span[1]/20
        if u_init is not None:
            self.u_init = u_init
        else:
            self.u_init = np.float64(self.u_i)
        if d_init is not None:
            self.d_init = d_init
        else:
            self.d_init = np.float64(self.d_i)
        ode_integration = si.solve_ivp(self.dmdt_vec_for_w_chi_beta0, 
                                       t_span, initial_state, max_step=max_step,
                                       method=ode_method,
                                       dense_output=do_densify)
        t_w_chi_beta0_vecs = np.vstack((ode_integration['t'], ode_integration['y'])).T
        w_chi_beta0_interp_as_t = interp1d(t_w_chi_beta0_vecs[:,0], 
                                           t_w_chi_beta0_vecs[:,1:4].T,
                                           kind=interp_method)
        t0, t1 = t_w_chi_beta0_vecs[0,0], t_w_chi_beta0_vecs[-1,0]
        t_vec = np.arange(t0,t1+interp_t_step,interp_t_step)
        w_chi_beta0_vecs = np.vstack((t_vec,w_chi_beta0_interp_as_t(t_vec))).T
        return ( ode_integration, 
                 t_w_chi_beta0_vecs, w_chi_beta0_interp_as_t, w_chi_beta0_vecs )

    def interp_resample_vecs(self, t_w_chi_beta0_vecs,
                             interp_method='quadratic', interp_t_step=None):
        all_vecs_array = np.zeros((t_w_chi_beta0_vecs.shape[0],9),dtype=np.float64)
        for idx,t_w_chi_beta0 in enumerate(t_w_chi_beta0_vecs):
#             print(idx, t_w_chi_beta0[0], t_w_chi_beta0[1:4])
            all_vecs_array[idx] \
                = self.all_for_w_chi_beta0(t_w_chi_beta0[0], t_w_chi_beta0[1:4])
        all_interp_as_t = interp1d(t_w_chi_beta0_vecs[:,0], 
                                  all_vecs_array.T,
                                  kind=interp_method)
#         for idx,interp_vec in enumerate(interp_vecs_array.T):
#             print(idx,interp_vec)
#         w_chi_beta0_vecs = np.vstack((t_vec,w_chi_beta0_interp_as_t(t_vec))).T
        return all_vecs_array
    
    def perform_ode_integrations_w_chi(self, 
                                     n_solutions=1, random_seed=None,
                                     t_span=[0.0,1e3], max_step=None, interp_t_step=100,
                                     initial_w_range=[10,200],
                                     initial_chi_range=[0.01,0.2],
                                     do_densify=False, interp_method='cubic'):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.initial_states_array = np.random.random_sample((n_solutions,2))
        self.initial_states_array[:,0] *= initial_w_range[1]-initial_w_range[0]
        self.initial_states_array[:,0] += initial_w_range[0]
        self.initial_states_array[:,1] *= initial_chi_range[1]-initial_chi_range[0]
        self.initial_states_array[:,1] += initial_chi_range[0]
        self.ode_integrations_list       = [None]*n_solutions
        self.t_w_chi_vecs_list           = [None]*n_solutions
        self.w_chi_interp_as_t_list      = [None]*n_solutions
        t_w_chi_resampled_vecs_list      = [None]*n_solutions
        for idx, initial_state in enumerate(self.initial_states_array):
#             self.set_params(  {u_i:u_init, d_i:d_init} )
#             self.u_init, self.d_init = u_init,d_init
            try:
                if max_step is not None:
                    (self.ode_integrations_list[idx], self.t_w_chi_vecs_list[idx], 
                     self.w_chi_interp_as_t_list[idx], t_w_chi_resampled_vecs_list[idx]) \
                        = self.ode_integrate_w_chi(t_span=t_span, 
                                                   interp_t_step=interp_t_step,
                                           initial_state=self.initial_states_array[idx],
                                                   max_step=max_step,
                                                   do_densify=do_densify,
                                                   interp_method=interp_method)
                else:
                    (self.ode_integrations_list[idx], self.t_w_chi_vecs_list[idx], 
                     self.w_chi_interp_as_t_list[idx], t_w_chi_resampled_vecs_list[idx]) \
                        = self.ode_integrate_w_chi(t_span=t_span,  
                                                   interp_t_step=interp_t_step,
                                           initial_state=self.initial_states_array[idx],
                                                   do_densify=do_densify,
                                                   interp_method=interp_method)
            except Exception as e:
                print('Failed to solve ODE #{2} with initial values: u_i={0},d_i={1}: {3}'
                       .format(self.u_init,self.d_init, idx, initial_state))
                print(e)
        # Strip out failures (Nones)?
        self.t_w_chi_resampled_vecs_array = np.array(t_w_chi_resampled_vecs_list)
        t_mean    = np.mean(self.t_w_chi_resampled_vecs_array[:,:,0].ravel())
        w_mean    = np.mean(self.t_w_chi_resampled_vecs_array[:,:,1].ravel())
        chi_mean  = np.mean(self.t_w_chi_resampled_vecs_array[:,:,2].ravel())   
        t_stdev   = np.std(self.t_w_chi_resampled_vecs_array[:,:,0].ravel())
        w_stdev   = np.std(self.t_w_chi_resampled_vecs_array[:,:,1].ravel())
        chi_stdev = np.std(self.t_w_chi_resampled_vecs_array[:,:,2].ravel())
        self.t_w_chi_means_array = np.array((t_mean,w_mean,chi_mean))
        self.t_w_chi_stdevs_array = np.array((t_stdev,w_stdev,chi_stdev))
