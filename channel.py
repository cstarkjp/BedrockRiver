import numpy as np                  
import sympy as sy
from scipy.interpolate import interp2d
import scipy.integrate as si

from symbols import *
import basic
from utils import switch_utils_mixin
import open_channel_flow
from bend import bend_mixin
from sediment import sediment_transport_mixin
from wear import bedrock_wear_mixin

class new_model(bedrock_wear_mixin, sediment_transport_mixin, 
                open_channel_flow.new_numerical_mixin, 
                open_channel_flow.revised_symbolic_mixin,
#                 open_channel_flow.new_symbolic_mixin,
                open_channel_flow.basic_mixin, 
                bend_mixin, switch_utils_mixin, 
                basic.empty_model):
    def __init__(self, *args, **kwargs):
        super(new_model, self).__init__(*args, **kwargs)
        self.initialize_hydraulics()

    def tanphi(self, u,v,theta):
        return ( (u/v)*self.cosec(theta)-self.cot(theta) )
    def cotphi(self, u,v,theta):
        return 1/self.tanphi(u,v,theta)
    def phi_from_u_v_theta(self, u,v,theta):
        return sy.atan(self.tanphi(u,v,theta))
    def phi_deg(self, u,v,theta):
        return np.rad2deg(self.phi(u,v,theta))

    def tanphi_xiz_xiy(self, xiz,xiy):
        return np.float64( (xiy/xiz - sy.cos(self.theta))
                            /  sy.sin(self.theta) )
    
    def create_w_chi_vecs_meshes(self, w_range,chi_range,res=30):
        w_vec = np.linspace(*w_range,res)
        chi_vec = np.linspace(*chi_range,res)
        w_mesh, chi_mesh = np.meshgrid(w_vec,chi_vec)
        return w_vec,chi_vec,w_mesh,chi_mesh
    
    def interpolate_grid_for_w_chi(self, w_mesh,chi_mesh,grid,kind='cubic'):
        return interp2d(w_mesh,chi_mesh,grid, kind=kind)

    def interpolate_grids_for_w_chi(self, w_mesh,chi_mesh,grid1,grid2,
                                  kinds=('linear','linear')):
        return self.interpolate_grid_for_w_chi(w_mesh,chi_mesh,grid1, kind=kinds[0]), \
               self.interpolate_grid_for_w_chi(w_mesh,chi_mesh,grid2, kind=kinds[1])    

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



class old_model(bedrock_wear_mixin, sediment_transport_mixin, 
                open_channel_flow.old_numerical_mixin, open_channel_flow.old_symbolic_mixin,
                open_channel_flow.basic_mixin, 
                bend_mixin, switch_utils_mixin, 
                basic.empty_model):
    def __init__(self, *args, **kwargs):
        super(old_model, self).__init__(*args, **kwargs)
        self.initialize_hydraulics()

    def tanphi(self, u,v,theta):
        return ( (u/v)*self.cosec(theta)-self.cot(theta) )
    def cotphi(self, u,v,theta):
        return 1/self.tanphi(u,v,theta)
    def phi_from_u_v_theta(self, u,v,theta):
        return sy.atan(self.tanphi(u,v,theta))
    def phi_deg(self, u,v,theta):
        return np.rad2deg(self.phi(u,v,theta))

    def tanphi_xiz_xiy(self, xiz,xiy):
        return np.float64( (xiy/xiz - sy.cos(self.theta))
                            /  sy.sin(self.theta) )
    
    def create_w_chi_vecs_meshes(self, w_range,chi_range,res=30):
        w_vec = np.linspace(*w_range,res)
        chi_vec = np.linspace(*chi_range,res)
        w_mesh, chi_mesh = np.meshgrid(w_vec,chi_vec)
        return w_vec,chi_vec,w_mesh,chi_mesh
    
    def interpolate_grid_for_w_chi(self, w_mesh,chi_mesh,grid,kind='cubic'):
        return interp2d(w_mesh,chi_mesh,grid, kind=kind)

    def interpolate_grids_for_w_chi(self, w_mesh,chi_mesh,grid1,grid2,
                                  kinds=('linear','linear')):
        return self.interpolate_grid_for_w_chi(w_mesh,chi_mesh,grid1, kind=kinds[0]), \
               self.interpolate_grid_for_w_chi(w_mesh,chi_mesh,grid2, kind=kinds[1])    

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
