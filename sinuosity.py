import numpy as np                  
import sympy as sy
from scipy.interpolate import interp1d
import scipy.integrate as si
import warnings

from symbols import *
from utils import *
from channel import channel_model

class bend_model(channel_model):
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
