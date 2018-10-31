import numpy as np                  
import sympy as sy
from scipy.optimize import curve_fit

from symbols import *
from utils import *

class set():
    def __init__(self, sm=None, Q_range=(500,3000), Q_step=500 ):
        self.sm = sm
        if not isinstance(Q_range,tuple):
            self.Q_array = np.array((Q_range,Q_range))
        else:
            self.Q_array = np.arange(Q_range[0],Q_range[1]+Q_step,Q_step)
        w_min_array = 0.2*(np.sqrt(4*4*self.Q_array).round(-1))
        w_max_array = 0.8*(np.sqrt(4*4*self.Q_array).round(-1))
        self.initial_w_range_array = np.stack((w_min_array,w_max_array)).T
        n = self.Q_array.size
        self.n = n
        self.w_vec_list = [None]*n
        self.chi_vec_list = [None]*n
        self.w_mesh_list = [None]*n
        self.chi_mesh_list = [None]*n
        self.xiz_array_list = [None]*n
        self.tanphi_plus_array_list = [None]*n
        self.tanphi_minus_array_list = [None]*n
        self.dwdt_array_list = [None]*n
        self.dchidt_array_list = [None]*n
        self.u_array_list = [None]*n
        self.d_array_list = [None]*n
        self.initial_states_array_list = [None]*n
        self.ode_integrations_lists = [None]*n
        self.t_w_chi_vecs_lists = [None]*n
        self.w_chi_interp_as_t_lists = [None]*n
        self.t_w_chi_resampled_vecs_array_list = [None]*n
        self.t_w_chi_means_array_list = [None]*n
        self.t_w_chi_stdevs_array_list = [None]*n

    def solve_combo_w_chi(self, idx, 
                            n_solutions=30, random_seed=None,
                            t_span=[0.0,1e3],
                            Q_x=3000, 
                            grid_w_range=(0.1,200),
                            grid_chi_range=(0,3),
                            initial_w_range=(20,180),
                            initial_chi_range=(0.01,0.5), 
                            res=20
                ):
        print('Computing grids at resolution {0}x{0}... '.format(res),end='')
        self.sm.set_params(  {Q:Q_x} )
        chi_range = grid_chi_range
        self.sm.specify_u_polynomial_constants({w:w,chi:chi})
        self.sm.specify_d_polynomial_constants({w:w,chi:chi})
        res = res
        w_vec,chi_vec,w_mesh,chi_mesh \
            = self.sm.create_w_chi_vecs_meshes(grid_w_range,chi_range,res=res)
        u_array,d_array = self.sm.nsolve_scipy_u_d_for_w_chi_meshes(w_mesh,chi_mesh)
        epsilon_array = self.sm.epsilon_wrap_for_w_chi_meshes(w_mesh,chi_mesh,d=d_array)
        
        xiz_array       = self.sm.xiz_for_u(u_array)
        xiy_plus_array  = self.sm.xiy_for_u_d_epsilon_w(u_array,d_array,
                                                   epsilon_array,w_mesh,+1)
        xiy_minus_array = self.sm.xiy_for_u_d_epsilon_w(u_array,d_array,
                                                   epsilon_array,w_mesh,-1)
        tanphi_plus_array  = self.sm.tanphi_xiz_xiy(xiz_array, xiy_plus_array)
        tanphi_minus_array = self.sm.tanphi_xiz_xiy(xiz_array, xiy_minus_array)
        dcdt_plus_array  = self.sm.dcdt_for_xiz_xiy(xiz_array, xiy_plus_array)
        dcdt_minus_array = self.sm.dcdt_for_xiz_xiy(xiz_array, xiy_minus_array)
        dwdt_array = self.sm.dwdt_for_dcdt(dcdt_plus_array,dcdt_minus_array)
        dmdt_array = self.sm.dmdt_for_dcdt(dcdt_plus_array,dcdt_minus_array)
        dchidt_array = self.sm.dchidt_for_dmdt(dmdt_array)
        print('done')
        
        print('Computing ODE trajectories... ',end='')
        self.sm.perform_ode_integrations_w_chi( t_span=t_span, interp_t_step=1,
                                                random_seed=random_seed,
                                                n_solutions=n_solutions, 
                                                initial_w_range=initial_w_range,
                                                initial_chi_range=initial_chi_range,
                                                do_densify=False, 
                                                interp_method='cubic')
        self.w_vec_list[idx] = w_vec
        self.chi_vec_list[idx] = chi_vec
        self.w_mesh_list[idx] = w_mesh
        self.chi_mesh_list[idx] = chi_mesh
        self.xiz_array_list[idx] = xiz_array
        self.tanphi_plus_array_list[idx] = tanphi_plus_array
        self.tanphi_minus_array_list[idx] = tanphi_minus_array
        self.dwdt_array_list[idx] = dwdt_array
        self.dchidt_array_list[idx] = dchidt_array
        self.u_array_list[idx] = u_array
        self.d_array_list[idx] = d_array
        self.initial_states_array_list[idx] = self.sm.initial_states_array
        self.ode_integrations_lists[idx] = self.sm.ode_integrations_list
        self.t_w_chi_vecs_lists[idx] = self.sm.t_w_chi_vecs_list
        self.w_chi_interp_as_t_lists[idx] = self.sm.w_chi_interp_as_t_list
        self.t_w_chi_resampled_vecs_array_list[idx] = self.sm.t_w_chi_resampled_vecs_array
        self.t_w_chi_means_array_list[idx] = self.sm.t_w_chi_means_array
        self.t_w_chi_stdevs_array_list[idx] = self.sm.t_w_chi_stdevs_array
        self.w_for_Q_model_fit = None
        self.w_for_chi_model_fit = None
        print('done')
        
    def solve_set_w_chi(self, n_solutions=30,
                        t_span=[0.0,1e3],
                        grid_w_range=(0.1,200),
                        grid_chi_range=(0,3),
                        initial_w_range=None,
                        initial_chi_range=(0.01,0.5) ):
        print(self.Q_array,'\n', self.initial_w_range_array.T)
        for idx,Q_x in enumerate(self.Q_array):
            if initial_w_range is None:
                initial_w_range_x = self.initial_w_range_array[idx]
            print('{0}: Q={1} w_g={2} chi_g={3} w_i={4} chi_i={5} '
                  .format(idx, Q_x, grid_w_range, grid_chi_range,
                          initial_w_range_x, initial_chi_range) )
            self.solve_combo_w_chi(idx, t_span=t_span, 
                                    n_solutions=n_solutions, Q_x=Q_x, 
                                    grid_w_range=grid_w_range, 
                                    grid_chi_range=grid_chi_range,
                                    initial_w_range=initial_w_range_x,
                                    initial_chi_range=initial_chi_range )
        print('Fitting models... ',end='')
        self.fit_w_for_Q_model()
        self.fit_chi_for_Q_model()
        print('done')
            
    def fit_w_for_Q_model(self):
        def wmodel(x,a):
            return a*np.sqrt(x)
        Qs    = self.Q_array/1000
        ws    = np.array([arr[1] for arr in self.t_w_chi_means_array_list])
        werrs = np.array([arr[1] for arr in self.t_w_chi_stdevs_array_list])
        popt,pcov = curve_fit(wmodel,Qs,ws, sigma=werrs)
        perr = np.sqrt(np.diag(pcov))
        self.w_for_Q_model_fit = np.array((popt,perr)).ravel()
        return self.w_for_Q_model_fit

    def fit_chi_for_Q_model(self):
        def chimodel(x,a,b):
            return a*np.power(x,b)
        Qs      = self.Q_array/1000
        chis    = np.array([arr[2] for arr in self.t_w_chi_means_array_list])
        chierrs = np.array([arr[2] for arr in self.t_w_chi_stdevs_array_list])
        popt,pcov = curve_fit(chimodel,Qs,chis, sigma=chierrs)
        perr = np.sqrt(np.diag(pcov))
        self.w_for_chi_model_fit = np.array((popt,perr)).ravel()
        return self.w_for_chi_model_fit

