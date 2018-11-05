import numpy as np                  
import sympy as sy
from scipy.optimize import newton

from symbols import *
from utils import *

class basic_mixin():
    def initialize_hydraulics(self, friction_model='chezy'):
        print('Initializing open channel flow hydraulics...', end='')
        self.r_eqn = self.define_r_eqn()
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
        
        self.friction_model = friction_model
        print('done')

    def set_friction_model(self, friction_model):
        if friction_model=='chezy':
            self.friction_model='chezy'
        elif friction_model=='manning':
            self.friction_model='manning'
        elif friction_model=='manning_depth':
            self.friction_model='manning_depth'
        else:
            raise NameError('Unknown friction model "{}"'.format(friction_model))
            
class revised_symbolic_mixin(trig_utils_mixin):
    def A_eqn_geom(self):
        return sy.Eq(A,d*(w+d/sy.tan(theta)))

    def A_eqn_dyn(self):
        return sy.Eq(A,Q/u)

    def p_eqn(self):
        return sy.Eq(p, 2*d/sy.sin(theta)+w)

    def R_eqn(self):
        return sy.Eq(R, self.A_eqn_dyn().rhs/self.p_eqn().rhs).simplify()

    def tau_eqn_raw(self):
        return sy.Eq(tau,(rho*g*A*sy.sin(beta)/p).subs(A,self.A_eqn_dyn().rhs))

    def tau_eqn_geom(self):
        return self.tau_eqn_raw().subs(p,self.p_eqn().rhs).simplify()

    def tau_eqn_friction(self):
        return sy.Eq(tau,f*rho*u**2)

    def friction_eqn_chezy(self):
        return sy.Eq(f,g*C**2)

#     def tau_eqn_manning_friction(self):
#         return sy.Eq(tau,rho*g*n_m*u**2/R**sy.Rational(1,3))

    def friction_eqn_manning_raw(self):
        return sy.Eq(f,g*n_m**2/R**sy.Rational(1,3))

    def friction_eqn_manning(self):
        return self.friction_eqn_manning_raw().subs(R,self.R_eqn().rhs) \
                .subs(sy.sin(theta)/(2*d+w*sy.sin(theta)),1/t**3) \
                .subs(t,(2*d+w*sy.sin(theta))**sy.Rational(1,3)
                      /(sy.sin(theta)**sy.Rational(1,3)))

    def friction_eqn_manning_depth(self):
        return self.friction_eqn_manning_raw().subs(R,d) \
                .subs(sy.sin(theta)/(2*d+w*sy.sin(theta)),1/t**3) \
                .subs(t,(2*d+w*sy.sin(theta))**sy.Rational(1,3)
                      /(sy.sin(theta)**sy.Rational(1,3)))

    def friction_eqn(self):
        if self.friction_model=='chezy':
            return self.friction_eqn_chezy()
        elif self.friction_model=='manning':
            return self.friction_eqn_manning()
        elif self.friction_model=='manning_depth':
            return self.friction_eqn_manning_depth()
        else:
            raise NameError('Unknown friction model "{}"'.format(friction_model))

    def tau_eqn_dyn(self):
        return self.tau_eqn_friction().subs(f,self.friction_eqn().rhs).simplify()

    def u_star_eqn_raw(self):
        return sy.Eq(u_star, sy.sqrt(tau/rho))

    def u_star_eqn(self):
        return self.u_star_eqn_raw().subs(tau,self.tau_eqn_dyn().rhs)

    def u_eqn_geom(self):
        return sy.Eq(u, sy.solve(sy.Eq(self.A_eqn_geom().rhs,self.A_eqn_dyn().rhs),u)[0])

    def u_eqn_dyn_chezy(self):
        self.u_dyn_expt = 3
        u_solns = (sy.solve(sy.Eq(self.tau_eqn_geom().rhs,
                                        self.tau_eqn_dyn().rhs),u**self.u_dyn_expt))
        return sy.Eq(u**self.u_dyn_expt,u_solns[0]).simplify()

    def u_eqn_dyn_manning(self):
        self.u_dyn_expt = 5
        u_eqn = sy.Eq(self.tau_eqn_geom().rhs*u,self.tau_eqn_dyn().rhs*u)
        return sy.Eq(u**self.u_dyn_expt,
              (sy.solve(u_eqn.subs(u**sy.Rational(self.u_dyn_expt*2,3),t),t)[0]
               .subs(t,u**sy.Rational(self.u_dyn_expt*2,3))**3)**sy.Rational(1,2))

    def u_eqn_dyn_manning_depth(self):
        self.u_dyn_expt = 3
        u_solns = (sy.solve(sy.Eq(self.tau_eqn_geom().rhs,
                                        self.tau_eqn_dyn().rhs),u**self.u_dyn_expt))
        return sy.Eq(u**self.u_dyn_expt,u_solns[0]).simplify()

    def u_eqn_dyn(self):
        if self.friction_model=='chezy':
            return self.u_eqn_dyn_chezy()
        elif self.friction_model=='manning':
            return self.u_eqn_dyn_manning()
        elif self.friction_model=='manning_depth':
            return self.u_eqn_dyn_manning_depth()
        else:
            raise NameError('Unknown friction model "{}"'.format(friction_model))

    def d_eqn_dyn(self):
        if self.friction_model=='chezy':
            return sy.Eq(d,(sy.solve(self.u_eqn_dyn(),d)[0]).collect(sy.sin(theta)))
        elif self.friction_model=='manning':
            return sy.Eq(d,(sy.solve(self.u_eqn_dyn(),d)[1]).collect(sy.sin(theta)))
        elif self.friction_model=='manning_depth':
            d_eqn = sy.Eq( (self.u_eqn_dyn().lhs*(2*d+w*sy.sin(theta))*n_m**2)**3,
                           (self.u_eqn_dyn().rhs*(2*d+w*sy.sin(theta))*n_m**2)**3 )
            solns = sy.solve(d_eqn,d)
            return sy.Eq(d,solns[1])
        else:
            raise NameError('Unknown friction model "{}"'.format(friction_model))

    def d_eqn_geom(self):
        d_soln = sy.Eq(d,sy.simplify(sy.solve(self.u_eqn_geom(),d)[1]))
        return sy.Eq(d,d_soln.rhs.subs(sy.tan(theta),t).simplify().collect(t) \
                     .subs(t,sy.tan(theta)))

    def d_eqn_poly(self):
        if self.friction_model=='chezy':
            d_eqn = sy.Eq(1/self.u_eqn_geom().rhs**self.u_dyn_expt,
                          1/(self.u_eqn_dyn_chezy().rhs))
        elif self.friction_model=='manning':
            d_eqn = sy.Eq(1/self.u_eqn_geom().rhs**self.u_dyn_expt,
                          1/(self.u_eqn_dyn_manning().rhs))
        else:
            raise NameError('Unknown friction model "{}"'.format(friction_model))
        return sy.Eq(d_eqn.as_poly(d).args[0])

    def u_eqn_poly(self):
        u_eqn = sy.Eq(self.d_eqn_geom().rhs,self.d_eqn_dyn().rhs)
        if self.friction_model=='chezy':
            return sy.Eq(u_eqn.as_poly(1/sy.sqrt(u)).args[0]*u**3)
        elif self.friction_model=='manning':
            tmp = u_eqn.as_poly(1/sy.sqrt(u)).args[0]
            return sy.Eq((tmp.subs(u,t**2)*t**5).simplify().subs(t,sy.sqrt(u))
                         *n_m**sy.Rational(3,2))
        else:
            raise NameError('Unknown friction model "{}"'.format(friction_model))

#     def x(self):
#         return xxxxx
# 
#     def x(self):
#         return xxxxx



class new_symbolic_mixin(trig_utils_mixin):
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

    def set_ud_lambdas(self):
        from sympy import sqrt
        u_eqn = self.specify_u_polynomial_constants({w:w,chi:chi}).args[0]
        d_eqn = self.specify_d_polynomial_constants({w:w,chi:chi}).args[0]
        self.u_lambda = sy.utilities.lambdify(((u,w,chi),), u_eqn, 'sympy')
        self.d_lambda = sy.utilities.lambdify(((d,w,chi),), d_eqn, "sympy")        
    
class new_numerical_mixin(trig_utils_mixin, hydraulics_utils_mixin):
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
#         warnings.simplefilter('ignore')
        self.u_init = np.float64(self.u_i)
        self.d_init = np.float64(self.d_i)
        ux = newton(self.u_for_w_chi,self.u_init, args=[w,chi])
        dx = newton(self.d_for_w_chi,self.d_init, args=[w,chi])
#         if w>100 and w<101 and chi>0.66 and chi<0.67:
#             print(w,chi,' : ',ux,dx)
        return ux,dx  

    def nsolve_u_d_for_w_chi_fast(self, w,chi):
        ux = newton(self.u_for_w_chi,self.u_init, args=[w,chi])
        dx = newton(self.d_for_w_chi,self.d_init, args=[w,chi])
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
#         wx = d/np.tan(np.float64(self.theta)) # if self.theta<self.pi/2 else 0
#         return np.float64( 
#                             ( w + wx )
#                             /( (w*d+ wx)/(w+2*wx) )    
#                         )
        return np.float64( ((w+d/np.tan(np.float64(self.theta)))/d) )
    
    
    
class old_numerical_mixin(trig_utils_mixin, hydraulics_utils_mixin):
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
#         warnings.simplefilter('ignore')
        self.u_init = np.float64(self.u_i)
        self.d_init = np.float64(self.d_i)
        ux = newton(self.u_for_w_chi,self.u_init, args=[w,chi])
        dx = newton(self.d_for_w_chi,self.d_init, args=[w,chi])
#         if w>100 and w<101 and chi>0.66 and chi<0.67:
#             print(w,chi,' : ',ux,dx)
        return ux,dx  

    def nsolve_u_d_for_w_chi_fast(self, w,chi):
        ux = newton(self.u_for_w_chi,self.u_init, args=[w,chi])
        dx = newton(self.d_for_w_chi,self.d_init, args=[w,chi])
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
#         wx = d/np.tan(np.float64(self.theta)) # if self.theta<self.pi/2 else 0
#         return np.float64( 
#                             ( w + wx )
#                             /( (w*d+ wx)/(w+2*wx) )    
#                         )
        return np.float64( ((w+d/np.tan(np.float64(self.theta)))/d) )
        
class old_symbolic_mixin(trig_utils_mixin):
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

    def set_ud_lambdas(self):
        from sympy import sqrt
        u_eqn = self.specify_u_polynomial_constants({w:w,chi:chi}).args[0]
        d_eqn = self.specify_d_polynomial_constants({w:w,chi:chi}).args[0]
        self.u_lambda = sy.utilities.lambdify(((u,w,chi),), u_eqn, 'sympy')
        self.d_lambda = sy.utilities.lambdify(((d,w,chi),), d_eqn, "sympy")        
    
