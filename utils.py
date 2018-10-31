import sympy as sy
import numpy as np                  
from scipy.interpolate import interp1d, interp2d

from symbols import *

def sinuous_arcs(L,R,is_below_threshold=True):
    R = np.float(R)
    L = np.float(L)
    thm=25
    xy = []
    if not is_below_threshold:
        phi = np.arcsin((L/2)/R)
        H = R*np.cos(phi)
        tha = np.linspace(phi,2*np.pi-phi)
        up_or_down = -1.0
        xy += [(  -L+R*np.sin(tha[0:thm]), -(H-R*np.cos(tha[0:thm])))]
        xy += [(     R*np.sin(tha),        +(H-R*np.cos(tha)))]
        xy += [(   L+R*np.sin(tha),        -(H-R*np.cos(tha)))]
        xy += [( 2*L+R*np.sin(tha),        +(H-R*np.cos(tha)))]
        xy += [( 3*L+R*np.sin(tha[-thm:]), -(H-R*np.cos(tha[-thm:])))]
        chi = ((2*(np.pi-phi)*R)/L-1)
    else:
        phi = np.arcsin((L/2)/R)
        H = R*np.cos(phi)
        tha = np.linspace(-phi,phi)
        xy += [(  -L+R*np.sin(tha[-thm:]), -(-H+R*np.cos(tha[-thm:])))]
        xy += [(     R*np.sin(tha),         (-H+R*np.cos(tha)))]
        xy += [(   L+R*np.sin(tha),        -(-H+R*np.cos(tha)))]
        xy += [( 2*L+R*np.sin(tha),        +(-H+R*np.cos(tha)))]
        xy += [( 3*L+R*np.sin(tha[:thm]), -(-H+R*np.cos(tha[:thm])))]
        chi = 2*phi*R/L-1
    return phi,H,chi,xy
    
def bessel_zero_approx(n_terms=9):
    return sy.series(sy.besselj(0, omega),n=n_terms).removeO()
def chi_bessel_lambda_sympy(n_terms=9):
    return sy.lambdify(omega, 1/bessel_zero_approx(n_terms)-1,'sympy')
def chi_bessel_lambda_numpy(n_terms=9):
    return sy.lambdify(omega,1.0/bessel_zero_approx(n_terms)-1.0,'numpy')
def struve_zero_series():
    return sy.Sum((-1)**n*omega**(2*n+1)/((sy.factorial2(2*n+1))**2), (n, 0, sy.oo))
def struve_zero_approx(n_terms):
    series = sy.simplify(struve_zero_series().doit().series(n=n_terms))
    return sy.Add(*series.args[0:-1])
def mndim_struve_lambda_sympy(n_terms=9):
    return sy.lambdify(omega, struve_zero_approx(n_terms)/sy.pi,'sympy')
def mndim_struve_lambda_numpy(n_terms=9):
    return sy.lambdify(omega, struve_zero_approx(n_terms)/np.pi,'numpy')

def m_chi_Ls_from_omega_L(omega,L):
    chi = chi_bessel_lambda(np.array(omega))
    Ls = L*(1+chi)
    m = Ls*mndim_struve_lambda(np.array(omega))
    return m,chi,Ls

def m_chi_Ls_R_from_omega_L(omega,L,ds=0.01):
    m,chi,Ls = m_chi_Ls_from_omega_L(np.float(omega),L)
    s_array = np.array((0,ds))
    sgc_dx_dy_array = sgc_dx_dy( s_array, np.float(omega), Ls)*ds
    x_array = sgc_dx_dy_array[0]
    y_array = sgc_dx_dy_array[1]
    b=np.hypot(x_array[0],y_array[0])
    c=0.5*np.hypot(x_array[0]+x_array[1],y_array[0]+y_array[1])
    R = b/(2*np.sqrt(1-(c/(b))**2)) if c/b!=1 else 100.0 #np.finfo(np.float32).max
    return (m,chi,Ls,R)

def m_chi_Ls_R_from_omegas_L(omega_array,L,ds=0.01):
    rtn_array = np.zeros((omega_array.shape[0],4),dtype=np.float64)
    for idx,omega in enumerate(omega_array):
        rtn_array[idx] = np.array(m_chi_Ls_R_from_omega_L(omega,L,ds),dtype=np.float64)
    return rtn_array.T
    
def s_samples(Ls=None, n_loops=3, n_steps=100):
    L_sampled = Ls*n_loops
    return \
        np.arange(-L_sampled/(n_steps),L_sampled*(1+1/n_steps/2),L_sampled/(n_steps))+Ls/2

def sgc_dx_dy(s,omega,Ls):
    return np.array( (np.cos(omega*np.sin(np.pi*s/Ls)),
                      np.sin(omega*np.sin(np.pi*s/Ls)))  )

def sgc_integrate(s_array,sgc_dx_dy_array):
    ds = s_array[1]-s_array[0]
    v = sgc_dx_dy_array[0]
    x_array = np.cumsum( (v[:-1]+v[1:])/2 )*ds
    x_array -= x_array[0]
    v = sgc_dx_dy_array[1]
    y_array = np.cumsum( (v[:-1]+v[1:])/2 )*ds
    y_array-= y_array[0]
    return x_array,y_array

def sgc(omega=2*sy.pi/3,L=1, n_loops=3, n_steps=100):
    _,_,Ls = m_chi_Ls_from_omega_L(omega,L)
    omega_x = np.float(sy.N(omega))
    Ls_x = np.float(sy.N(Ls))
    s_array = s_samples(Ls=Ls_x, n_loops=n_loops, n_steps=n_steps)
    sgc_dx_dy_array = sgc_dx_dy(s_array,omega_x,Ls_x)
    x_array,y_array = sgc_integrate(s_array,sgc_dx_dy_array)
    return s_array, sgc_dx_dy_array, x_array,y_array
    
chi_bessel_lambda = chi_bessel_lambda_numpy()
mndim_struve_lambda = mndim_struve_lambda_numpy()

def interp_R_chi(m_chi_Ls_R_array, interp_method = 'quadratic'):
    chi_interp_for_m = interp1d(m_chi_Ls_R_array[0],m_chi_Ls_R_array[1],
                                kind=interp_method)
    R_interp_for_chi = interp1d(m_chi_Ls_R_array[1],m_chi_Ls_R_array[3],
                                kind=interp_method)
    return chi_interp_for_m,R_interp_for_chi

class trig_mixin():
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

class switch_mixin():
    def gt_smooth(self, x,x0, k):
        return 0.5*(1+np.tanh((x-np.float64(x0))/k))

    def lt_smooth(self, x,x0, k):
        return self.gt_smooth(-x,-x0,k)

    def switch_smooth(self, x,x0, fn_below, fn_above, k=1.0):
#         return ( self.lt_smooth(x,x0,k)*fn_below + self.gt_smooth(x,x0,k)*fn_above )
#         lt_smooth = self.lt_smooth(x,x0,k)
#         return ( lt_smooth*fn_below + (1-lt_smooth)*fn_above )
        return ( self.lt_smooth(x,x0,k)*(fn_below-fn_above)+fn_above )
    
