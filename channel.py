import numpy as np                  
import sympy as sy
from scipy.interpolate import interp2d
import mixin

from symbols import *
from utils import switch_mixin
from basic import model
from hydraulics import open_channel_flow
from bend import bend_mixin
from sediment import sediment_transport_mixin
from wear import bedrock_wear_mixin

class channel_model(model, open_channel_flow, switch_mixin, 
                    bend_mixin, sediment_transport_mixin, bedrock_wear_mixin):
    def tanphi(self, u,v,theta):
        return ( (u/v)*self.cosec(theta)-self.cot(theta) )
    def cotphi(self, u,v,theta):
        return 1/self.tanphi(u,v,theta)
    def phi(self, u,v,theta):
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


