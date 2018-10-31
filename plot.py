import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from decimal import Decimal
from scipy.optimize import curve_fit

default_dpi = 100
default_rect_xy_inches = (6,4)
default_sqr_xy_inches = (6,6)
default_cmap = cm.brg_r
# default_cmap = cm.cool
# default_cmap = cm.viridis

def create_fig(dpi=None, xy_inches=None):
    fig = plt.figure();
    if dpi is None:
        dpi = default_dpi
    fig.set_dpi(dpi);
    if xy_inches is None:
        xy_inches = default_rect_xy_inches
    fig.set_size_inches(*xy_inches)
    return fig

def plot_arcs(xy_array,yminmax=None):
    fig = plt.figure(xy_inches=(8,8))
#     fig.set_size_inches(8,8)
    [plt.plot(*xy,'b') for xy in xy_array]
    axes = plt.gca()
    axes.set_aspect('equal')
    plt.ylim(-np.float(yminmax),np.float(yminmax))

def make_polyline(points, color, ls, lw, alpha):
    return plt.Polygon(points, 
                       closed=None, fill=None, edgecolor=color,
                       lw=lw, ls=ls,  alpha=alpha);
def make_polygon(points, fcolor, ecolor, ls, lw, alpha):
    return plt.Polygon(points, 
                       closed=True, fill=True, 
                       edgecolor=ecolor, facecolor=fcolor,
                       lw=lw, ls=ls,  alpha=alpha);
def make_arrow(points, color, ls, lw, alpha):
    xy,dxy = points[0], points[1]
    return mpatches.FancyArrow(*xy, dxy[0]-xy[0], dxy[1]-xy[1], 
                               head_width=0.15, head_length=0.3,
                               length_includes_head=True, color=color,
                               lw=lw, ls=ls, alpha=alpha);
    
def plot_corner_fig(sm):
    fig = create_fig(xy_inches=(7, 8));
    text_params = {'ha': 'center', 'family': 'sans-serif', 'fontweight': 'medium'}
    la = 0.5
    hw = 2.0
    d = 1.5
    dw = 1.2
    dt = 1.0
    theta = np.pi/3
    u = 2.0
    v = 3.0
    dd = d+v*dt
    theta_deg = np.rad2deg(theta)
    dv = np.float64(v*sm.tanphi(u,v,theta))
    
    def cosec(th):
        return 1.0/np.sin(th)
    
    def cot(th):
        return 1.0/np.tan(th)

    def plot_lines(vector_list, line_type='line'):
        for ld in vector_list: 
            if ld is not None:
                pts = ld['pts']
                if line_type is 'arrow':
                    plt.gca().add_patch(
                        make_arrow(ld['pts'],ld['clr'],
                                   ld['ls'],ld['lw'],ld['la']));
                else:
                    plt.gca().add_line(
                        make_polyline(pts,ld['clr'],
                                      ld['ls'],ld['lw'],ld['la']));
                midpt = (pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2
                if pts[0][0]!=pts[1][0]:
                    a = np.array((pts[0][1]-pts[1][1])/(pts[0][0]-pts[1][0]),
                                 dtype=np.float64)
                    angle = np.rad2deg(np.arctan(a))
                else:
                    angle = 90
                this_text_params = text_params.copy()
                if 'va' in ld.keys():
                    this_text_params.update({'va': ld['va']})
                else:
                    this_text_params.update({'va': 'bottom' })

                plt.text(*midpt, ld['lbl'], color=ld['clr'], size=12, 
                         **this_text_params,
                         rotation=angle, rotation_mode='anchor');

    fills = (
        {'pts': ((0, 0), (hw, 0), (hw+dw*cot(theta), dw), (0, dw)), 
         'eclr':'w', 'fclr':'b', 'ls':'-', 'lw':2, 'la':0.05},
        {'pts': ( (0, -v*dt), (hw, -v*dt), (hw+dv, -v*dt), 
                  (hw+dv+dd*cot(theta), -v*dt+dd),
                  (hw+dv+dd*cot(theta)*1.1, -v*dt+dd),
                 (hw+dv*1.1, -v*dt*1.1), (hw, -v*dt*1.1), (0, -v*dt*1.1)),
         'eclr':'w', 'fclr':'brown', 'ls':'-', 'lw':2, 'la':0.1},
        None)
    for ld in fills:
        if ld is not None:
            plt.gca().add_line(
                make_polygon(ld['pts'],ld['fclr'],ld['eclr'],
                             ld['ls'],ld['lw'],ld['la']));

    bdry_lines = ( 
        {'pts':((0, 0), (hw, 0), (hw+d*cot(theta), d)), 
                 'clr':'brown', 'ls':'-', 'lw':1.5, 'la':la},
        {'pts':((0, -v*dt), (hw, -v*dt), (hw+dv, -v*dt), 
                (hw+dv+dd*cot(theta), -v*dt+dd)), 
                 'clr':'brown', 'ls':'-', 'lw':2, 'la':la}
    )
    for ld in bdry_lines:    
        plt.gca().add_line(
            make_polyline(ld['pts'],ld['clr'],ld['ls'],
                          ld['lw'],ld['la']));

    dashed_lines = ( 
        {'pts':((hw,0), (hw, -v*dt)), 
          'clr':'r', 'ls':'-.', 'lw':1, 'la':la, 
          'lbl':'$\\xi_z$'},
        {'pts':( (hw, -v*dt), (hw+dv, -v*dt)), 
          'clr':'r', 'ls':'-.', 'lw':1, 'la':la, 
          'lbl':'$\\xi_z\,\\tan\,\\phi$', 'va': 'top'},
        {'pts':( (hw,0), (hw+u*dt*cosec(theta), 0) ), 
          'clr':'r', 'ls':'-.', 'lw':1, 'la':la, 
          'lbl':'$\\xi_y\\,cosec\,\\theta$', 'va': 'top'},
        {'pts':( (hw+u*dt*(cosec(theta)-np.sin(theta)),u*dt*np.cos(theta)), 
                 (hw+u*dt*cosec(theta), 0) ), 
          'clr':'r', 'ls':'-.', 'lw':1, 'la':la, 
          'lbl':'$\\xi_y$'},
#         {'pts':(  (hw+dv, -v*dt), (hw+u*dt*cosec(theta), 0) ), 
#           'clr':'r', 'ls':'-', 'lw':1, 'la':la, 
#           'lbl':'$\\xi_z\\,cosec\,\\theta$'},
        {'pts':(  (hw-u*dt*cosec(theta)+v*dt*sm.tanphi(u,v,theta), -v*dt), 
                (hw, 0) ), 
          'clr':'r', 'ls':'-.', 'lw':1, 'la':la, 
          'lbl':'$\\xi_z\\,cosec\,\\theta$'},
        {'pts':(  (hw-u*dt*cosec(theta)+v*dt*sm.tanphi(u,v,theta), -v*dt), 
                  (hw, -v*dt) ), 
          'clr':'r', 'ls':'-.', 'lw':1, 'la':la, 'va': 'top',
          'lbl':'$\\xi_z\\,cot\,\\theta$'},
        None
    )
    als = 0.35
    alf = 1-als
    zoff = 0.2
    yoff = 1.0
    motion_vectors = ( 
        {'pts': ((hw,0), ((hw+dv), (-v*dt))), 
          'clr':'b', 'ls':'-', 'lw':2, 'la':la, 
          'lbl':'$\\xi_z\\,sec\,\\phi$'}, 
        {'pts': ((hw*zoff,-v*dt*als), (hw*zoff, -v*dt*alf)),
          'clr':'brown', 'ls':'-', 'lw':2, 'la':la, 'lbl':''}, 
        {'pts':((hw+u*dt*(yoff*np.cos(theta)+np.sin(theta)*als), 
                    u*dt*(yoff*np.sin(theta)-np.cos(theta)*als)), 
                (hw+u*dt*(yoff*np.cos(theta)+np.sin(theta)*alf), 
                    u*dt*(yoff*np.sin(theta)-np.cos(theta)*alf))), 
          'clr':'brown', 'ls':'-', 'lw':2, 'la':la, 'lbl':''},
        None
    )
    plot_lines(dashed_lines)
    plot_lines(motion_vectors, line_type='arrow')
    angles = ( 
        {'pts':(hw+max(0.15,u*dt*(cosec(theta)-np.sin(theta))/3), 
                u*dt*np.cos(theta)/3/3),  'clr':'r', 'lbl': '$\\theta$'},
        {'pts':
         ( hw+v*dt*0.3*np.tan(np.float64(sm.phi(u,v,theta)/2)),-v*dt*0.3 ),  
         'clr':'b', 'lbl': '$\\phi$'},
#         {'pts': (hw+dv-dv*0.4, -v*dt+dv*0.4*np.tan(phi(u,v,theta)/2)) ,
#          'clr':'b', 'lbl': '$\\frac{\\pi}{2}-\\phi$'},
#         {'pts': (hw+u*dt*cosec(theta)*0.15, 
#                   -u*dt*cosec(theta)*0.075*cot(phi(u,v,theta)/2)) ,
#          'clr':'b', 'lbl': '$\\frac{\\pi}{2}-\\phi$'},
        None
    )
    angle_text_params = {'ha':'center', 'va':'center', 
                         'family':'sans-serif','fontweight':'medium'}
    for ld in angles:
        if ld is not None:
            plt.text(*ld['pts'], ld['lbl'], color=ld['clr'], size=12, 
                             **angle_text_params);
    plt.axis('scaled')
    sm.figs.update({'corner': fig})

def plot_sgc_loops(sm, x_array,y_array,L_x,m_x):
    fig = create_fig(xy_inches=(8,8));
    plt.plot(x_array,y_array,'k-', label='channel centerline')
#     plt.plot(0,0,'ro')
    plt.plot(L_x,0,'bo',label='$x=L$')
    plt.plot(L_x/2,m_x,'mo',label='$y=m$')
    axes = plt.gca()
    axes.set_aspect('equal')
    plt.grid(ls=':')
    plt.ylabel('y/L   [-]')
    plt.xlabel('x/L   [-]')
    axes.legend()
    sm.figs.update({'sgc_loops': fig}) 
    
def plot_R_for_chi_bessel(sm, chi_array,R_array, interp_fn=None, 
                          classic_approx=None, data=None):
    fig = create_fig(xy_inches=(7,5));
    plt.plot(chi_array,R_array, 'blue', lw=1.5, label='SGC full Bessel-Struve series')
    if interp_fn is not None:
        plt.plot(chi_array,interp_fn(chi_array), 'k', dashes=[3, 6], lw=3,
                 label='interpolating function')
    if classic_approx is not None:
        shrink = chi_array.shape[0]-classic_approx.shape[0]
        plt.plot(chi_array[shrink:],classic_approx, 'k:', 
                 label='Langbein & Leopold approx')
    def R_model(chi,a,b):
        return a*(chi+1)**b/(np.sqrt((np.sqrt(chi+1)-1)))
    popt,pcov = curve_fit(R_model,chi_array[1:],R_array[1:])
    print(popt);
    plt.plot(chi_array[1:],R_model(chi_array[1:],*popt), 'r', dashes=[2, 8], lw=3,
                 label='Mecklenburg & Jayakaran model fit')
    if data is not None:
        plt.plot(*data[0], 'ko', label=data[1], alpha=0.4, ms=7)
    axes = plt.gca()
    plt.grid(ls=':')
    plt.ylabel('Non-dimensional radius of curvature  $R/L$   [-]');
    plt.xlabel('Sinuosity  $\chi$   [-]');
#     plt.autoscale(enable=True,tight=True)
    plt.xlim(0,3)
    plt.ylim(0,0.9)
    axes.legend(loc='lower right')
    sm.figs.update({'R_chi_bessel': fig}) 
    
def plot_omega_for_chi(sm, chi_array,omega_array, 
                       interp_fn=None, closed_form=None, data=None):
    fig = create_fig()
    plt.plot(chi_array,omega_array, 'blue', lw=1.5, label='SGC full Bessel series')
    if interp_fn is not None:
        plt.plot(chi_array,interp_fn(chi_array), 'k', dashes=[3, 6], lw=2,
                 label='interpolating function')
    if closed_form is not None:
        shrink = chi_array.shape[0]-closed_form.shape[0]
        plt.plot(chi_array[shrink:],closed_form, 'r', dashes=[2, 4], lw=2.5,
                 label='4-term closed-form approx')
    if data is not None:
        plt.plot(*data[0], 'ko', label=data[1], alpha=0.4, ms=7)
    axes = plt.gca()
    plt.grid(ls=':')
    plt.ylabel('Maximum angular deflection  $\omega$   [rads]');
    plt.xlabel('Sinuosity  $\chi$   [-]');
    plt.autoscale(enable=True, tight=True)
    plt.xlim(0,3)
    plt.ylim(0,2)
#     axes.legend(loc='best', bbox_to_anchor=(0.25, 0., 0.5,1))
    axes.legend(loc='lower right') #, bbox_to_anchor=(0.25, 0., 0.5,1)
    sm.figs.update({'omega_chi': fig})

def plot_chi_for_m(sm, m_array,chi_array, interp_fn=None, data=None):
    fig = create_fig()
    plt.plot(m_array,chi_array, 'blue', lw=1.5, label='SGC full Bessel-Struve series')
    if interp_fn is not None:
        plt.plot(m_array,interp_fn(m_array), 'k', dashes=[3, 6], lw=3,
                 label='interpolating function')
    def chi_model(m,a,b,c):
        return m**a/(1+m**np.abs(c))*b
#         return m**2/(1+m**1)*np.pi
    popt,pcov = curve_fit(chi_model,m_array,chi_array)
    plt.plot(m_array,chi_model(m_array,*popt), 'r', dashes=[2, 8], lw=3,
                 label='model fit')
    if data is not None:
        plt.plot(*data[0], 'ko', label=data[1], alpha=0.4, ms=7)
    axes = plt.gca()
    plt.grid(ls=':')
    plt.ylabel('Sinuosity  $\chi$   [-]');
    plt.xlabel('Non-dimensional bend distance  $m/L$   [-]');
    plt.autoscale(enable=True,tight=True)
    plt.xlim(0,1.6)
    plt.ylim(0,3)
    axes.legend()
    sm.figs.update({'chi_m': fig})

def plot_R_for_chi(sm, L=1000):
    fig = create_fig()
    chi_vec = np.linspace(0.05,3)
    plt.plot(chi_vec, sm.R_for_chi(chi_vec,L)/L,
                   label='$L={}$'.format(L));
    plt.ylabel('Radius of curvature  $R(\\chi)/L$   [-]');
    plt.xlabel('Sinuosity $\\chi$   [-]');
    axes = plt.gca();
#     axes.legend()
    axes.set_ylim(0,None)
    axes.set_xlim(0,chi_vec[-1])
#     axes.axhline(y=L/2,alpha=0.5,color='k',
#                  linewidth=1,linestyle=':');
#     axes.axvline(x=1.0,alpha=0.5,color='k',
#                  linewidth=1,linestyle=':');
    plt.grid(ls=':')
    sm.figs.update({'R_chi': fig})

def plot_R_for_R(sm, interp_fn, df):
    fig = create_fig(xy_inches=(6.5,6.5))
    plt.plot(interp_fn(df['sinuosity_chi'])*df['wavelength']/2,
             df['radius_R'],
             'ko',alpha=0.4, label='Williams 1986 data for $\chi_{obs}$, $R_{obs}$')
    plt.xlabel('Bessel-Struve SGC model radius of curvature $R_{model}(\chi_{obs})$  [m]')
    plt.ylabel('Radius of curvature $R_{obs}$  [m]')
    plt.xlim(1,1e4)
    axes = plt.gca()
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_aspect('equal')
    plt.plot(np.arange(*axes.get_xlim()),np.arange(*axes.get_xlim()), 
             label='$R_{model} = R_{obs}$')
    plt.legend();
    plt.grid(ls=':')
    axes.autoscale(enable=True, tight=True);
    sm.figs.update({'R_R': fig})

def plot_epsilon_for_chi(sm, w=10,L=1000):
    fig = create_fig()
    chi_vec = np.linspace(0.,3)
    line = plt.plot(chi_vec,sm.epsilon_for_w_chi_L(w,chi_vec,L),
                   label='$w/L = {}$'.format(w/L));
    plt.ylabel('Bend excess speed $\\epsilon(\\chi)$   [-]');
    plt.xlabel('Sinuosity $\\chi$   [-]');
    axes = plt.gca();
    axes.legend()
    axes.set_ylim(0,None)
    axes.set_xlim(0,chi_vec[-1])
#     axes.axhline(y=2*w/L,alpha=0.5,color='k',
#                  linewidth=1,linestyle=':');
#     axes.axvline(x=1.0,alpha=0.5,color='k',
#                  linewidth=1,linestyle=':');
    plt.grid(ls=':')
    sm.figs.update({'epsilon_chi': fig})

def plot_dydt_xiz(sm):
    fig = create_fig()
    theta = np.pi/3
    uv_array = np.linspace(0.1,3)
    u_list = [0.1,1,2]
    for u in u_list:
        phi_array = np.array([v*sm.tanphi(u,v,theta) for v in uv_array])
        plt.plot(uv_array,phi_array, label='$\\xi_y={}$'.format(u));
    plt.xlabel('$\\xi_z$');
    plt.ylabel('$\\xi_z\,\\phi(\\xi_y,\\xi_z)$');
    axes = plt.gca();
    axes.legend()
    axes.axhline(alpha=0.5,color='k',linewidth=1,linestyle=':');
    sm.figs.update({'dydt_xiz': fig})

def plot_dydt_xiy(sm):
    fig = create_fig()
    theta = np.pi/3
    uv_array = np.linspace(0.1,2)
    v_list = [0.1,1,2]
    for v in v_list:
        phi_array = np.array([v*sm.tanphi(u,v,theta) for u in uv_array])
        plt.plot(uv_array,phi_array, label='$\\xi_z={}$'.format(v));
    plt.xlabel('$\\xi_y$');
    plt.ylabel('$\\xi_z\,\\phi(\\xi_y,\\xi_z)$');
    axes = plt.gca();
    axes.legend()
    axes.axhline(alpha=0.5,color='k',linewidth=1,linestyle=':');
    sm.figs.update({'dydt_xiy': fig})

def plot_u_w(sm, chi_list, wud_vec_list):
    fig = create_fig()
    for chi_i in chi_list:
        line, = plt.plot(wud_vec_list[chi_i][:,0],
                         wud_vec_list[chi_i][:,1]);
        line.set_label('$\\chi = {}$'.format(chi_i))
    plt.xlabel('Base width $w$');
    plt.ylabel('Flow speed $u(w, \\chi)$');
    axes = plt.gca();
    axes.legend()
#     axes.autoscale(enable=True, axis='x', tight=True)
    axes.set_ylim(0,)
    axes.set_xlim(0,100)
    sm.figs.update({'u_w': fig})

def plot_d_w(sm, chi_list, wud_vec_list):
    fig = create_fig()
    for chi_i in chi_list:
        line, = plt.plot(wud_vec_list[chi_i][:,0],
                         wud_vec_list[chi_i][:,2]);
        line.set_label('$\\chi = {}$'.format(chi_i))
    plt.xlabel('Base width $w$');
    plt.ylabel('Flow depth $d(w, \\chi)$');
    axes = plt.gca();
    axes.legend()
#     axes.autoscale(enable=True, axis='x', tight=True)
    axes.set_ylim(0,)
    axes.set_xlim(0,100)
    sm.figs.update({'d_w': fig})
    
def plot_u_d(sm, chi_list, wud_vec_list):
    fig = create_fig();
    for chi_i in chi_list:
        line, = plt.plot(wud_vec_list[chi_i][:,1],
                         wud_vec_list[chi_i][:,2]);
        line.set_label('$\\chi = {}$'.format(chi_i))
    plt.xlabel('Flow speed $u(w, \\chi)$');
    plt.ylabel('Flow depth $d(w, \\chi)$');
    axes = plt.gca();
    axes.legend()
    axes.set_ylim(0,)
#     axes.autoscale(enable=True, tight=True)
    sm.figs.update({'u_d': fig})

def start_fig(sm):
    fig = create_fig(xy_inches=default_sqr_xy_inches)
    return fig

def finish_fig(sm, fig, fig_name=''):
    sm.figs.update({fig_name: fig})
    
def contour_grid_for_w_chi_overlay(sm, fig, w_vec, chi_vec, grid_array, 
                                    levels=20, title='', fig_name='',
                                    sf=1, fmt='%1.1f', is_tanphi=False,
                                    fat_line=None, do_plot_contours=True):
#     fig = create_fig(xy_inches=default_sqr_xy_inches);
    cmap = default_cmap
    if is_tanphi:
        grid_array = np.rad2deg(np.arctan(grid_array))
    plt.xlabel('Base width $w(t)$');
    plt.ylabel('Sinuosity $\chi(t)$');
    plt.title(title);
    axes = plt.gca();
    if do_plot_contours:
        contour_list = plt.contour(w_vec, chi_vec, grid_array*sf, levels, cmap=cmap)
        axes.clabel(contour_list, inline=1, fontsize=10, fmt=fmt)
    if fat_line is not None:
        fatcontour = plt.contour(w_vec,chi_vec, grid_array*sf, np.array([fat_line[0]]), 
                                   colors=[fat_line[1]], alpha=fat_line[4], 
                                   linewidths=[fat_line[3]], 
                                   linestyles=[fat_line[5]])
        if fat_line[2]:
            axes.clabel(fatcontour, inline=1, fontsize=10, fmt=fmt)
#     sm.figs.update({fig_name: fig})
    return fig
    
def contour_grid_for_w_chi(sm, w_vec, chi_vec, grid_array, 
                            levels=20, title='', fig_name='',
                            sf=1, fmt='%1.1f', is_tanphi=False,
                            fat_line=None, do_plot_contours=True):
    fig = start_fig(sm);
    contour_grid_for_w_chi_overlay(sm, fig, w_vec, chi_vec, grid_array, 
                                    levels=levels, title=title, fig_name=fig_name,
                                    sf=sf, fmt=fmt, is_tanphi=is_tanphi,
                                    fat_line=fat_line, do_plot_contours=do_plot_contours)
    finish_fig(sm, fig, fig_name=fig_name)
    
def quiver_grids_for_w_chi_overlay(sm, fig, w_mesh, chi_mesh, dwdt_array, dchidt_array, 
                                   subsample=1, title='', fig_name='',
                                   scale=100, fmt='%1.1f',cmap='brg_r'):
    q_cmap = getattr(cm,cmap)
    plt.xlabel('Base width $w(t)$');
    plt.ylabel('Sinuosity $\chi(t)$');
    plt.title(title);
    axes = plt.gca();
    X, Y = w_mesh, chi_mesh
    U = np.array(dwdt_array,dtype=np.float64)
    V = np.array(dchidt_array,dtype=np.float64)
    H = np.hypot(U,V)
    expt = 1
    U /= np.power(H,expt)
    V /= np.power(H,expt)
    C = np.log(H+0.0001)
    ss = subsample
    arrow_list = axes.quiver(X[::ss,::ss], Y[::ss,::ss], U[::ss,::ss], V[::ss,::ss], 
                             C[::ss,::ss], cmap=q_cmap
                             , pivot='mid',  angles='xy'
                            , units='inches', scale=scale
                             )
    axes.autoscale(enable=True, axis='x', tight=True)
    axes.autoscale(enable=True, axis='y', tight=True)

def quiver_grids_for_w_chi(sm, w_mesh, chi_mesh, dwdt_array, dchidt_array, 
                            subsample=1, title='', fig_name='',
                            scale=100, fmt='%1.1f',cmap='brg_r', 
                            do_channel_line=False, 
                            ode_integrations=None, ode_interps=None, ode_resamples=None,
                            ensemble_mean = None, 
                            n_interp_pts=100,
                            initial_points=['gray',0.8,4],
                            solution_line=['k',0.8,1,3],
                            final_points=['gray',0.6,8]):
    fig = start_fig(sm);
    quiver_grids_for_w_chi_overlay(sm, fig, w_mesh, chi_mesh, dwdt_array, dchidt_array, 
                                   subsample=subsample, title=title, fig_name=fig_name,
                                   scale=scale, fmt=fmt,cmap=cmap);
    if do_channel_line:
        contour_grid_for_w_chi_overlay(sm, fig, w_mesh[0,:], chi_mesh[:,0], dwdt_array, 
                                        levels=1, title=title, fig_name=fig_name,
                                        sf=1, fmt=fmt, is_tanphi=True,
                                        fat_line=[0,'k',False,2,0.5,':'], 
                                        do_plot_contours=False)
    def plot_solns_points(w_chi_vecs,idx):
        if idx==0:
            initial_point_label='initial state'
            solution_line_label='evolution'
            final_point_label  ='final state'
        else:
            initial_point_label=None
            solution_line_label=None
            final_point_label  =None
        plt.plot(w_chi_vecs[0,0],w_chi_vecs[1,0], 
                 'o', color=initial_points[0], markeredgecolor='k',
                 alpha=initial_points[1], markersize=initial_points[2],
                 label=initial_point_label)
        plt.plot(w_chi_vecs[0],w_chi_vecs[1], 
                 solution_line[0], alpha=solution_line[1],
                 linewidth=solution_line[2],markersize=solution_line[3],
                 label=solution_line_label)
        plt.plot(w_chi_vecs[0,-1],w_chi_vecs[1,-1], 
                 'o', color=final_points[0], markeredgecolor='k',
                 alpha=final_points[1], markersize=final_points[2],
                 label=final_point_label)
        
    if ode_resamples is not None:
        for idx,ode_resample in enumerate(ode_resamples):
            w_chi_vecs = np.array((ode_resample[:,1],ode_resample[:,2]))
            plot_solns_points(w_chi_vecs,idx)
    elif ode_interps is not None:
        for idx,ode_interp in enumerate(ode_interps):
            t_vec = np.linspace(ode_integrations[idx][0,0],
                                ode_integrations[idx][-1,0], n_interp_pts)
            w_chi_vecs = ode_interp(t_vec)
            plot_solns_points(w_chi_vecs,idx)
    elif ode_integrations is not None:
        for idex,ode_integration in enumerate(ode_integrations):
            w_chi_vecs = np.array((ode_integration[:,1],ode_integration[:,2]))
            plot_solns_points(w_chi_vecs,idx)
            
    if ensemble_mean is not None:
        plt.errorbar(*ensemble_mean[0], 
                     xerr=ensemble_mean[1][0],
                     yerr=ensemble_mean[1][1],
                     ecolor='w', mec='w', 
                     color='w', fillstyle='full', 
                     alpha=ensemble_mean[3],
                     fmt='-o', markersize=ensemble_mean[4], markeredgewidth=3,
                     elinewidth=3,capthick=3,capsize=7);
        plt.errorbar(*ensemble_mean[0], 
                     xerr=ensemble_mean[1][0],
                     yerr=ensemble_mean[1][1],
                     ecolor=ensemble_mean[2], mec=ensemble_mean[2], 
                     color='w', fillstyle='full', 
                     alpha=ensemble_mean[3],
                     fmt='-o', markersize=ensemble_mean[4]/2, markeredgewidth=2,
                     elinewidth=0,capthick=0,capsize=0,
                     label='ensemble average');
        plt.errorbar(*ensemble_mean[0], 
                     xerr=ensemble_mean[1][0],
                     yerr=ensemble_mean[1][1],
                     ecolor=ensemble_mean[2], mec=ensemble_mean[2], 
                     color='w', fillstyle='full', 
                     alpha=ensemble_mean[3],
                     fmt='-o', markersize=ensemble_mean[4], markeredgewidth=2,
                     elinewidth=2,capthick=2,capsize=6);

    axes = plt.gca()
    axes.set_xlim(0,w_mesh[0,-1])
    axes.set_ylim(0,chi_mesh[-1,0])
    plt.legend(facecolor='w',edgecolor='k',framealpha=1,
               borderpad=0.5)
#     axes.set_xlim((w_mesh[0],w_mesh[1]))
    
    finish_fig(sm, fig, fig_name=fig_name)
    
def ensemble_chi_w_for_t(sm,ss):
    for idx,Qx in enumerate(ss.Q_array):
        title = 'Width & sinuosity evolution' \
                +' for $Q=${0}'.format(Qx)+' m$^3\,$s$^{-1}$'
        quiver_grids_for_w_chi(
                sm, ss.w_mesh_list[idx], ss.chi_mesh_list[idx], 
                ss.dwdt_array_list[idx], ss.dchidt_array_list[idx],
                title=title, 
                fig_name='quiver_dwchidt_from_w_chi',
                subsample=1, scale=6, fmt='%1.2f',
                do_channel_line=True,
                ode_integrations=ss.t_w_chi_vecs_lists[idx],
                ode_interps=ss.w_chi_interp_as_t_lists[idx],
                ode_resamples=ss.t_w_chi_resampled_vecs_array_list[idx],
                n_interp_pts=300,
                ensemble_mean=[ss.t_w_chi_means_array_list[idx][1:], 
                               ss.t_w_chi_stdevs_array_list[idx][1:],
                               'k',1,20]
            )

def ensemble_mean_chi_for_Q(sm,ss):
    Qs   = ss.Q_array
    chis = np.array([arr[2] for arr in ss.t_w_chi_means_array_list])
    chi_errs = np.array([arr[2] for arr in ss.t_w_chi_stdevs_array_list])
    Q_vec = np.linspace(0.,Qs[-1])
    fig = create_fig()
    axes = plt.gca()
    ab = ss.w_for_chi_model_fit
    a_str = str( ab[0].round(2) )
    b_str = str( ab[1].round(2) )
    plt.errorbar(Qs,chis,yerr=chi_errs,color='b',ecolor='gray',fmt='-o',
                 linewidth=1.5,elinewidth=1,capthick=1,capsize=5,
                 label='ensemble averages')
    model_line = axes.plot(Q_vec,ab[0]*np.power(Q_vec/1000,ab[1]), 
                            '-.', color='r', linewidth=1.5, alpha=1,
                            label='$\chi \\approx $' \
                            + '{0:.2f}'.format( Decimal(a_str).normalize() ) \
                            + '$(Q/1000)^{'+'{}'.format(Decimal(b_str).normalize())+'}$'        )
    plt.xlabel('Discharge $Q$  [m$\,$s$^{-1}]$')
    plt.ylabel('Sinuosity $\chi$   [-]')
    plt.legend(frameon=True,facecolor='w',edgecolor='w',framealpha=1)
    plt.xlim(0,);
    plt.ylim(0,);
    plt.grid(ls=':')
    sm.figs.update({'ensemble_mean_chi_for_Q': fig})
        
def ensemble_mean_w_for_Q(sm,ss):
    Qs   = ss.Q_array
    ws   = np.array([arr[1] for arr in ss.t_w_chi_means_array_list])
    w_errs = np.array([arr[1] for arr in ss.t_w_chi_stdevs_array_list])
    Q_vec = np.linspace(0.,Qs[-1])
    fig = create_fig()
    axes = plt.gca()
    a = ss.w_for_Q_model_fit
    a_str = str(a[0].round(0))
    data_line = axes.errorbar(Qs,ws,yerr=w_errs,color='k',ecolor='gray',fmt='-o',
                             linewidth=1.5,elinewidth=1,capthick=1,capsize=5,
                             label='ensemble averages');
    model_line = axes.plot(Q_vec,a[0]*np.sqrt(Q_vec/1000), 
                            '-.', color='r', linewidth=1.5, alpha=1,
                            label='$w \\approx $' \
                            + '{0:.0f}'.format( Decimal(a_str).normalize() ) \
                            + '$\sqrt{Q/1000}$'        )
    plt.xlabel('Discharge $Q$   [m$^3\,$s$^{-1}$]')
    plt.ylabel('Base width $w$  [m]')
    plt.legend(frameon=True,facecolor='w',edgecolor='w',framealpha=1)
    plt.xlim(0,);
    plt.ylim(0,);
    plt.grid(ls=':')
    sm.figs.update({'ensemble_mean_chi_for_Q': fig})
        
def ensemble_mean_w_for_rootQ(sm,ss):
    Qs   = ss.Q_array
    ws   = np.array([arr[1] for arr in ss.t_w_chi_means_array_list])
    w_errs = np.array([arr[1] for arr in ss.t_w_chi_stdevs_array_list])
    Q_vec = np.linspace(0.,Qs[-1])/1000
    fig = create_fig()
    axes = plt.gca()
    a = ss.w_for_chi_model_fit
    a_str = str(a[0].round(0))
#     plt.errorbar(np.sqrt(Qs/100),ws,yerr=w_errs,color='k',ecolor='k',fmt='-o',
#                  linewidth=1.5,elinewidth=1,capthick=1,capsize=5)
    model_line = axes.plot(Q_vec,a[0]*np.power(Q_vec/1000,a[1]), 
                            '-.', color='r', linewidth=2, alpha=1)
    plt.xlabel('Root normalized discharge $\sqrt{Q/Q_r}$   [-]')
    plt.ylabel('Base width $w$  [m]') 
    sm.figs.update({'ensemble_mean_chi_for_Q': fig})
        

        