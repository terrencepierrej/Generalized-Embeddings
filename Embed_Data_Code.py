# Terrence Pierre Jacques
# 02/5/2018
# This code is Version 3.0 of the embeddings code

import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy import optimize as opt
import SphHarmExpansion as she # Robs things
import SphHarmExpansion_new as nshe
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def Functions(parameters, theta, phi):
    var, var_theta, var_phi = nshe.Ylm(parameters, theta, phi)
    return var, var_theta, var_phi

p = Pool(3)

def Main(params, theta, phi):
    
    x_a = params[0:N]
    y_a = params[N:2*N]
    z_a = params[2*N:3*N]
    
    newFunc = partial(Functions, theta=theta, phi=phi)
    outputs = p.map(newFunc, [x_a, y_a,z_a])
    X, X_theta, X_phi = outputs[0]
    Y, Y_theta, Y_phi = outputs[1]
    Z, Z_theta, Z_phi = outputs[2]

    Gtt_e = X_theta*X_theta + Y_theta*Y_theta + Z_theta*Z_theta
    Gpp_e = X_phi*X_phi + Y_phi*Y_phi + Z_phi*Z_phi
    Gtp_e = X_theta*X_phi + Y_theta*Y_phi + Z_theta*Z_phi

#     R = np.sum((Gtt_e - g_th)*(Gtt_e - g_th)*sintheta_sq + 
#           (Gpp_e - g_ph)*(Gpp_e - g_ph)/sintheta_sq + 
#         (Gtp_e - g_th_ph)*(Gtp_e - g_th_ph)*2)*delta_th*delta_ph + np.sum(La*
#                     (x_a[n:]**EE + y_a[n:]**EE + z_a[n:]**EE))
    
    R = np.sum((Gtt_e - g_th)*(Gtt_e - g_th)*sintheta + 
               (Gpp_e - g_ph)*(Gpp_e - g_ph)/sintheta_cu + 
        (Gtp_e - g_th_ph)*(Gtp_e - g_th_ph)*2./sintheta)*delta_th*delta_ph
    
    
    return R

def Jac(params, theta, phi):
    
    x_a = params[0:N]
    y_a = params[N:2*N]
    z_a = params[2*N:3*N]
    
#     time1 = time.time() 
    
    newFunc = partial(Functions, theta=theta, phi=phi)
    outputs = p.map(newFunc, [x_a, y_a,z_a])
    X, X_theta, X_phi = outputs[0]
    Y, Y_theta, Y_phi = outputs[1]
    Z, Z_theta, Z_phi = outputs[2]
    
#     print time.time() - time1

    Gtt_e = X_theta*X_theta + Y_theta*Y_theta + Z_theta*Z_theta
    Gpp_e = X_phi*X_phi + Y_phi*Y_phi + Z_phi*Z_phi
    Gtp_e = X_theta*X_phi + Y_theta*Y_phi + Z_theta*Z_phi
    
    D_Y = nshe.D_Y(N, theta, phi)
    Dth_Y = D_Y[0]
    Dph_Y = D_Y[1]
    
    G1 = 4*(Gtt_e - g_th)*sintheta
    G2 = 4*(Gpp_e - g_ph)/sintheta_cu
    G3 = 4*(Gtp_e - g_th_ph)/sintheta
    
#     time2 = time.time()
    
    dXa = (np.einsum('ij,kij -> k', G1*X_theta, Dth_Y) + 
        np.einsum('ij,kij -> k', G2*X_phi, Dph_Y) + 
           np.einsum('ij,kij -> k', G3*X_phi, Dth_Y) + 
           np.einsum('ij,kij -> k', G3*X_theta, Dph_Y)
          )

    dYa = (np.einsum('ij,kij -> k', G1*Y_theta, Dth_Y) + 
        np.einsum('ij,kij -> k', G2*Y_phi, Dph_Y) + 
           np.einsum('ij,kij -> k', G3*Y_phi, Dth_Y) + 
           np.einsum('ij,kij -> k', G3*Y_theta, Dph_Y)
          )

    dZa = (np.einsum('ij,kij -> k', G1*Z_theta, Dth_Y) + 
        np.einsum('ij,kij -> k', G2*Z_phi, Dph_Y) + 
           np.einsum('ij,kij -> k', G3*Z_phi, Dth_Y) + 
           np.einsum('ij,kij -> k', G3*Z_theta, Dph_Y)
          )
#     print time.time() - time2
    
#     grad = np.concatenate((dXa + (EE*x_a**ex)*la, 
#                            dYa + (EE*y_a**ex)*la, dZa + (EE*z_a**ex)*la), axis=0)
    grad = np.concatenate((dXa, dYa, dZa),axis=0)
    

    return grad

Nfeval = 0
def callbackF(Xi):
    global Nfeval
    print '{0:4d}  {1: 3.10f}'.format(Nfeval,Main(Xi,theta,phi))
    Nfeval += 1
    
def callbackNew(x, f, accepted):
    global Nfeval
    print '{0:2d}  {1: 3.10f}  {2: 4d}'.format(Nfeval, f, int(accepted))
    Nfeval += 1
    
def Data_read(Gth_path, Gph_path, Gtp_path):
    th_data = open(Gth_path, 'r')
    ph_data = open(Gph_path, 'r')
    tp_data = open(Gtp_path, 'r')
    
    counterA = 0
    counterB = 0
    counterC = 0
    shp = (1,1)
    i=0
        
    for line in th_data:
        
        if 'Extents' in line:
            firstpar1 = line.find('(')
            mid1 = line.find(',')
            lastpar1 = line.find(')')
            shp = ( int(line[firstpar1 + 1 : mid1]), int(line[mid1 + 1 : lastpar1]) )
            new_arr = np.zeros(shp) 
            i=1
            
        if counterA == 0 and i==1:
            initial_arr0 = np.zeros((0,shp[0],shp[1]))
            counterA += 1

        if '(:' in line:
            firstpar = line.find('(')
            mid = line.find(',')
            lastpar = line.find(')')
            col = int(line[mid + 1 : lastpar])
            num_list = np.fromstring(line[lastpar + 3:], dtype='float64', sep=',')
            new_arr[:, col] = num_list
            i += 1

        if i>shp[1] and counterA == 1:
            intermed0 = np.concatenate((initial_arr0,new_arr[None,:,:]), axis=0)
            initial_arr0 = intermed0
            i=0
            
    for line in ph_data:
        
        if 'Extents' in line:
            firstpar1 = line.find('(')
            mid1 = line.find(',')
            lastpar1 = line.find(')')
            shp = ( int(line[firstpar1 + 1 : mid1]), int(line[mid1 + 1 : lastpar1]) )
            new_arr = np.zeros(shp) 
            i=1
            
        if counterB == 0 and i==1:
            initial_arr1 = np.zeros((0,shp[0],shp[1]))
            counterB += 1

        if '(:' in line:
            firstpar = line.find('(')
            mid = line.find(',')
            lastpar = line.find(')')
            col = int(line[mid + 1 : lastpar])
            num_list = np.fromstring(line[lastpar + 3:], dtype='float64', sep=',')
            new_arr[:, col] = num_list
            i += 1

        if i>shp[1] and counterB == 1:
            intermed1 = np.concatenate((initial_arr1,new_arr[None,:,:]), axis=0)
            initial_arr1 = intermed1
            i=0
            
    for line in tp_data:
        
        if 'Extents' in line:
            firstpar1 = line.find('(')
            mid1 = line.find(',')
            lastpar1 = line.find(')')
            shp = ( int(line[firstpar1 + 1 : mid1]), int(line[mid1 + 1 : lastpar1]) )
            new_arr = np.zeros(shp) 
            i=1
            
        if counterC == 0 and i==1:
            initial_arr2 = np.zeros((0,shp[0],shp[1]))
            counterC += 1

        if '(:' in line:
            firstpar = line.find('(')
            mid = line.find(',')
            lastpar = line.find(')')
            col = int(line[mid + 1 : lastpar])
            num_list = np.fromstring(line[lastpar + 3:], dtype='float64', sep=',')
            new_arr[:, col] = num_list
            i += 1

        if i>shp[1] and counterC == 1:
            intermed2 = np.concatenate((initial_arr2,new_arr[None,:,:]), axis=0)
            initial_arr2 = intermed2
            i=0
        
    return np.array([initial_arr0, initial_arr1, initial_arr2])

def Embed(num_par):
    E_surfaces = yee.shape[1]
    
    
    global N
    global g_th, g_ph, g_th_ph
    global theta, phi, sintheta_cu, sintheta
    global delta_ph, delta_th
    
    N = num_par
#     jj = 4

    init = np.zeros(3*N)
    init[3] = 1
    init[N+1] = 1
    init[2*N+2] = 1
        
    for j in xrange(E_surfaces):
               
        g_th = yee[0,j]
        g_ph = yee[1,j]
        g_th_ph = yee[2,j]
        
        g_shape = g_th.shape
               
        delta_th = np.pi / g_shape[0]
        delta_ph = 2*np.pi / g_shape[1]

        theta0 = np.empty(g_shape[0])
        phi0 = np.empty(g_shape[1])
        
        for i in xrange(g_shape[0]):
            theta0[i] = (i + 0.5)*delta_th
            
        for k in xrange(g_shape[1]):
            phi0[k] = (k)*delta_ph

        phi, theta = np.meshgrid(phi0,theta0)
        sintheta_cu = np.power(np.sin(theta), 3)
        sintheta = np.sin(theta)
        
        FF = opt.minimize(Main,init,args=(theta,phi),method='TNC',
                jac=Jac,options={'disp': False, 'maxiter':300},callback=callbackF)
        
#             
        init = FF.x
        print 'final surface for time step', j
        thing = FF.x
        x_a = thing[0:N]
        y_a = thing[N:2*N]
        z_a = thing[2*N:3*N]

        Xx = nshe.Ylm(x_a, theta, phi)
        Yy = nshe.Ylm(y_a, theta, phi)
        Zz = nshe.Ylm(z_a, theta, phi)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xx[0], Yy[0], Zz[0])
        # ax.set_title('a = ' + str(a))
        ax.axes.set_aspect('equal')
        ax.set_xlim3d(-2.5,2.5)
        ax.set_ylim3d(-2.5,2.5)
        ax.set_zlim3d(-2.,2.)
        ax.set_zlabel('z')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zticks([])
        ax.set_autoscale_on(False)
        #     plt.zlabel('z')
        ax.view_init(90,90)
        plt.savefig('/home/terrence/Desktop/python.scripts/Honors Code/Figs2/funnn' + str(j),
                    dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
        
        plt.show()