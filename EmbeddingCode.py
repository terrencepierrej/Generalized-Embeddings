import numpy as np
from multiprocessing import Pool
from scipy import optimize as opt
import SphHarmExpansion as she 
import SphHarmExpansion_new as nshe
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Number of parameters
N = 49

# These next few lines are for implementing the penalty version of the minimization code
# They haven't been as successful because of the convergence difficulties for the gradient calculations
n = 6
EE = 8
ex = EE - 1
# np.int(np.floor(N/2))
La = np.floor(np.sqrt(np.arange(n,N)))
la = np.floor(np.sqrt(np.arange(N)))
la[:n] = 0.


# Delta parameter for distance between points 
delta_th = 0.08
delta_ph = 0.08

theta0 = np.arange(0.001,np.pi + 2*delta_th + 0.001,delta_th)
phi0 = np.arange(0.001,2*np.pi + 2*delta_ph + 0.001,delta_ph)

phi, theta = np.meshgrid(phi0,theta0)
sintheta_sq = np.square(np.sin(theta))

# Initial guess
init = np.zeros(3*N)
init[3] = 1
init[N+1] = 1
init[2*N+2] = 1

# Prolate Spheroid 
# a = 1.
# ee = 5.0
# g_th = (a*a)*(1 + ee*np.square(np.cos(theta))*np.square(np.cos(phi)))
# g_ph = (a*a)*np.square(np.sin(theta))*(1 + ee*np.square(np.sin(phi)))
# g_th_ph = -(1/2)*(a*a)*ee*np.sin(2*theta)*np.sin(2*phi)


# Kerr Metric
a = 1.
M = 1.
r = M + np.sqrt(M*M - a*a)
Sigma = r*r + a*a*np.square(np.cos(theta))
g_th = Sigma
g_ph = (r*r + a*a + (2*M*r*a*a/Sigma)*np.square(np.sin(theta)))*np.square(np.sin(theta))
g_th_ph = 0.

# Peanut Geomtry
# a = 1
# g_th = a*a
# g_ph = a*a*(np.sin(theta)*(1 - 0.75*np.square(np.sin(theta))))**2
# g_th_ph = 0.


def Functions(parameters):
    var, var_theta, var_phi = nshe.Ylm(parameters, theta, phi)
    return var, var_theta, var_phi

# Multiprocessing tool
p = Pool(3)

def Main(params, theta, phi):
    
    x_a = params[0:N]
    y_a = params[N:2*N]
    z_a = params[2*N:3*N]
           
    outputs = p.map(Functions, [x_a, y_a,z_a])
    X, X_theta, X_phi = outputs[0]
    Y, Y_theta, Y_phi = outputs[1]
    Z, Z_theta, Z_phi = outputs[2]

    Gtt_e = X_theta*X_theta + Y_theta*Y_theta + Z_theta*Z_theta
    Gpp_e = X_phi*X_phi + Y_phi*Y_phi + Z_phi*Z_phi
    Gtp_e = X_theta*X_phi + Y_theta*Y_phi + Z_theta*Z_phi


# Penalty version of R

#     R = np.sum((Gtt_e - g_th)*(Gtt_e - g_th)*sintheta_sq + (Gpp_e - g_ph)*(Gpp_e - g_ph)/sintheta_sq + 
#         (Gtp_e - g_th_ph)*(Gtp_e - g_th_ph)*2)*delta_th*delta_ph + np.sum(La*
#                     (x_a[n:]**EE + y_a[n:]**EE + z_a[n:]**EE))
    
    R = np.sum((Gtt_e - g_th)*(Gtt_e - g_th)*sintheta_sq + (Gpp_e - g_ph)*(Gpp_e - g_ph)/sintheta_sq + 
        (Gtp_e - g_th_ph)*(Gtp_e - g_th_ph)*2)*delta_th*delta_ph
    
    
    return R

def Jac(params, theta, phi):
    
    x_a = params[0:N]
    y_a = params[N:2*N]
    z_a = params[2*N:3*N]
    
#     time1 = time.time() 
    
    outputs = p.map(Functions, [x_a, y_a, z_a])
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
    
    G1 = 4*(Gtt_e - g_th)*sintheta_sq
    G2 = 4*(Gpp_e - g_ph)/sintheta_sq
    G3 = 4*(Gtp_e - g_th_ph)
    
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

      # Penalty version of gradient
    
#     grad = np.concatenate((dXa + (EE*x_a**ex)*la, 
#                            dYa + (EE*y_a**ex)*la, dZa + (EE*z_a**ex)*la), axis=0)
    grad = np.concatenate((dXa, dYa, dZa),axis=0)
    

    return grad


# These functions are to spit out the progress of the minimization algorithms 
# The first is for regular minimization
# The second is for the bashinhopping method

Nfeval = 0
def callbackF(Xi):
    global Nfeval
    print '{0:4d}  {1: 3.10f}'.format(Nfeval,Main(Xi,theta,phi))
    Nfeval += 1
    
def callbackNew(x, f, accepted):
    global Nfeval
    print '{0:2d}  {1: 3.10f}  {2: 4d}'.format(Nfeval, f, int(accepted))
    Nfeval += 1

FF = opt.minimize(Main,init,args=(theta,phi),method='TNC', jac=Jac,options={'disp': False, 
        'maxiter':300},callback=callbackF)


# Bashinhopping method

# FF = opt.basinhopping(Main, init, niter=15, minimizer_kwargs={"method" : "TNC",
#         "jac" : Jac, "args" : (theta, phi), "options":{'maxiter':300}, 
#             "callback" : callbackF},callback=callbackNew,
#                       niter_success=3)
