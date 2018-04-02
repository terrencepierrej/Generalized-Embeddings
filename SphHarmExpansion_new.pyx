import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
# from cython_gsl cimport *

# many numpy functions can be directly accessed by the c compiler, but 
# we have to do a cimport for this.

# Unfortunately, scipy's spherical harmonic function can't be accessed 
# directly as by C code. (It can be called, but then it will be evaluated 
# by python. It can't be compiled directly into our code.) So instead, 
# we use functions from the gnu scientific library, "gsl." They don't 
# have a function for the Ylm's directly, but they do have a function 
# for the associated Legendre function, normalized in such a way that 
# we'd just need to multiply them by cos(m*phi) or sin(m*phi) for Ylm.


# Import enums -----> look up


cdef extern from "gsl/gsl_sf_legendre.h":


     ctypedef enum gsl_sf_legendre_t:
 
       GSL_SF_LEGENDRE_SCHMIDT,
       GSL_SF_LEGENDRE_SPHARM,
       GSL_SF_LEGENDRE_FULL,
       GSL_SF_LEGENDRE_NONE

     size_t gsl_sf_legendre_array_index(const size_t l, const size_t m)
     
     int gsl_sf_legendre_deriv_alt_array_e(const gsl_sf_legendre_t norm, const size_t lmax, double x, const double csphase,
      double result_array[], double result_deriv_array[])

     size_t gsl_sf_legendre_array_n(const size_t lmax)

     double gsl_sf_legendre_sphPlm(const int l, const int m, const double x)


cdef extern from "math.h":
     double cos(double x)
     double sin(double x)
     double sqrt(double x)
     int floor(double x)

# Here is the function that does the spherical harmonic sum. It's basically 
# just like a python function, except that items have to be declared and 
# typed as in C.

cpdef np.ndarray[double, ndim=3] Ylm(np.ndarray[double, ndim=1] coeffs, np.ndarray[double, ndim=2] theta,
  np.ndarray[double, ndim=2] phi):

 cdef int ncoeffs = len(coeffs)
 cdef int nth, nph
 nth, nph = np.shape(theta)
 cdef size_t size = gsl_sf_legendre_array_n(ncoeffs)
 cdef double *results = <double *>malloc(size * sizeof(double))
 cdef double *results_deriv = <double *>malloc(size * sizeof(double))
 cdef np.ndarray[double, ndim=2] outval = np.zeros((nth, nph))
 cdef np.ndarray[double, ndim=2] outval_deriv_theta = np.zeros((nth, nph))
 cdef np.ndarray[double, ndim=2] outval_deriv_phi = np.zeros((nth, nph))
 cdef int l, m, a
 cdef double costheta
 cdef double costerm
 cdef double sinterm
 cdef size_t index
 # cdef gsl_sf_legendre_t norm = GSL_SF_LEGENDRE_SPHARM
 
 for i in range(nth):

   costheta = cos(theta[i,0]) # there are some issues with not allowing 1 and -1 for the double x arg

   gsl_sf_legendre_deriv_alt_array_e(GSL_SF_LEGENDRE_SPHARM, ncoeffs, costheta, -1, results, results_deriv)

   for a in range(ncoeffs):

     l = floor(sqrt(a))
     m = a - l**2 - l

     if m>=0:

       index = gsl_sf_legendre_array_index(l,m)

       for j in range(nph):

         costerm = cos(m*phi[i,j])

         outval[i,j] += coeffs[a]*results[index]*costerm
         outval_deriv_theta[i,j] += coeffs[a]*results_deriv[index]*costerm
         outval_deriv_phi[i,j] += coeffs[a]*results[index]*-m*sin(m*phi[i,j])


     if m<0:

       index = gsl_sf_legendre_array_index(l,-m)

       for j in range(nph):

         sinterm = sin(m*phi[i,j])

         outval[i,j] += coeffs[a]*results[index]*sinterm
         outval_deriv_theta[i,j] += coeffs[a]*results_deriv[index]*sinterm
         outval_deriv_phi[i,j] += coeffs[a]*results[index]*m*cos(m*phi[i,j])


 free(results)
 free(results_deriv)

 return np.stack((outval, outval_deriv_theta, outval_deriv_phi), axis=0)


# =================================================================================================================================
# =================================================================================================================================

cpdef np.ndarray[double, ndim=3] D_Y(int N, np.ndarray[double, ndim=2] theta, np.ndarray[double, ndim=2] phi):

 cdef int nth, nph
 nth, nph = np.shape(theta)
 cdef size_t size = gsl_sf_legendre_array_n(N)
 cdef double *results = <double *>malloc(size * sizeof(double))
 cdef double *results_deriv = <double *>malloc(size * sizeof(double))
 cdef np.ndarray[double, ndim=2] outval_theta = np.zeros((nth, nph))
 cdef np.ndarray[double, ndim=3] d_theta = np.zeros((N, nth, nph))
 cdef np.ndarray[double, ndim=2] outval_phi = np.zeros((nth, nph))
 cdef np.ndarray[double, ndim=3] d_phi = np.zeros((N, nth, nph))
 cdef double costheta
 cdef double costerm
 cdef double sinterm
 cdef size_t index
 # cdef gsl_sf_legendre_t norm = GSL_SF_LEGENDRE_SPHARM

 for i in range(nth):

   costheta = cos(theta[i,0]) # there are some issues with not allowing 1 and -1 for the double x arg

   gsl_sf_legendre_deriv_alt_array_e(GSL_SF_LEGENDRE_SPHARM, N, costheta, -1, results, results_deriv)

   for a in range(N):

     l = floor(sqrt(a))
     m = a - l**2 - l

     if m>=0:

       index = gsl_sf_legendre_array_index(l,m)

       for j in range(nph):

         costerm = cos(m*phi[i,j])
         
         outval_theta[i,j] = results_deriv[index]*costerm

         outval_phi[i,j] = results[index]*-m*sin(m*phi[i,j])


     if m<0:

       index = gsl_sf_legendre_array_index(l,-m)

       for j in range(nph):

         sinterm = sin(m*phi[i,j])

         outval_theta[i,j] = results_deriv[index]*sinterm

         outval_phi[i,j] = results[index]*m*cos(m*phi[i,j])

     d_phi[a] += outval_phi
     outval_phi = np.zeros((nth,nph))
     d_theta[a] += outval_theta
     outval_theta = np.zeros((nth,nph))



 free(results)
 free(results_deriv)

 return np.stack((d_theta, d_phi), axis=0)