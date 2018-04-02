
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([Extension("SphHarmExpansion_new", ["SphHarmExpansion_new.pyx"], libraries=["gsl","gslcblas"])]), 
    include_dirs=[numpy.get_include()]
)

# LDFLAGS="-L/home/tpierrej/Honors/gsl2p4/lib/"
# CFLAGS="-I/home/tpierrej/Honors/gsl2p4/include/"

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules = cythonize([Extension("SphHarmExpansion_new", ["SphHarmExpansion_new.pyx"], libraries=["gsl","gslcblas"],
#     include_dirs=["/home/tpierrej/Honors/gsl2p4/include/gsl"], library_dirs=["/home/tpierrej/Honors/gsl2p4/lib"])]), 
#     include_dirs=[numpy.get_include()]
# )