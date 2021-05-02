# from setuptools import setup
# from Cython.Build import cythonize
# # python setup.py build_ext --inplace

# setup(
#     name='test',
#     ext_modules=cythonize('./entity/DBEngine.py'),
# )

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('myfile',
                         ['myfile.pyx'],
                        )]
setup(
name = 'myfile',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)

# from setuptools import setup, find_packages
 
# setup(
#     name='DBEngine',
#     version='1.0.0',
#     # packages=find_packages(include=['greet_pkg', 'greet_pkg.*']),
#     url='',
#     license='psc',
#     author='vanessafan',
#     author_email='vanessafan@powerchip.com',
#     description='DB Engine',
#     py_modules=['greet2'],
#     install_requires=['pyjokes']
# )