# from setuptools import setup
# from Cython.Build import cythonize #conda install -c anaconda cython
# # python setup.py build_ext --inplace
##python setup.py build_ext --inplace --build-lib ./lib

# setup(
#     name='test',
#     ext_modules=cythonize('./entity/DBEngine.py'),
# )

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# ext_modules = [Extension('myfile',
#                          ['myfile.pyx'],
#                         )]
# setup(
# name = 'myfile',
# cmdclass = {'build_ext': build_ext},
# ext_modules = ext_modules
# )



import Cython.Build
import distutils.core

def py2c(file):
    cpy = Cython.Build.cythonize(file) # 返回distutils.extension.Extension对象列表

    distutils.core.setup(
	    name = 'DBEngine', # 包名称
	    version = "1.0.0",    # 包版本号
	    ext_modules= cpy,     # 扩展模块
	    author = "vanessa",#作者
	    author_email='vanessafan@powerchip.com'#作者email
	)

if __name__ == '__main__':
    # file = "./entity/DBEngine.py"
	file = "./Util/Encrypt.py"
	py2c(file)

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