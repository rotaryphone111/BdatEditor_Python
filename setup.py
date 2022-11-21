from setuptools import setup, find_packages, Extension

USE_CYTHON = True

try:
    import Cython
except ImportError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("BdatReader", ["Bdat/BdatReader"+ext]) ,Extension("BdatWriter", ["Bdat/BdatWriter"+ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='BdatEditor',
    version='0.0.5a1',
    description='Bdat editor python rewrite',
    author='hydra',
    author_email='knjtkshm@gmail.com',
    license='GNU General Public License v3.0',
    packages=find_packages(),
    ext_modules=extensions,
    entry_points={
        'console_scripts': [
            'bdateditor=BdatEdit:main',
        ]
    },
    install_requires=[
        'numpy>1.20',
        'pandas>1.2.0',
        'PySimpleGuiQt>=0.35.0'
    ]
)
