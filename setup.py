from setuptools import setup, find_packages

setup(
    name='BdatEditor',
    version='0.0.0',
    description='Bdat editor python rewrite',
    author='hydra',
    author_email='knjtkshm@gmail.com',
    license='GNU General Public License v3.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'bdateditor=run:main',
        ]
    },
    install_requires=[
        'numpy>1.20',
        'pandas>1.2.0',
        'PySimpleGuiQt>=0.35.0'
    ]
)
