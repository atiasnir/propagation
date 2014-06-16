from distutils.core import setup

setup(
    name='propagate',
    version='0.1.0',
    author='Nir Atias',
    packages=['propagate'], 
    url='http://pypi.python.org/pypi/propagate',
    license='LICENSE.txt', 
    description='Propagation belief graph algorithm', 
    long_description=open('README.txt').read(), 
    install_requires=[
        "numpy >= 1.8.1", 
        "scipy >= 0.14.0", 
        "pandas >= 0.13.0",
    ],
)

