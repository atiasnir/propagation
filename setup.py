from setuptools import setup, find_packages

setup(
    name='propagate',
    version='0.2.1',
    author='Nir Atias',
    author_email='rinatias@gmail.com',
    packages=find_packages(),
    url='https://github.com/atiasnir/propagation',
    license='LICENSE.txt', 
    description='Propagation belief graph algorithm', 
    long_description=open('README.txt').read(), 
    install_requires=[
        "numpy >= 1.8.1", 
        "scipy >= 0.14.0", 
        "pandas >= 0.13.0",
        "scikit-learn >= 0.14.1",
    ],
    package_data={'test.data' : ['*.txt']},
    keywords="belief propagation diffusion graph algorithm",
)

