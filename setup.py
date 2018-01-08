from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = ['google-cloud-bigquery==0.28.0', 'google-cloud-storage==1.6.0']

if __name__ == '__main__':
    setup(
        name='trainer',
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages()
    )
