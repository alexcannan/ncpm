from setuptools import setup, find_packages

setup(
    name='ncpm',
    author='Alex Cannan, Ben Cannan',
    author_email='alexfcannan@gmail.com',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'Pillow',
    ],
)