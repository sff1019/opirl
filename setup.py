from setuptools import setup, find_packages

setup(
    name='opirl',
    packages=[
        package for package in find_packages() if package.startswith('src')
    ],
    version='0.1.0',
)
