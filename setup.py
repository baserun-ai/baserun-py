from setuptools import setup, find_packages

setup(
    name="baserun",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'requests>=2.31.0',
    ],
    entry_points={
        'pytest11': ['baserun = baserun.pytest_plugin'],
        'console_scripts': [
            'baserun = baserun.cli:main',
        ],
    },
)
