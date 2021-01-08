from setuptools import setup, find_packages

setup(
    name='semantic_analysis',
    version=1.0,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'semantic_analysis = Semantic_analysis.main:main'
        ]
    },
)
