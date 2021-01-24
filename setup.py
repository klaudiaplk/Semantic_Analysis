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
    install_requires=[
        'matplotlib>=3.3.3',
        'textblob>=0.15.3',
        'nltk>=3.5',
        'GetOldTweets3',
        'snscrape',
        'pandas>=1.2.0'
    ]
)
