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
        'snscrape',
        'pandas>=1.2.0',
        'scikit-learn>=0.24',
        'Keras>=2.4.3',
        'gensim>=3.8.3',
        'tensorflow<2.4.0,>=2.3.0',
        'h5py<2.11.0,>=2.10.0',
        'notebook',
        'afinn'
    ]
)
