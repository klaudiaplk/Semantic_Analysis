# Semantic Analysis
Discovering emotional states from Tweets using Afinn Lexicon dictionary method, Naive Bayessian Classifier and Neural Network.


# Programming language
Semantic analysis program was tested only in Python 3.7 on Windows.

# Libraries
The list of used libraries with their versions can be found in the setup.py file.

To download tweets from twitter:
```
snscrape
```
To prepare the input data:
```
pandas,
nltk,
gensim
```
For drawing output plots:
```
matplotlib
```
To run Afinn Lexicon dictionary method, Naive Bayessian Classifier and Neural Network:
```
afinn,
textblob,
scikit-learn,
Keras
tensorflow,
notebook
```

# Virtual environment
It is recommended to create virtual environment to install the appropriate python versions and libraries used in the program and described in the setup.py file. An example of creating a virtual environment:
```
path/to/python -m venv /path/to/new/virtual/environment
```

# Installation
```
git clone https://github.com/klaudiaplk/Semantic_Analysis.git
cd Semantic_Analysis
pip install -e .
```

# Usage
We can run Afinn Lexicon dictionary method and Naive Bayessian Classifier using the command:
```
semantic_analysis [--library {textblob,bayes,lexicon}] --tweets-file TWEETS_FILE --output-file OUTPUT_FILE --plot-dir PLOT_DIR
```
Parameters that are required to run the script:
```
--library -> choices=['textblob', 'bayes', 'lexicon'], Possibility to choose method.
--tweets-file -> type=str, Path to file to save tweets for semantic analysis.
--output-file -> type=str, Path to file to save the program output with semantic analysis.
--plot-dir -> type=str, Path to directory to save the output plot for semantic analysis.
```
All application launch parameters are also available under the command:
```
semantic_analysis -h
```
We can start the neural network by running jupyter notebook with the command:
```
jupyter notebook
```
However, remember to connect our virtual environment as the kernel, which we created earlier.We can run the next stages of our neural network directly in the notebook jupyter.

# Additional information
In addition to python files, the folder also includes plots with results for individual methods with the png extension. These are results for tweets in the input folder (tweets.csv). There are 10,000 tweets with the word "coronavirus vaccination" in the period from January 1, 2021 to February 1, 2021.
