import pandas as pd
import re
import string
import random

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier


def naive_bayes_classifier(output_file, data):
    """Train the Naive Bayes model and use it to classify the data provided by the data parameter.

    :param output_file: Path to save the program output with semantic analysis.
    :param data: List of sentences for semnatic analysis.
    """

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # For information and the most common positive words, please comment out the section below.
    # all_pos_words = get_all_words(positive_cleaned_tokens_list)
    # freq_dist_pos = FreqDist(all_pos_words)
    # print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_texts_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_texts_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    # division of a dataset into positive and negative tweets in a ratio of 70% training data and 30% test data
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy for Naive Bayes Classifier is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    # classification of ours tweets using a trained model
    negative_feedbacks = []
    positive_feedbacks = []
    for text in data:
        # data cleaning
        clened_text = remove_noise(word_tokenize(text))
        # classification
        if classifier.classify(dict([token, True] for token in clened_text)) == 'Negative':
            negative_feedbacks.append([text, -1])
        elif classifier.classify(dict([token, True] for token in clened_text)) == 'Positive':
            positive_feedbacks.append([text, 1])
    final = positive_feedbacks + negative_feedbacks
    df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    df.to_csv(output_file, index=False)


def remove_noise(text_tokens, stop_words=()):
    """

    :param text_tokens: tweet tokens
    :param stop_words: tuple of stop words
    :return: list of cleaned tokens making up the entire text_tokens
    """

    cleaned_tokens = []

    for token, tag in pos_tag(text_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    """Takes a list of texts as an argument to provide a list of words in all of the texts tokens joined.

    :param cleaned_tokens_list: list of words in all of the texts tokens joined
    """
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_texts_for_model(cleaned_tokens_list):
    """ Preparation of data format for input to the Naive Bayes Classifier.

    :param cleaned_tokens_list: list of words in all of the texts tokens joined
    """
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
