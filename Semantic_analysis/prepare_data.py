import logging
import os
import re
import string

import matplotlib.pyplot as plt
import pandas as pd
import snscrape.modules.twitter as sntwitter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"


def get_input(tweets_file):
    """Receiving list of 10,000 tweets with the word "coronavirus vaccination"
    in the period from January 1, 2021 to February 1, 2021.

    :param tweets_file: path to save tweets for semantic analysis
    :return: list of downloaded tweets from twitter
    """
    text_tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('coronavirus vaccination since:2021-01-01 until:2021-02-01').get_items()):
        if i > 10000:
            break
        text_tweets.append(tweet.content)

    df = pd.DataFrame(text_tweets, columns=['Text'])
    df.to_csv(tweets_file, index=False)

    return text_tweets


def plot(data, method_name, plot_dir):
    """Drawing a plot with the results of semantic analysis of tweets.

    :param data: analyzed tweets along with labels
    :param method_name: name of the method that performed the semantic analysis of the tweets
    :param plot_dir: path to save the output plot for semantic analysis
    """
    cnt_positive = 0
    cnt_negative = 0

    for index, row in data.iterrows():
        if row['Feedback'] == 'Positive':
            cnt_positive = cnt_positive + 1
        elif row['Feedback'] == 'Negative':
            cnt_negative = cnt_negative + 1

    results = [["Positive", "Negative"], [cnt_positive, cnt_negative]]
    plt.bar(results[0], results[1])
    plt.title("Semantic analysis results for coronavirus vaccination tweets")
    plt.ylabel("Number of tweets")
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(method_name)))
    plt.close()


def prepare_test_dataset():
    """Prepare only a test dataset.

    :return: list of test tweets and a list of categories assigned to them
    """
    cur_path = os.path.dirname(__file__)
    dataset_filename = os.listdir(os.path.join(cur_path, '../input'))[0]
    dataset_path = os.path.join(cur_path, "..", "input", dataset_filename)
    df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    decode_map = {0: "NEGATIVE", 4: "POSITIVE"}

    df.target = df.target.apply(lambda x: decode_sentiment(x, decode_map))

    positive_tweets = []
    negative_tweets = []

    for index, row in df.iterrows():
        if row['target'] == 'POSITIVE':
            positive_tweets.append(df['text'][index])
        elif row['target'] == 'NEGATIVE':
            negative_tweets.append(df['text'][index])

    positive_dataset_len = len(positive_tweets)
    negative_dataset_len = len(negative_tweets)
    twenty_percent_positive = round(positive_dataset_len * 0.2)
    twenty_percent_negative = round(negative_dataset_len * 0.2)

    # create test data
    positive_test_data = positive_tweets[:twenty_percent_positive]
    negative_test_data = negative_tweets[:twenty_percent_negative]

    stop_words = stopwords.words('english')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_test_data:
        logging.info("Remove noise for positive tweet: {}".format(tokens))
        positive_cleaned_tokens_list.append(remove_noise(word_tokenize(tokens), stop_words))

    for tokens in negative_test_data:
        logging.info("Remove noise for negative tweet: {}".format(tokens))
        negative_cleaned_tokens_list.append(remove_noise(word_tokenize(tokens), stop_words))

    positive_cleaned_tokens_list = [' '.join(positive_cleaned_tokens) for positive_cleaned_tokens in
                                    positive_cleaned_tokens_list]
    negative_cleaned_tokens_list = [' '.join(negative_cleaned_tokens) for negative_cleaned_tokens in
                                    negative_cleaned_tokens_list]

    test_data = positive_cleaned_tokens_list + negative_cleaned_tokens_list
    results = ["POSITIVE"] * twenty_percent_positive + ["NEGATIVE"] * twenty_percent_negative

    return test_data, results


def decode_sentiment(label, decode_map):
    """Decoding numerical semantic names into nominal names.
    """
    return decode_map[int(label)]


def prepare_cleaned_dataset():
    """Preparation of a clean dataset.

    :return: list of cleaned positive tweets and list of cleaned negative tweets
    """
    cur_path = os.path.dirname(__file__)
    dataset_filename = os.listdir(os.path.join(cur_path, '../input'))[0]
    dataset_path = os.path.join(cur_path, "..", "input", dataset_filename)
    df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    decode_map = {0: "NEGATIVE", 4: "POSITIVE"}

    df.target = df.target.apply(lambda x: decode_sentiment(x, decode_map))

    positive_tweets = []
    negative_tweets = []

    for index, row in df.iterrows():
        if row['target'] == 'POSITIVE':
            positive_tweets.append(df['text'][index])
        elif row['target'] == 'NEGATIVE':
            negative_tweets.append(df['text'][index])

    stop_words = stopwords.words('english')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweets:
        logging.info("Remove noise for positive tweet: {}".format(tokens))
        positive_cleaned_tokens_list.append(remove_noise(word_tokenize(tokens), stop_words))

    for tokens in negative_tweets:
        logging.info("Remove noise for negative tweet: {}".format(tokens))
        negative_cleaned_tokens_list.append(remove_noise(word_tokenize(tokens), stop_words))

    return positive_cleaned_tokens_list, negative_cleaned_tokens_list


def remove_noise(text_tokens, stop_words=()):
    """Removing noise from text.

    :param text_tokens: tweet tokens
    :param stop_words: tuple of stop words
    :return: list of cleaned tokens making up the entire text_tokens
    """

    cleaned_tokens = []

    for token, tag in pos_tag(text_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        token = token.replace("'", '')

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


def calculate_accuracy(final_test, results):
    """Compute accuracy.

    :param final_test: list of data evaluation by the selected method
    :param results: list of original values assigned to the data
    :return: value of accuracy
    """
    correct_evaluate = 0
    for i, final_result in enumerate(final_test):
        if final_result[1].lower() == results[i].lower():
            correct_evaluate = correct_evaluate + 1

    all_results = len(results)
    accuracy = correct_evaluate / all_results

    return accuracy
