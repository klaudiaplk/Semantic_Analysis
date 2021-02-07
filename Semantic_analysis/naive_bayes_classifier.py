import random

import pandas as pd
from Semantic_analysis.prepare_data import prepare_cleaned_dataset, remove_noise
from nltk import classify, FreqDist, NaiveBayesClassifier
from nltk.tokenize import word_tokenize

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"


def naive_bayes_classifier(output_file, data):
    """Train the Naive Bayes model and use it to classify the data provided by the data parameter.

    :param output_file: Path to save the program output with semantic analysis.
    :param data: List of sentences for semnatic analysis.
    """

    positive_cleaned_tokens_list, negative_cleaned_tokens_list = prepare_cleaned_dataset()

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

    dataset_len = len(dataset)
    eighty_percent = round(dataset_len * 0.8)

    # division of a dataset into positive and negative tweets in a ratio of 70% training data and 30% test data
    train_data = dataset[:eighty_percent]
    test_data = dataset[eighty_percent:]

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
            negative_feedbacks.append([text, 'Negative'])
        elif classifier.classify(dict([token, True] for token in clened_text)) == 'Positive':
            positive_feedbacks.append([text, 'Positive'])
    final = positive_feedbacks + negative_feedbacks
    df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    df.to_csv(output_file, index=False)

    return df


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
