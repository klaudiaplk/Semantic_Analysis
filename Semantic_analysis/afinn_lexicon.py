import logging

import pandas as pd
from Semantic_analysis.prepare_data import calculate_accuracy
from Semantic_analysis.prepare_data import prepare_test_dataset
from afinn import Afinn


def affin_lexicon(output_file, data):
    """Semantic analysis using the Afinn Lexicon library.

    :param output_file: path to save the program output with semantic analysis
    :param data: list of sentences for semnatic analysis
    :return: dataframe containing texts and labels for each of texts
    """
    test_data, results = prepare_test_dataset()
    final_test = affin_lexicon_evaluation(test_data)
    accuracy = calculate_accuracy(final_test, results)
    logging.info("Afinn Lexicon library accuracy: {}.".format(accuracy))

    final = affin_lexicon_evaluation(data)
    df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    df.to_csv(output_file, index=False)

    return df


def affin_lexicon_evaluation(data):
    """Compute semantic evaluation.

    :param data: list of sentences for semnatic analysis
    :return: dataframe containing texts and labels for each of texts
    """
    af = Afinn()
    sentiment_scores = [af.score(text) for text in data]
    sentiment_results = []
    for i, score in enumerate(sentiment_scores):
        if score > 0:
            sentiment_results.append([data[i], 'Positive'])
            continue
        sentiment_results.append([data[i], 'Negative'])

    return sentiment_results
