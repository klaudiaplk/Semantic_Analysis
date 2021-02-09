import logging

import pandas as pd
from Semantic_analysis.prepare_data import calculate_accuracy
from Semantic_analysis.prepare_data import prepare_test_dataset
from textblob import TextBlob


def textblob_semantic_analysis(output_file, data):
    """ Semantic analysis using the TextBlob library.

    :param output_file: Path to save the program output with semantic analysis.
    :param data: List of sentences for semnatic analysis.
    """

    test_data, results = prepare_test_dataset()
    final_test = textblob_evaluation(test_data)

    accuracy = calculate_accuracy(final_test, results)
    logging.info("TextBlob library accuracy: {}.".format(accuracy))

    final = textblob_evaluation(data)
    df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    df.to_csv(output_file, index=False)

    return df


def textblob_evaluation(data):
    """Compute semantic evaluation.

    :param data: list of sentences for semnatic analysis
    :return: dataframe containing texts and labels for each of texts
    """
    sentiment_results = []
    for feedback in data:
        feedback_polarity = TextBlob(feedback).sentiment.polarity
        if feedback_polarity > 0:
            sentiment_results.append([feedback, 'Positive'])
            continue
        sentiment_results.append([feedback, 'Negative'])

    return sentiment_results
