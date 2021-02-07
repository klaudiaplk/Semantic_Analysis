import logging
import pandas as pd
from textblob import TextBlob


def textblob_semantic_analysis(test_data, results, output_file, data):
    """ Semantic analysis using the TextBlob library.

    :param output_file: Path to save the program output with semantic analysis.
    :param data: List of sentences for semnatic analysis.
    """

    final_test = textblob_evaluation(test_data)

    accuracy = calculate_accuracy(final_test, results)
    logging.info("TextBlob library accuracy: {}.".format(accuracy))

    # final = textblob_evaluation(data)
    # df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    # df.to_csv(output_file, index=False)
    df = []

    return df


def textblob_evaluation(data):
    positive_feedbacks = []
    negative_feedbacks = []
    for feedback in data:
        feedback_polarity = TextBlob(feedback).sentiment.polarity
        if feedback_polarity > 0:
            positive_feedbacks.append([feedback, 'Positive'])
            continue
        negative_feedbacks.append([feedback, 'Negative'])

    return positive_feedbacks + negative_feedbacks


def calculate_accuracy(final_test, results):
    correct_evaluate = 0
    for i, final_result in enumerate(final_test):
        if final_result[1].lower() == results[i].lower():
            correct_evaluate = correct_evaluate + 1

    all_results = len(results)
    accuracy = correct_evaluate / all_results

    return accuracy
