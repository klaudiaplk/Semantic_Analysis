import pandas as pd
from textblob import TextBlob


def textblob_semantic_analysis(output_file, data):
    """ Semantic analysis using the TextBlob library.

    :param output_file: Path to save the program output with semantic analysis.
    :param data: List of sentences for semnatic analysis.
    """
    positive_feedbacks = []
    negative_feedbacks = []
    for feedback in data:
      feedback_polarity = TextBlob(feedback).sentiment.polarity
      if feedback_polarity > 0:
        positive_feedbacks.append([feedback, 'Positive'])
        continue
      negative_feedbacks.append([feedback, 'Negative'])

    final = positive_feedbacks + negative_feedbacks
    df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    df.to_csv(output_file, index=False)

    return df
