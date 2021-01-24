import pandas as pd
from textblob import TextBlob


def textblob_semantic_analysis(output_file, data):
    """ Semantic analysis using the TextBlob library.

    :param output_file: Path to save the program output with semantic analysis.
    :param data: List of sentences for semnatic analysis.
    """
    positive_feedbacks = []
    negative_feedbacks = []
    neutral_feedbacks = []
    for feedback in data:
      feedback_polarity = TextBlob(feedback).sentiment.polarity
      if feedback_polarity > 0.1:
        positive_feedbacks.append([feedback, 1])
        continue
      if feedback_polarity < 0.1 and feedback_polarity > -0.1:
        neutral_feedbacks.append([feedback, 0])
        continue
      negative_feedbacks.append([feedback, -1])

    final = positive_feedbacks + neutral_feedbacks + negative_feedbacks
    df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    df.to_csv(output_file, index=False)
