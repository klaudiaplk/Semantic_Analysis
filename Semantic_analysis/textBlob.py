from textblob import TextBlob


def textblobSemanticAnalysis(input_file, output_file):
    """ Semantic analysis using the TextBlob library.

    :param input_file: Text for semantic analysis from a user-specified file.
    :param output_file: Path to save the program output with semantic analysis.
    """
    input = open(input_file, "r")
    feedbacks = []
    for line in input:
        feedbacks.append(line)
    input.close()
    positive_feedbacks = []
    negative_feedbacks = []
    neutral_feedbacks = []
    for feedback in feedbacks:
      feedback_polarity = TextBlob(feedback).sentiment.polarity
      if feedback_polarity > 0.1:
        positive_feedbacks.append(feedback)
        continue
      if feedback_polarity < 0.1 and feedback_polarity > -0.1:
        neutral_feedbacks.append(feedback)
        continue
      negative_feedbacks.append(feedback)

    output = open(output_file, "w")
    for feedback in positive_feedbacks:
        output.write("Positive feedback: " + feedback)
    for feedback in negative_feedbacks:
        output.write("Negative feedback: " + feedback)
    for feedback in neutral_feedbacks:
        output.write("Neutral feedback: " + feedback)
    output.close()
