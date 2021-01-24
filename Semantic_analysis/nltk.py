import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import os
import pandas as pd

# uncomment below two lines for first time to download important collections
# import nltk
# nltk.download()


def nltk(output_file, data):
    positive_feedbacks = []
    negative_feedbacks = []
    neutral_feedbacks = []
    for text in data:
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

        # splitting text into words
        tokenized_words = word_tokenize(cleaned_text, "english")

        # removing the stop words from the tokenized word list
        final_words = []
        for word in tokenized_words:
            if word not in stopwords.words('english'):
                final_words.append(word)

        # NLP
        # 1) check if the word in the final word list is also present in emotions.txt
        # 2) if word is present -> add the emotion to emotion_list
        # 3) finally count each emotion in the emotion list

        emotion_list = []
        emotion_file = os.path.join(os.getcwd(), 'emotions.txt')
        with open(emotion_file, 'r') as file:
            for line in file:
                clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
                word, emotion = clear_line.split(':')

                if word in final_words:
                    emotion_list.append(emotion)

        w = Counter(emotion_list)
        sentiment_analyse(cleaned_text, positive_feedbacks, negative_feedbacks, neutral_feedbacks)

        # fig, ax1 = plt.subplots()
        # ax1.bar(w.keys(), w.values())
        # fig.autofmt_xdate()
        # plt.savefig('graph.png')
        # plt.show()
    final = positive_feedbacks + neutral_feedbacks + negative_feedbacks
    df = pd.DataFrame(final, columns=['Text', 'Feedback'])
    df.to_csv(output_file, index=False)


def sentiment_analyse(sentiment_text, positive_feedbacks, negative_feedbacks, neutral_feedbacks):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']

    if neg > pos:
        negative_feedbacks.append([sentiment_text, -1])
    elif pos > neg:
        positive_feedbacks.append([sentiment_text, -1])
    else:
        neutral_feedbacks.append([sentiment_text, -1])
