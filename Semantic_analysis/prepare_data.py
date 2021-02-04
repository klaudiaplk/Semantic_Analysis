import matplotlib.pyplot as plt
import os
import pandas as pd
import snscrape.modules.twitter as sntwitter


def get_input(input_file):
    text_tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('coronavirus vaccination since:2021-01-01 until:2021-02-01').get_items()):
        if i > 10000:
            break
        text_tweets.append(tweet.content)

    df = pd.DataFrame(text_tweets, columns=['Text'])
    df.to_csv(input_file, index=False)

    return text_tweets


def plot(data, method_name, plot_dir):
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
