import snscrape.modules.twitter as sntwitter


def get_input():
    text_tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('lockdown since:2015-09-30 until:2019-08-01').get_items()):
        if i > 20:
            break
        text_tweets.append(tweet.content)

    return text_tweets
