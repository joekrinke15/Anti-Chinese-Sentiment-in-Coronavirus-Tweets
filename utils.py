"""Various functions/utilities used to analalyze coronavirus tweets."""

import pandas as pd
from twarc import Twarc
import re

LINKS_re = re.compile(r'https?://.+?(\s|$)')
NONALPHANUMERIC_re = re.compile(r'[^\w ]')

def read_data():
    data = pd.read_csv('full_dataset-clean.tsv.gz', sep='\t', compression='gzip')
    return(data)

def hydrate_tweets(data, consumer_key, consumer_secret, access_token, access_token_secret):
    t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
    tweet_text = []
    favorite_count =[]
    retweet_count = []
    
    for tweet in t.hydrate(data['tweet_id']):
        tweet_text.append(tweet['full_text'])
        favorite_count.append(tweet['favorite_count'])
        retweet_count.append(tweet['retweet_count'])
        
    data['tweet_text'] = tweet_text
    data['favorite_count'] = favorite_count
    data['retweet_count'] = retweet_count
    
    data.to_csv("HydratedTweets")
    return(data)


def clean_text(comment):
    cleaned = re.sub(LINKS_re, ' ', comment)
    cleaned = re.sub(NONALPHANUMERIC_re, ' ', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)  # collapse consequtive spaces
    return cleaned