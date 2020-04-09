"""Various functions/utilities used to analalyze coronavirus tweets."""

from random import sample
import pandas as pd
from twarc import Twarc
import numpy as np
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
#Remove all data before March.
def date_filter(data):
    data['date'] = pd.to_datetime(data['date'])
    value_to_check = pd.Timestamp(2020,3,1)
    filter_mask = data['date'] > value_to_check
    filtered_df = data[filter_mask]
    return(filtered_df)
    
#Randomly sample 1% of tweets and create txt file of ids.
def write_txt_file(data): 
    tweet_ids = data['tweet_id']
    sample_ids = sample(tweet_ids, int(len(tweet_ids)/100))
    with open('tweet_ids.txt', 'w') as f:
    for item in sample_ids:
        f.write("%s\n" % item)
