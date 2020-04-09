"""Various functions/utilities used to analalyze coronavirus tweets."""

import pandas as pd
from twarc import Twarc
import numpy as np

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

#Remove all data before March. Generate a random sample of 10% of the tweets from each day. 
def time_sample(data):
    data['date'] = pd.to_datetime(data['date'])
    value_to_check = pd.Timestamp(2020,3,1)
    filter_mask = data['date'] > value_to_check
    filtered_df = data[filter_mask]
    random_dataframe = pd.DataFrame(columns = ['tweet_id', 'date', 'time'])
    for i in filtered_df['date'].unique():
        date_subset = filtered_df[filtered_df['date']==i]
        random_indices = (np.random.randint(low = 0, high = len(date_subset)-1, size = round(len(date_subset)*.05)))
        random_subset = date_subset.iloc[random_indices,:]
        random_dataframe = random_dataframe.append(random_subset)
        return(random_dataframe)