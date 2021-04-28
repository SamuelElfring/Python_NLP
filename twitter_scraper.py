# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:08:24 2020

@author: saelf
"""

import tweepy
import re
#for encoding to the terminal window


consumer_key = 'srQEXS4PpnLMElk9Y1RNNjeOm'
consumer_secret = '8BuaAJK9U1C8rWaNo6141LODNtAB0y42Qwwc5HBBCy9vnp4gpP'
access_token = '1346110123295600640-g48wSUL4tPPB7CPzpHXT3YUTsromD7'
access_token_secret = 'y5BTKQpCysqdAVB1NcffnjPrk4zzEW575aUYA71NSdyRn'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

num_tweets = 50

tweetList = []

f1 = open("tweet_quotes.txt", "w")
f2 = open("tweets.txt", "w")

"""TBC: Use append and collect tweet date info for tracking sentiment ovet time"""
#f = open("tweets.txt", "a") append, or create if exists

f1.write("================\n")
f2.write("================\n")

keyword = "Biden"
for tweet in api.search(q=keyword, lang="en", count=num_tweets, tweet_mode='extended',include_entities = False):

    try:
       tweetList.append(tweet.retweeted_status.full_text)

    except AttributeError:  # Not a Retweet
       tweetList.append(tweet.full_text)

#Write tweetList to file and clean format

tweetQuotes = []
clean_tweets = []
import io
with io.open("tweets_quotes.txt", "w", encoding="utf-8") as f1:
    with io.open("tweets.txt", "w", encoding="utf-8") as f2:
        for tweet_item in tweetList:
            clean_tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', tweet_item)
            clean_tweet = re.sub("(@[A-Za-z0-9_]+)","", tweet_item)
            clean_tweets.append(clean_tweet)

        clean_tweet_quotes = list(map( lambda s : s  if s.find(" ") == -1 else '"' + s + '",', clean_tweets))
        #tweetQuotes = map( lambda s : s  if s.find(" ") == -1 else '"' + s + '"', tweetList)

        f1.write('\n'.join([ str(myelement) for myelement in clean_tweet_quotes ]))
        f2.write('\n'.join([ str(myelement) for myelement in clean_tweets ]))

f1.close()
f2.close()

print((clean_tweets))
