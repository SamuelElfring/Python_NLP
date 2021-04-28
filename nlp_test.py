# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:58:33 2021

@author: saelf
using
https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
"""



"""1. Install NTLK and sample data"""

# This will import three datasets from NLTK that contain various tweets to train and test the model:

# negative_tweets.json: 5000 tweets with negative sentiments
# positive_tweets.json: 5000 tweets with positive sentiments
# tweets.20150430-223406.json: 20000 tweets with no sentiments
# Next, create variables for positive_tweets, negative_tweets, and text:

# The strings() method of twitter_samples will print all of the tweets within a dataset as strings.
# Setting the different tweet collections as a variable will make processing and testing easier.

# Before using a tokenizer in NLTK, you need to download an additional resource, punkt.
# The punkt module is a pre-trained model that helps you tokenize words and sentences.
# For instance, this model knows that a name may contain a period (like “S. Daityari”) and the presence of this period in a sentence does not necessarily end it.
# First, start a Python interactive session: (using Spyder == interactive session - alt would be cmd type "Python3")




# """Once the download is complete, you are ready to use NLTK’s tokenizers.
# NLTK provides a default tokenizer for tweets with the .tokenized() method.
# Add a line to create an object that tokenizes the positive_tweets.json dataset:"""

"""2. Tokenizing the Data"""

from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0] # test tokenized method on a single tweet form pos twets json dataset

for tweet in tweet_tokens:
    print(tweet)

#print("tweet token of item zero is: ",tweet_tokens[0])


# Before running a lemmatizer, you need to determine the context for each word in your text.
# This is achieved by a tagging algorithm, which assesses the relative position of a word in a sentence.
# In a Python session, Import the pos_tag function, and provide a list of tokens as an argument to get the tags.
# Let us try this out in Python:


"""3. Normalize the Data"""

import nltk #removes illegal characters
nltk.download('punkt')
# """Packages for normalisation: wordnet is a lexical database for the English language that helps the script determine the base word.
# You need the averaged_perceptron_tagger resource to determine the context of a word in a sentence."""
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


from nltk.tag import pos_tag
from nltk.corpus import twitter_samples

tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
print("positive tags for tokenized tweet zero are: ", pos_tag(tweet_tokens[0]))

# examle tags (tags illustrate context):
# NNP: Noun, proper, singular
# NN: Noun, common, singular or mass
# IN: Preposition or conjunction, subordinating
# VBG: Verb, gerund or present participle
# VBN: Verb, past participle

# Full normalisation of a sentence is:
# 1. generate tags for each token in the text
# 2. lemmatize each word using the tag

from nltk.stem.wordnet import WordNetLemmatizer



# def lemmatize_sentence(tokens):

# # gets the position tag of each token of a tweet.
# # if the tag starts with NN, the token is assigned as a noun.
# # if the tag starts with VB, the token is assigned as a verb.

#     lemmatizer = WordNetLemmatizer()
#     lemmatized_sentence = []
#     for word, tag in pos_tag(tokens):
#         if tag.startswith('NN'): #proper noun, singluar
#             pos = 'n'
#         elif tag.startswith('VB'): #Verbs of all tenses
#             pos = 'v'
#         else:
#             pos = 'a'
#         lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
#     return lemmatized_sentence

# print("lemmatized sentence of tweet zero is: ", lemmatize_sentence(tweet_tokens[0]))
# #e.g. in lemmatization, "being" is exchanged for "be"

"""4. Remove noise from data"""

import re, string

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

nltk.download('stopwords')


from nltk.corpus import stopwords #stop words are words like "a, the , it, which add no meaning"
stop_words = stopwords.words('english')

print("lemmatized and cleaned tweet zero is: ", remove_noise(tweet_tokens[0], stop_words))

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#print(remove_noise(tweet_tokens[0], stop_words))

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


"""Compare original tokens of tweet 500 to cleaned token of tweet 500"""

print("pos tokens of tweet 500:", positive_tweet_tokens[500])
print("cleaned pos tokens of tweet 500: ", positive_cleaned_tokens_list[500])

"""5. Determining Word Density"""

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list) #finds out which unique words exist in sample of tweets

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words) #finds which words is the most frequent

mostfrequenttweets = 10

# print(mostfrequenttweets, "most common positive words in tweet selection is", freq_dist_pos.most_common(mostfrequenttweets)
#       )


"""6: Preparing Data for the Model"""
#supervised machine learning model which uses postive and negative sentiments
#need to create a training set to associate each dataset with a "sentiment" for training
#only + and - categories are used here but more could be used
#split model into two parts:
# 6.1. build model: prepare data for sentiment analysis by converting tokens -> dictionary form,
# then split data for training and testing
# 6.2. test performance of model


"""6.1: Convert tokens to a Dictionary"""
#use the Naive Bayes classifier in NLTK to perform the modeling exercise
#model requires not just a list of words in a tweet, but a Python dictionary with words as keys and True as values


#makes a generator function to change the format of the cleaned data.

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


"""6.2: Split Dataset for training and test the model"""
import random
positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train_data = dataset[:7000] #first 7000 tweets
test_data = dataset[7000:] #tweets 7000-10,000


"""7: Build and Test the model"""
# use the NaiveBayesClassifier class to build the model
# Use the .train() method to train the model
# use  .accuracy() method to test the model on the testing data.

...
from nltk import classify
from nltk import NaiveBayesClassifier #simle Bayesian network, highly scalable and when combine with kernel density est can be accurate
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))
#Accuracy is defined as the percentage of tweets in the testing dataset for which the model was correctly able to predict the sentiment

print(classifier.show_most_informative_features(10))


"""Test using random tweets from twitter"""
from nltk.tokenize import word_tokenize

custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
#custom_tweet = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))
