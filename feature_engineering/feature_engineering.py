# Import required libraries

import pandas as pd
import numpy as np
import pickle
import time
import nltk
# a tokenizer will be used for pre-processing the review strings for feature extraction
from nltk import tokenize
nltk.download('punkt')
# Vader and its lexicon will be used for the extraction of a sentiment analysis-based feature
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')
# spaCy will be used for tokenising (into words) reviews and tagging parts of speech within them
import spacy
# from scikit-learn import the TF-IDF vectoriser to be used for the n-grams feature
from sklearn.feature_extraction.text import TfidfVectorizer
# stats package will be used for feature selection
from scipy import stats
# pyplot will be used for creating boxplots
import matplotlib.pyplot as plt

# Load dataframe to be used for feature engineering

# load dataframe with no features (output by data preparation code)
### SET OWN PATH HERE
### 
path = "/Users/artembutbaev/OneDrive/University of Bath 20-21 " + \
"(Year 4)/CM - Individual Project/2. Code/Yelp data/reviews.pkl"
###
###

df = pd.read_pickle(path)

###
# Extract Feature 1 - Review length (number of characters in review string)
###

number_of_reviews = len(df)
reviewLengthList = []
for review_index in range(number_of_reviews):
    length_of_review = len(df["reviewContent"].iloc[review_index])
    reviewLengthList.append(length_of_review)
df["reviewLength"] = reviewLengthList
df.reset_index(drop = True)

###
# Extract Feature 11 - Maximum number of reviews written by reviewer in a single day
###

# create series with all unique reviewerID values
unique_reviewers = pd.Series(df["reviewerID"].unique())
# create empty column to be used for filling in values
df["maxReviews"] = np.nan

for reviewer_index in range(0, len(unique_reviewers)):
    
    # 1. get maximum number of reviews posted in a single day for given reviewer
    max_reviews_reviewer = df["date"][df["reviewerID"] == unique_reviewers[reviewer_index]].value_counts()[0]
    # 2. update the column "maxReviews" for every row with the given reviewer
    df.loc[df.reviewerID == unique_reviewers[reviewer_index], "maxReviews"] = max_reviews_reviewer
    
# convert column to integers after all values filled in
df["maxReviews"] = df["maxReviews"].apply(np.int64)

###
# Extract Feature 2 - Average sentiment of review
###

# initialise Sentiment Intensity Analyzer (SIA)
analyzer = SIA()
# create empty list to store all sentiment values
sentiment_list = []

# calculate for each review an average sentiment 
# value based on individual sentences
for review in df['reviewContent']:

    # 1. split review into individual sentences (tokenise)
    review_sentences = tokenize.sent_tokenize(review)
    
    # 2. get the average sentiment for the review 
    # (average sentiment of sentences in review)
    total_sentiment = 0.0
    for sentence in review_sentences:
        sentiment_score = analyzer.polarity_scores(sentence)
        total_sentiment += sentiment_score["compound"]
    
    # calculate average sentiment
    average_sentiment = round(total_sentiment / no_sentences, 4)
    
    # 3. append average sentiment to sentiment_list
    sentiment_list.append(average_sentiment)
    
# add new feature to dataframe
df["avgSentiment"] = sentiment_list

###
# Extract Features 3-8 - Parts of Speech (PoS)
###

# load parts of speech tagger from spaCy library
tagger = spacy.load("en_core_web_sm")

noun_proportion_list = []
adj_proportion_list = []
verb_proportion_list = []
propnoun_proportion_list = []
num_count_list = []
symbol_count_list = []

for review in df['reviewContent']:
        
    # 1. tokenise and tag review
    tagged_review = tagger(review)
    
    # get count of tokens (words) in review
    token_count = len(tagged_review)
    
    noun_count = 0
    adj_count = 0
    verb_count = 0
    propnoun_count = 0
    num_count = 0
    symbol_count = 0
    
    # 2. Get counts of nouns, adjectives, verbs, 
    # proper nouns, numbers and symbols
    for token in tagged_review:
        
        if(token.pos_ == "NOUN"):
            noun_count += 1
        if(token.pos_ == "ADJ"):
            adj_count += 1
        if(token.pos_ == "VERB"):
            verb_count += 1
        if(token.pos_ == "PROPN"):
            propnoun_count += 1
        if(token.pos_ == "NUM"):
            num_count += 1
        if(token.pos_ == "SYM"):
            symbol_count += 1
        
    # 3. append proportions to lists
    noun_proportion_list.append(noun_count / token_count)
    adj_proportion_list.append(adj_count / token_count)
    verb_proportion_list.append(verb_count / token_count)
    propnoun_proportion_list.append(propnoun_count / token_count)
    num_count_list.append(num_count)
    symbol_count_list.append(symbol_count)
    
# add new features to dataframe
df["nounProp"] = noun_proportion_list
df["adjProp"] = adj_proportion_list
df["verbProp"] = verb_proportion_list
df["propernounProp"] = propnoun_proportion_list
df["numCount"] = num_count_list
df["symCount"] = symbol_count_list
df = df.reset_index(drop = True)

###
# Extract Feature 9 - Maximum TF-IDF score for a given review
###

# create corpus - a list of all reviews
corpus = df["reviewContent"].tolist()
# initialise vectoriser and assign stop word list
# words appearing in more than 85% of documents will
# not be part of the vector
vectoriser = TfidfVectorizer(max_df = 0.85, stop_words = "english")
# fit and transform TF-IDF model on corpus
vocabulary = vectoriser.fit_transform(corpus)
max_tfidf_list = []

# loop through all reviews and 
# retrieve their maximum TF-IDF value
for document_index in range(len(corpus)):
        
    max_value = vocabulary[document_index].max()
    max_tfidf_list.append(max_value)
    
# add new feature to dataframe
df["max_tfidf"] = max_tfidf_list

###
# Feature X - Reviewer membership length (skipped)
###

# length of time reviewer has been a member of the online platform (Yelp)
# this feature has been skipped because the reviewer table provided in the data is incomplete, 
# with over 40,000 of the 60,019 reviewerIDs unmatched in the reviews table

###
# Feature X - Friend count (skipped)
###
# skipped due to same reason as the previous feature (unmatched reviewerIDs)

###
# Extract Feature 12 - Count of reviews posted by reviewer
###

# create series with all unique reviewerID values
unique_reviewers = pd.Series(df["reviewerID"].unique())
print("Number of unique reviewers: ", len(unique_reviewers))
# create empty column to be used for filling in values
df["postCount"] = np.nan

for reviewer_index in range(0, len(unique_reviewers)):
        
    # 1. get number of reviews posted by a given reviewer
    posts_by_reviewer = df["reviewerID"][df["reviewerID"] == unique_reviewers[reviewer_index]].value_counts().sum()
    
    # 2. update the column "postCount" for every row with the given reviewer
    df.loc[df.reviewerID == unique_reviewers[reviewer_index], "postCount"] = posts_by_reviewer
    
# convert column to integers after all values filled in
df["postCount"] = df["postCount"].apply(np.int64)

# Pickle (serialise) dataframe with all features ready to be used in model evaluation
df.to_pickle("./df3.pkl")

# Load dataframe for feature selection

# load dataframe with all features extracted
### SET OWN PATH HERE
###
path = "/Users/artembutbaev/OneDrive/University of Bath 20-21 " + \
"(Year 4)/CM - Individual Project/2. Code/Model Building/df4.pkl"
###
###

df = pd.read_pickle(path)

##########################

# Feature Selection

# Define functions to be used for feature selection

# print quartiles and spread of data points for a given feature
def print_quartiles(df, feature, bins_count):
    
    df_genuine = df[df["flagged"] == 'N']
    df_fake = df[df["flagged"] == 'Y']
    
    feature_genuine = df_genuine[feature]
    feature_fake = df_fake[feature]

    print(feature + " - Genuine reviews: ")
    print("Median (Q2): ", np.quantile(feature_genuine, .50)) 
    print("Q1: ", np.quantile(feature_genuine, .25))
    print("Q3: ", np.quantile(feature_genuine, .75))
    print("IQR: ", np.quantile(feature_genuine, .75) - np.quantile(feature_genuine, .25))
    print()
    print(feature + " - Fake reviews: ")
    print("Median (Q2): ", np.quantile(feature_fake, .50)) 
    print("Q1: ", np.quantile(feature_fake, .25)) 
    print("Q3: ", np.quantile(feature_fake, .75)) 
    print("IQR: ", np.quantile(feature_fake, .75) - np.quantile(feature_fake, .25)) 

    # value counts (in bins) - use for bar charts to visually compare
    print("Total: ", len(feature_genuine))
    print("Value counts - Genuine reviews: ")
    print(feature_genuine.value_counts(bins=bins_count, dropna=False))
    print()
    print("Total: ", len(feature_fake))
    print("Value counts - Fake reviews: ")
    print(feature_fake.value_counts(bins=bins_count, dropna=False))

# create a boxplot for a given feature and an attribute to group it by (typically, the label)
def create_boxplot(df, feature, group_by):
    
    boxplot = df.boxplot(column=[feature], by=group_by, return_type=None, showfliers=False)
    figure = boxplot.get_figure()
    figure.suptitle('')
    plt.show()
    
# Print quartiles and create associated boxplots

# 1. reviewLength feature
print_quartiles(df, "reviewLength", 5)
create_boxplot(df, "reviewLength", "flagged")
# 2. maxReviews feature
print_quartiles(df, "maxReviews", 1)
create_boxplot(df, "maxReviews", "flagged")
# 3. avgSentiment feature
print_quartiles(df, "avgSentiment", 5)
create_boxplot(df, "avgSentiment", "flagged")
# 4. adjective proportion feature
print_quartiles(df, "adjProp", 10)
create_boxplot(df, "adjProp", "flagged")
# 5. max tf-idf feature
print_quartiles(df, "max_tfidf", 5)
create_boxplot(df, "max_tfidf", "flagged")
# 6. post count feature
print_quartiles(df, "postCount", 5)
create_boxplot(df, "postCount", "flagged")
# 7. noun proportion feature
print_quartiles(df, "nounProp", 5)
create_boxplot(df, "nounProp", "flagged")
# 8. verb proportion feature
print_quartiles(df, "verbProp", 5)
create_boxplot(df, "verbProp", "flagged")
# 9. propernoun proportion feature
print_quartiles(df, "propernounProp", 5)
create_boxplot(df, "propernounProp", "flagged")
# 10. numcount feature
print_quartiles(df, "numCount", 10)
create_boxplot(df, "numCount", "flagged")
# 11. symcount feature
print_quartiles(df, "symCount", 10)
create_boxplot(df, "symCount", "flagged")
# 12. hasProfile feature
print_quartiles(df, "hasProfile", 1)
create_boxplot(df, "hasProfile", "flagged")
# 13. usefulCount feature
print_quartiles(df, "usefulCount", 1)
create_boxplot(df, "usefulCount", "flagged")
# 14. coolCount feature
print_quartiles(df, "coolCount", 1)
create_boxplot(df, "coolCount", "flagged")
# 15. funnyCount feature
print_quartiles(df, "funnyCount", 1)
create_boxplot(df, "funnyCount", "flagged")
%matplotlib inline

# calculates point-biserial correlation between labels and a given feature
def point_biserial(labels, feature):
    
    # convert feature to numpy array
    feature = np.asarray(feature)
    
    # Output: -1 indicates a perfect negative association, 
    # +1 indicates a perfect positive association, and 0 indicates no association
    #return stats.pointbiserialr(labels, feature)
    result, p_value = stats.pointbiserialr(labels, feature)
    return result

# Calculate point-biserial correlation for each feature and the associated labels
###

# convert labels to binary integers ("Y" = 1, "N" = 0)
labels = df["flagged"].tolist()
labels_binary = list(labels)
for label_index in range(len(labels)):
    if(labels[label_index] == 'Y'):
        labels_binary[label_index] = 1
    elif(labels[label_index] == 'N'):
        labels_binary[label_index] = 0
# convert labels list to numpy array
labels_binary = np.asarray(labels_binary)

# feature 1 - length of a review
feature_1 = df["reviewLength"].tolist()
print("reviewLength feature: ")
print(point_biserial(labels_binary, feature_1))
# feature 2 - maximum number of reviews by reviewer in a single day
feature_2 = df["maxReviews"].tolist()
print("maxReviews feature: ")
print(point_biserial(labels_binary, feature_2))
# feature 3 - average sentiment of a review
feature_3 = df["avgSentiment"].tolist()
print("avgSentiment feature: ")
print(point_biserial(labels_binary, feature_3))
# feature 4 - proportion of nouns
feature_4 = df["nounProp"].tolist()
print("nounProp feature: ")
print(point_biserial(labels_binary, feature_4))
# feature 5 - proportion of adjectives
feature_5 = df["adjProp"].tolist()
print("adjProp feature: ")
print(point_biserial(labels_binary, feature_5))
# feature 6 - proportion of verbs
feature_6 = df["verbProp"].tolist()
print("verbProp feature: ")
print(point_biserial(labels_binary, feature_6))
# feature 7 - proportion of proper nouns
feature_7 = df["propernounProp"].tolist()
print("propernounProp feature: ")
print(point_biserial(labels_binary, feature_7))
# feature 8 - max tf-idf
feature_8 = df["max_tfidf"].tolist()
print("max_tfidf feature: ")
print(point_biserial(labels_binary, feature_8))
# feature 9 - count of numbers in review
feature_9 = df["numCount"].tolist()
print("numCount feature: ")
print(point_biserial(labels_binary, feature_9))
# feature 10 - count of symbols in review
feature_10 = df["symCount"].tolist()
print("symCount feature: ")
print(point_biserial(labels_binary, feature_10))
# feature 11 - count of posts
feature_11 = df["postCount"].tolist()
print("postCount feature: ")
print(point_biserial(labels_binary, feature_11))
# feature 12 - hasProfile
feature_12 = df["hasProfile"].tolist()
print("hasProfile feature: ")
print(point_biserial(labels_binary, feature_12))
# feature 13 - usefulCount
feature_13 = df["usefulCount"].tolist()
print("usefulCount feature: ")
print(point_biserial(labels_binary, feature_13))
# feature 14 - coolCount
feature_14 = df["coolCount"].tolist()
print("coolCount feature: ")
print(point_biserial(labels_binary, feature_14))
# feature 15 - funnyCount
feature_15 = df["funnyCount"].tolist()
print("funnyCount feature: ") 
print(point_biserial(labels_binary, feature_15))

# Calculate t-test for groups of features for genuine and fake review
###

df_genuine = df[df["flagged"] == "N"]
df_fake = df[df["flagged"] == "Y"]

# 1. review length
feature_genuine = df_genuine["reviewLength"]
feature_fake = df_fake["reviewLength"]
print('\033[1m' + "1. Review length:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 2. maximum reviews posted in a single day by reviewer
feature_genuine = df_genuine["maxReviews"]
feature_fake = df_fake["maxReviews"]
print('\033[1m' + "2. Maximum reviews posted by reviewer in a single day:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 3. average sentiment of sentences in review
feature_genuine = df_genuine["avgSentiment"]
feature_fake = df_fake["avgSentiment"]
print('\033[1m' + "3. Average sentiment of sentences in review:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 4. noun proportion out of all words in review
feature_genuine = df_genuine["nounProp"]
feature_fake = df_fake["nounProp"]
print('\033[1m' + "4. Noun proportion:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 5. adjective proportion out of all words in review
feature_genuine = df_genuine["adjProp"]
feature_fake = df_fake["adjProp"]
print('\033[1m' + "5. Adjective proportion:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 6. verb proportion out of all words in review
feature_genuine = df_genuine["verbProp"]
feature_fake = df_fake["verbProp"]
print('\033[1m' + "6. Verb proportion:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 7. proper noun proportion out of all words in review
feature_genuine = df_genuine["propernounProp"]
feature_fake = df_fake["propernounProp"]
print('\033[1m' + "7. Proper noun proportion:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 8. maximum tf-idf value in review
feature_genuine = df_genuine["max_tfidf"]
feature_fake = df_fake["max_tfidf"]
print('\033[1m' + "8. Maximum tf-idf value:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 9. maximum tf-idf value in review
feature_genuine = df_genuine["postCount"]
feature_fake = df_fake["postCount"]
print('\033[1m' + "9. Post count:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 10. symbol count in review
feature_genuine = df_genuine["symCount"]
feature_fake = df_fake["symCount"]
print('\033[1m' + "10. Symbol count:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 11. number count in review
feature_genuine = df_genuine["numCount"]
feature_fake = df_fake["numCount"]
print('\033[1m' + "11. Number count:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 12. HasProfile
feature_genuine = df_genuine["hasProfile"]
feature_fake = df_fake["hasProfile"]
print('\033[1m' + "12. Has reviewer got a profile:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 13. usefulCount
feature_genuine = df_genuine["usefulCount"]
feature_fake = df_fake["usefulCount"]
print('\033[1m' + "13. Useful count:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 14. coolCount
feature_genuine = df_genuine["coolCount"]
feature_fake = df_fake["coolCount"]
print('\033[1m' + "14. Cool count:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)
# 15. funnyCount
feature_genuine = df_genuine["funnyCount"]
feature_fake = df_fake["funnyCount"]
print('\033[1m' + "15. Funny count:" + '\033[0m')
statistic, p_value = stats.ttest_ind(feature_genuine, feature_fake)
print(statistic)