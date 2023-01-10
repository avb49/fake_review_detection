# import framework to work with sqlite database in Python
import sqlite3
# pandas will be used for manipulating data sets retrieved from the database file
import pandas as pd
import unicodedata
# pickle will be used to serialise the prepared dataframe to be loaded later
import pickle

# set up connection to the database
connection = sqlite3.connect("yelpResData.db")
# create cursor object for interaction with the database
cur = connection.cursor()
# specify how to handle bytes in database
connection.text_factory = lambda x: str(x, 'utf-8', 'ignore')

# previously attempted text factories below
#connection.text_factory = bytes
#connection.text_factory = lambda x: str(x, 'iso-8859-1')

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())

review_table = pd.read_sql_query('SELECT * FROM review', connection)
#restaurant_table = pd.read_sql_query('SELECT * FROM restaurant', connection)
reviewer_table = pd.read_sql_query('SELECT * FROM reviewer', connection)

series = pd.Series(review_table['reviewContent'])
series = series.str.normalize("NFKD")
review_table['reviewContent'] = series.values

reviews = review_table[(review_table["flagged"] == 'Y') | (review_table["flagged"] == 'N')].reset_index(drop = True)

number_of_reviews = len(reviews)
number_of_genuine_reviews = (reviews["flagged"] == 'N').sum()   
number_of_filtered_reviews = (reviews["flagged"] == 'Y').sum()   

print("Number of reviews:", number_of_reviews)
print("Number of genuine reviews:", number_of_genuine_reviews)
print("Number of filtered reviews:", number_of_filtered_reviews)

# check if there are any duplicate reviews
reviews["reviewID"].nunique()

# save file to csv for further inspection
#df.to_csv("dataframe.csv", encoding='utf-8', index=False)

# identify indeces containing "Updated" to remove the strings
date_column = pd.Series(reviews["date"])
indeces = date_column.str.contains('updated', case=False, regex=True)
print(indeces.value_counts())

true_indeces = indeces[indeces == True].index

# print values to fix
for index in range(0, len(true_indeces)):
    print(date_column.iloc[true_indeces[index]])
    
# "slice" the individual values to remove "Updated -"
for index in true_indeces:
    date_column.iloc[index] = date_column.iloc[index][10:]
    
# check that update is successful
indeces = date_column.str.contains('updated', case=False, regex=True)
print(indeces.value_counts())

# make date formatting consistent in the date column of the dataframe
reviews["date"] = date_column
reviews["date"] = pd.to_datetime(reviews["date"])

# show count of reviews by year
years = reviews["date"].dt.year
print(years.value_counts())

# remove empty reviews
empty_review_index_list = [62005, 62792]
reviews = reviews.drop(reviews.index[empty_review_index_list])

reviews = reviews.sort_index()

columns = [reviews["date"], reviews["reviewerID"], reviews["reviewContent"], reviews["flagged"]]
headers = ["date","reviewerID", "reviewContent", "flagged"]
reviews = pd.concat(columns, axis=1, keys=headers).reset_index(drop = True)

reviews

# temporary - get usefulCount, coolCount and funnyCount features from dataframe
columns = [reviews["usefulCount"], reviews["coolCount"], reviews["funnyCount"]]
headers = ["usefulCount","coolCount", "funnyCount"]
extra_features = pd.concat(columns, axis=1, keys=headers).reset_index(drop = True)

# pickle
extra_features.to_pickle("./extra_features.pkl")

# explore "anonymous" reviewers
# i.e., those not found in the reviewer table (from the exercise in the above cell)

# "anonymous" means ID in reviews table but NOT in reviewerID
anonymous_reviewers = []
has_profile = []

reviewers_in_table1 = pd.Series(list(reviews["reviewerID"].unique()))
reviewers_in_table2 = set(list(reviewer_table["reviewerID"]))

for reviewer in reviewers_in_table1:
    if(reviewer not in reviewers_in_table2):
        anonymous_reviewers.append(reviewer)
        has_profile.append(0)
    else:
        has_profile.append(1)
        
print("Number of unique reviewers in reviews table: ", len(reviews['reviewerID'].unique()))
print("Number of unique reviewers in reviewer table: ", len(reviewer_table['reviewerID'].unique()))

print("Number of \"anonymous\" reviewers: ", len(anonymous_reviewers))
# e.g. reviewer ID "xMYPc5tzV2PSryKFK_y1PQ" is found in reviews but not reviewers table
print()
print("Examples of \"anonymous\" reviewers: ")
print("Number of reviews in reviews table with reviewer ID xMYPc5tzV2PSryKFK_y1PQ: ", 
      len(reviews[reviews["reviewerID"] == "xMYPc5tzV2PSryKFK_y1PQ"]))
print("Number of reviews in reviewers table with reviewer ID xMYPc5tzV2PSryKFK_y1PQ: ", 
      len(reviewer_table[reviewer_table["reviewerID"] == "xMYPc5tzV2PSryKFK_y1PQ"]))
print()
print("Number of reviews in reviews table with reviewer ID ciAaaK5kBPGM1y8CtkJtXQ: ", 
      len(reviews[reviews["reviewerID"] == "ciAaaK5kBPGM1y8CtkJtXQ"]))
print("Number of reviews in reviewers table with reviewer ID ciAaaK5kBPGM1y8CtkJtXQ: ", 
      len(reviewer_table[reviewer_table["reviewerID"] == "ciAaaK5kBPGM1y8CtkJtXQ"]))

reviews[reviews["reviewerID"] == "xMYPc5tzV2PSryKFK_y1PQ"]
# for this particular reviewer, we can see that they are not present in the reviewers' table, yet have posted 
# many times at different dates, from 2007-2009

# check if any two groups of users do not have fake reviews
x1 = len(reviews[(reviews['hasProfile'] == 1) & (reviews['flagged'] == 'Y' )])
x2 = len(reviews[(reviews['hasProfile'] == 1) & (reviews['flagged'] == 'N' )])
print("Proportion of fake reviews by reviewers with a profile: ")
print(x1 / (x1 + x2))
print("Proportion of genuine reviews by reviewers with a profile: ")
print(x2 / (x1 + x2))
print("Total: ", x1 + x2)
print()
x3 = len(reviews[(reviews['hasProfile'] == 0) & (reviews['flagged'] == 'Y' )])
x4 = len(reviews[(reviews['hasProfile'] == 0) & (reviews['flagged'] == 'N' )])
print("Proportion of fake reviews by reviewers without a profile: ")
print(x3 / (x3 + x4))
print("Proportion of genuine reviews by reviewers without a profile: ")
print(x4 / (x3 + x4))
print("Total: ", x3 + x4)

# add column in reviews table to indicate whether a reviewer is "anonymous"
reviews["hasProfile"] = has_profile

reviews.to_pickle("./reviews.pkl")

cur.close()
connection.close()