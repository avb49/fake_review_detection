# Import required libraries

import pandas as pd
import numpy as np
import pickle
# import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# import functionality to split data into training and testing
from sklearn.model_selection import train_test_split
# import functionality for creating stratified k-fold cross validation
from sklearn.model_selection import StratifiedKFold
# import metrics module to construct confusion matrix
from sklearn import metrics

# function for balancing dataset using oversampling and undersampling
def resample(x_train, y_train):

    # 1. "combine" x_train and y_train
    x_train = x_train.copy()
    x_train["label"] = y_train

    # create a sub-dataframes
    genuine = x_train[x_train["label"] == "N"]
    fake = x_train[x_train["label"] == "Y"]

    # 1. oversampling

    # 2x oversampling
    fake_copy = fake.copy()
    fake = pd.concat([fake, fake_copy])

    # 2. undersampling

    # shuffle rows
    genuine = genuine.reindex(np.random.permutation(genuine.index))
    # under-sample majority class
    genuine = genuine[:len(fake)]

    # combine two sub-dataframes
    x_train = pd.concat([genuine, fake])
    x_train = x_train.reset_index(drop = True)

    # shuffle rows
    x_train = x_train.reindex(np.random.permutation(x_train.index))

    # 3. "separate" x_train and y_train again
    y_train = pd.Series(x_train["label"])
    x_train = x_train.drop("label", 1)
    
    return x_train, y_train

# Load dataframe for model evaluation

# SET PATH HERE FOR DATAFRAME WITH ALL FEATURES
# load dataframe with all 15 features ready
path = "/Users/artembutbaev/OneDrive/University of Bath 20-21 " + \
"(Year 4)/CM - Individual Project/2. Code/Model Building/df4.pkl"
df = pd.read_pickle(path)

# Fake and genuine review examples included in dissertation
# df[df['flagged'] == 'N'].iloc[2333].reviewContent
# df[df['flagged'] == 'Y'].iloc[233].reviewContent
# df

### Define feature sets

labels = pd.Series(df["flagged"])

# feature set 1 - review-centric features only
columns = [df["max_tfidf"], df["reviewLength"], df["numCount"], 
           df["symCount"], df["adjProp"], df["avgSentiment"], df["nounProp"]]
headers = ["max_tfidf", "reviewLength", "numCount", 
           "symCount", "adjProp", "avgSentiment", "nounProp"]
fs1 = pd.concat(columns, axis=1, keys=headers)

# feature set 2 - reviewer-centric features only
columns = [df["hasProfile"], df["postCount"], df["usefulCount"], 
           df["coolCount"], df["funnyCount"], df["maxReviews"]]
headers = ["hasProfile", "postCount", "usefulCount", 
           "coolCount", "funnyCount", "maxReviews"]
fs2 = pd.concat(columns, axis=1, keys=headers)

# feature set 3 - all features
columns = [df["max_tfidf"], df["reviewLength"], df["numCount"], 
           df["symCount"], df["adjProp"], df["avgSentiment"], df["nounProp"], 
           df["hasProfile"], df["postCount"], df["usefulCount"], 
           df["coolCount"], df["funnyCount"], df["maxReviews"]]
headers = ["max_tfidf", "reviewLength", "numCount", 
           "symCount", "adjProp", "avgSentiment", "nounProp", 
           "hasProfile", "postCount", "usefulCount", 
           "coolCount", "funnyCount", "maxReviews"]
fs3 = pd.concat(columns, axis=1, keys=headers)

# feature set 4 - both categories, top 10 overall based on correlation and t-test
columns = [df["hasProfile"], df["postCount"], 
           df["usefulCount"], df["max_tfidf"], df["reviewLength"], 
           df["coolCount"], df["funnyCount"], df["maxReviews"], 
           df["numCount"], df["symCount"]]
headers = ["hasProfile", "postCount", "usefulCount", "max_tfidf", 
           "reviewLength", "coolCount", "funnyCount", 
           "maxReviews", "numCount", "symCount"]
fs4 = pd.concat(columns, axis=1, keys=headers)

# feature set 5 - both categories, top 5 overall based on correlation and t-test
columns = [df["hasProfile"], df["postCount"], 
           df["usefulCount"], df["max_tfidf"], df["reviewLength"]]
headers = ["hasProfile", "postCount", "usefulCount", 
           "max_tfidf", "reviewLength"]
fs5 = pd.concat(columns, axis=1, keys=headers)

# feature set 6 - both categories, top 5 overall based on feature 
# importance of model trained on FS3 (all features)
columns = [df["usefulCount"], df["postCount"], 
           df["avgSentiment"], df["reviewLength"], df["max_tfidf"]]
headers = ["usefulCount", "postCount", 
           "avgSentiment", "reviewLength", "max_tfidf"]
fs6 = pd.concat(columns, axis=1, keys=headers)

# feature set 7 - both categories, top 10 overall based on feature 
# importance of model trained on FS3 (all features)
columns = [df["usefulCount"], df["postCount"], df["avgSentiment"], 
           df["reviewLength"], df["max_tfidf"], df["nounProp"], 
           df["adjProp"], df["hasProfile"], df["numCount"], df["coolCount"]]
headers = ["usefulCount", "postCount", "avgSentiment", "reviewLength", 
           "max_tfidf", "nounProp", "adjProp", "hasProfile", "numCount", "coolCount"]
fs7 = pd.concat(columns, axis=1, keys=headers)

###
# Evaluate average model performance using stratified 10-fold cross-validation
###

###
# SPECIFY FEATURE SET HERE
features = fs4
###

max_depth = 6
number_of_folds = 10
skf = StratifiedKFold(n_splits = number_of_folds, shuffle = True)
skf.get_n_splits(features, labels)
print(skf)
count = 1

acc_sum = 0
precision_sum = 0
recall_sum = 0
f_sum = 0
balanced_sum = 0

# loop through train/test data splits
for train_index, test_index in skf.split(features, labels):
    x_train, x_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

    ###
    # 1. re-sample training set ONLY (keep test data class distribution the same)
    ###
    # Comment out the following line to train model on natural 
    # class distribution of fake/genuine reviews
    x_train, y_train = resample(x_train, y_train)
    
    # check effect of resampling on the number of fake 
    # and genuine review samples in training dataset
    #print("Fake reviews in training set:")
    #print(len(y_train[y_train == 'Y']))
    #print("Genuine reviews in training set:")
    #print(len(y_train[y_train == 'N']))
    #print()

    ###
    # 2. train classifier
    ###
    # use the following line for training a fully grown 
    # tree with no hyper-parameter tuning
    # dt = DecisionTreeClassifier()
    # use the following line for training a tree with 
    # hyper-parameter tuning applied
    dt = DecisionTreeClassifier(max_depth = max_depth)
    
    # build decision tree from training data
    dt.fit(x_train, y_train)

    ###
    # 3. predict test data
    predictions = dt.predict(x_test)
    ###
    
    # create confusion matrix based on predictions 
    # vs. actual labels of reviews
    confusion_matrix = metrics.confusion_matrix(y_test, predictions, labels=['N','Y'])
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
    
    ###
    # 4. calculate evaluation metrics
    ###
    
    # a. accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # b. precision
    if(tp == 0 and fp == 0):
        precision = 0
    else:
        precision = tp / (tp + fp)
    # c. recall
    if(tp == 0 and fn == 0):
        recall = 0
    else:
        recall = tp / (tp + fn)
    # d. f score
    if(precision == 0 and recall == 0):
        f_score = 0
    else:
        f_score = (2 * precision * recall) / (precision + recall)
    if(tn == 0 and fp == 0):
        true_negative_rate = 0
    else:
        true_negative_rate = tn / (tn + fp)
    # e. balanced accuracy
    balanced_accuracy = (true_negative_rate + recall) / 2
    
    # get feature importances - Gini importance
    # it represents the normalised total reduction of the criterion by a given feature
    #importances = dict(zip(features.columns, dt.feature_importances_))
    #feature_importances = dt.feature_importances_
    #for index in range(len(feature_importances)):
        #list_of_feature_importance[index] += feature_importances[index]
        #print(feature_importances[index])
    #print()

    print("Run ", count, ": ")
    print("Overall accuracy: ", accuracy * 100)
    print("Balanced accuracy: ", balanced_accuracy * 100)
    print("Precision: ", precision * 100)
    print("Recall: ", recall * 100)
    print("TNR: ", true_negative_rate * 100)
    print("F Score: ", f_score * 100)
    print()
    count += 1
    
    acc_sum += accuracy
    precision_sum += precision
    recall_sum += recall
    f_sum += f_score
    balanced_sum += balanced_accuracy

print()
print("Average accuracy over", number_of_folds, "runs: ", 
      round(acc_sum / number_of_folds * 100, 2))
print("Average balanced accuracy over", number_of_folds, "runs: ", 
      round(balanced_sum / number_of_folds * 100, 2))
print("Average precision over", number_of_folds, "runs: ", 
      round(precision_sum / number_of_folds * 100, 2))
print("Average recall over", number_of_folds, "runs: ", 
      round(recall_sum / number_of_folds * 100, 2))
print("Average F score over", number_of_folds, "runs: ", 
      round(f_sum / number_of_folds * 100, 2))
print()

# get average feature importance of each feature during k-fold CV
#for index in range(len(list_of_feature_importance)):
    #list_of_feature_importance[index] = list_of_feature_importance[index] / number_of_folds


###
# Hyper-parameter optimisation (max_depth feature)
###


###
# SPECIFY FEATURE SET HERE
features = fs4
###

number_of_folds = 10
skf = StratifiedKFold(n_splits = number_of_folds, shuffle = True)
skf.get_n_splits(features, labels)

# specify range of values here to test for hyper-parameter optimisation
max_depth_range = (range(3,26))
max_depth_avg_f1 = []

for max_depth in max_depth_range:
    
    print("Performing 10-fold cross-validation for max_depth: ", max_depth)
    
    f_sum = 0
    
    # loop through train/test data splits
    for train_index, test_index in skf.split(features, labels):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
        # balance training set
        x_train, y_train = resample(x_train, y_train)
        
        dt = DecisionTreeClassifier(max_depth = max_depth)
        dt.fit(x_train, y_train)
        
        predictions = dt.predict(x_test)
        confusion_matrix = metrics.confusion_matrix(y_test, predictions, labels=['N','Y'])
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        
        # b. precision
        if(tp == 0 and fp == 0):
            precision = 0
        else:
            precision = tp / (tp + fp)
        # c. recall
        if(tp == 0 and fn == 0):
            recall = 0
        else:
            recall = tp / (tp + fn)
        # d. f score
        if(precision == 0 and recall == 0):
            f_score = 0
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        
        f_sum += f_score
    avg_score = round(f_sum / number_of_folds * 100, 2)
    #print("Average F score over", number_of_folds, " + \
    #"runs for max_depth", max_depth, "was:", avg_score)
    #print()
    #print("...")
    #print()
    max_depth_avg_f1.append(avg_score)

scores = dict(zip(max_depth_range, max_depth_avg_f1))
print(scores)

# plot the average F1 score against the max_depth value on a scatter plot 
import matplotlib.pyplot as plt
plt.scatter(scores.keys(), scores.values())
plt.show()

###
# get feature importances and decision tree characteristics
###

# Print feature importances in construction of tree

# Also known as Gini importance, it represents the normalised total 
# reduction of the criterion by a given feature
# importances add up to 1.0
importances = dict(zip(features.columns, dt.feature_importances_))
for value in importances:
    print(value, ":", importances[value])
print()
    
# Print decision tree characteristics

print("decision tree characteristics: ")
print("depth of decision tree: ", dt.get_depth())
print("number of leaves: ", dt.get_n_leaves())
print()
dt.get_params(deep=True)

#### Serialise classifier

# Serialise decision tree classifier and train/test 
# data and test reviews for front end

# serialise classifier
with open('dt_final.pkl', 'wb') as f:
    pickle.dump(dt, f)

# serialise train/test data
with open('x_train.pkl', 'wb') as f:
    pickle.dump(x_train, f)
with open('x_test.pkl', 'wb') as f:
    pickle.dump(x_test, f)
with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)    
with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
    
# serialise reviews associated with test data
df1 = df[df.index.isin(x_test.index)]
with open('df_test.pkl', 'wb') as f:
    pickle.dump(df1, f)