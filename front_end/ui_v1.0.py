# Import required libraries

import numpy as np
import pickle
from sklearn import tree
from matplotlib import pyplot as plt
# import graphviz - library which will be used to visualise the decision tree and decision paths
import graphviz
# import pydotplus - an interface to the DOT language used by graphviz, which will be used to modify graphs
import pydotplus
# import webbrowser - will be used to open a new tab with the content of the HTML file created
import webbrowser
# import datetime - for including the current date and time in the HTML file generated
import datetime
# import functionality to get the current directory path
from pathlib import Path

# Define functions for getting the decision path for a given review and to create HTML output

# creates a diagram of the decision tree with the decision path for a given review highlighted and saves it as a png
def create_decision_path_png(dt, feature_names, filename, sample):
    
    # export decision tree in DOT (graphviz) format
    dot_data = tree.export_graphviz(dt, out_file=None, 
                                feature_names=feature_names, class_names=dt.classes_, 
                                filled=True, rounded=True, special_characters=True, node_ids = True)
    
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    # set all nodes to be of white colour
    for node in graph.get_node_list():
        node.set_fillcolor('white')
        
    # get decision path for chosen review sample
    decision_path = dt.decision_path(sample)

    # loop through each node in tree and its associated decision-path value, 
    # i.e., if the node is part of the decision path then it has a value of 1, otherwise, it has a value of 0
    for node_counter, node_value in enumerate(decision_path.toarray()[0]):
        
        # if node is not part of decision path for sample, continue 
        if(node_value == 0):
            continue
        # update colour of nodes part of decision path to be of yellow colour
        dot_node = graph.get_node(str(node_counter))[0]
        dot_node.set_fillcolor('yellow')
        
    # export updated graph to a png
    graph.write_png(filename)
    
# gets for a sample review the decision path in text form
# inspiration from: 
# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
def get_decision_path_text(classifier, feature_names, sample):

    # retrieve probabilities for classification for sample
    probabilities = classifier.predict_proba(sample)
    # retrieve decision paths for test sample
    node_indicator = classifier.decision_path(sample)
    # retrieve leaf IDs reached by test sample
    leaf_identifier = classifier.apply(sample)

    # get list of feature names
    feature_names = feature_names
    # get a list of all thresholds (of all nodes in tree)
    threshold = classifier.tree_.threshold
    # get node id of leaf that the sample "lands on"
    leaf_id = leaf_identifier[0]
    # obtain ids of the nodes sample goes through
    nodes_traversed = node_indicator.indices[node_indicator.indptr[0]:
                                    node_indicator.indptr[1]]

    decision_counter = 0
    decisions_made = []

    for node_id in nodes_traversed:
    
        # increment counter
        decision_counter += 1
    
        # break for loop if it is a leaf node
        if(node_id == leaf_identifier[0]):
            leaf_node_string = "Leaf node ({node}) reached. End of decision path.".format(node = node_id)
            decisions_made.append(leaf_node_string)
            break

        # check if value of the split feature for sample i is below threshold
        if (sample.iloc[0][classifier.tree_.feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        current_decision = "decision {decision_no} (at node {node}) : {feature_name} = {value} {inequality} {threshold}".format(
                  decision_no=decision_counter,
                  node=node_id,
                  feature_name=feature_names[classifier.tree_.feature[node_id]],
                  value=round(sample.iloc[0][classifier.tree_.feature[node_id]], 3),
                  inequality=threshold_sign,
                  threshold=round(threshold[node_id], 3))
        
        decisions_made.append(current_decision)

    # get predicted classification of sample based on samples at the leaf node
    predicted_classification = classifier.classes_[np.argmax(classifier.tree_.value[leaf_id])]
    class_proportion = max(classifier.tree_.value[leaf_id])
    class_probability = round(100 * max(probabilities[0]), 1)

    # get the predicted classification and confidence probability
    if(predicted_classification == "N"):
        predicted_string = "This review has been classified as Genuine with confidence probability: {prob}%".format(prob=class_probability)
    else:
        predicted_string = "This review has been classified as Fake with confidence probability: {prob}%".format(prob=class_probability)
        
    return decisions_made, predicted_string

# prepares the string to wrap in the HTML document to be displayed to the user
def prepare_string_to_wrap(index, review_text, decisions_made, predicted_string):
    
    # prepare string to wrap in html document

    string_to_wrap = ""

    # add title
    index_string = "Decision path for sample " + str(index) + ":"
    string_to_wrap += index_string
    string_to_wrap += "<br/><br/>"

    # add review text
    string_to_wrap += review_text
    string_to_wrap += "<br/><br/>"
    
    # add decisions made to string
    for decision in decisions_made:
        string_to_wrap += decision
        string_to_wrap += "<br/>"

    # add classification and accuracy to string
    string_to_wrap += "<br/>"
    string_to_wrap += predicted_string
    string_to_wrap += "<br/>"
    
    return string_to_wrap

# wraps decision path text, decision path image and other information in an HTML document 
# and displays it in the users browser
def wrapOutputInHTML(title, body, image):
    
    now = datetime.datetime.today().strftime("%d/%m/%Y - %H:%M:%S")

    filename = title + '.html'
    f = open(filename,'w')
    
    # set style for image
    style = "height:75%;"
    
    wrapper = """<html>
    <head>
    <title>%s - %s</title>
    </head>
    <body><p>%s</p>
    <a href="decision_path_1.png" download></body>
    <img src=%s style=%s>
    </html>"""

    whole = wrapper % (title, now, body, image, style)
    f.write(whole)
    f.close()

    path = "file://" + str(Path().absolute()) + "/%s"
    path = path % (filename)

    return path
    
# Load decision tree classifier, test data and associated reviews

###
# SET PATH FOR CLASSIFIER AND TEST DATA HERE
path = "/Users/artembutbaev/OneDrive/University of Bath 20-21 " + \
"(Year 4)/CM - Individual Project/2. Code/Model Building/"
###

# load decision tree classifier
with open(path + 'dt_final.pkl', 'rb') as f:
    dt = pickle.load(f)

# load test data
with open(path + 'x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
    
# load associated reviews with test data
with open(path + 'df_test.pkl', 'rb') as f:
    df_test = pickle.load(f)
    

# Main user input loop - user selects a review for which to show its decision path

while(True):

    print("Enter a review index from 0 to", len(x_test), ":")
    user_input = input()
    index_chosen = int(user_input)

    review_text = df_test["reviewContent"].iloc[index_chosen]
    print("Is this the review you want to classify? (y/n)\n")
    print(review_text)
    user_input_2 = input()

    if(user_input_2 == 'y'):
        print("Generating output...")
        # get review sample from test data based on chosen index
        sample = x_test.iloc[index_chosen:index_chosen+1]
        
        # 1. get review text for chosen index
        review_text = df_test["reviewContent"].iloc[index_chosen]

        # 2. get decision path (text)
        decisions_made, predicted_string = get_decision_path_text(dt, x_test.columns, sample)

        # 3. get decision path (visual - save as png)
        create_decision_path_png(dt, x_test.columns, 'decision_path_1.png', sample)

        # 4. embed review text/decision path/visual in HTML document and open in browser
        string_to_wrap = prepare_string_to_wrap(index_chosen, review_text, decisions_made, predicted_string)
        image_to_wrap = "decision_path_1.png"
        path = wrapOutputInHTML("output1", string_to_wrap, image_to_wrap)
        webbrowser.open_new_tab(path)
        
        break