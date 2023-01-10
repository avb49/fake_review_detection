# Import required libraries

# pickle will be used to load the saved classifier and 
# test data to be used for the front end
import pickle
# scikit-learn will be used for getting the decision path
# of samples and generating the tree diagram
from sklearn import tree
# pydotplus will be used to create an interface to the 
# DOT language which will be used to modify the tree diagram
import pydotplus
# webbrowser will be used to open a new tab with the 
# content of the HTML file created
import webbrowser
# import datetime - for including the current date 
# and time in the HTML file generated
import datetime
# import functionality to get the current directory path
from pathlib import Path

####################################################
####################################################


# Define functions to be used for generating the front end

# function 1 - generates for a sample review the decision path information
# inspiration from: 
# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
def get_decision_path_info(classifier, feature_names, sample):

    # retrieve probabilities for classification for sample
    probabilities = classifier.predict_proba(sample)
    # retrieve decision paths for test sample
    node_indicator = classifier.decision_path(sample)
    # retrieve leaf IDs reached by test sample
    leaf_identifier = classifier.apply(sample)
    # get a list of all thresholds (of all nodes in tree)
    threshold = classifier.tree_.threshold
    # get node id of leaf that the sample "lands on"
    leaf_id = leaf_identifier[0]
    # obtain ids of the nodes sample goes through
    nodes_traversed = node_indicator.indices[node_indicator.indptr[0]:
                                    node_indicator.indptr[1]]

    features_used = []
    threshold_values = [] 
    threshold_signs = []

    for node_id in nodes_traversed:
        
        # break for loop
        if(node_id == leaf_id):
            break

        # check if value of the split feature for sample i is below threshold
        if (sample.iloc[0][classifier.tree_.feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
            
        features_used.append(feature_names[classifier.tree_.feature[node_id]])
        #thresholds.append(threshold_sign + " " + str(round(threshold[node_id], 3)))
        threshold_values.append(round(threshold[node_id], 3))
        threshold_signs.append(threshold_sign)

    # get proportions of labels at leaf node and calculate probabilities of the output
    class_proportion = classifier.tree_.value[leaf_id]
    class_probability = round(100 * max(probabilities[0]), 1)
    samples_at_leaf = int(class_proportion[0][0]) + int(class_proportion[0][1])
    genuine_probability = round(100 * probabilities[0][0], 1)
    fake_probability = round(100 * probabilities[0][1], 1)

    predicted_string = "According to the model, this review is likely to be Genuine " \
    "with probability {prob1}% and Fake with probability {prob2}%. " \
    .format(prob1 = genuine_probability, prob2 = fake_probability)
    predicted_string += "These probabilities are based on " + str(samples_at_leaf) + \
    " other reviews with the same decision path."
    
    return features_used, threshold_values, threshold_signs, predicted_string

# function 2 - this function writes the series of decisions in a natural language form
# based on the features used and their associated thresholds
# in the decision path
# returns the series of decisions and the 
# features not used in the decision path
def write_decisions(features_used, feature_names, threshold_values, threshold_signs):
    
    unique_features = []
    features_not_used = []
    list_of_statements = []
    
    # create list of unique features (in the order they appear in decision path) 
    # (not efficient but our lists will be very small)
    for feature in features_used:
        if(feature not in unique_features):
            unique_features.append(feature)
            
    # create list of features not used in decision path
    # (not efficient but our lists will be very small)
    for feature in feature_names:
        if(feature not in unique_features):
            features_not_used.append(feature)

    # create a dict with all unique features and the number of times
    # they appear in the decision path
    feature_counts = dict((element,0) for element in unique_features)

    # loop through each decision
    for index in range(len(features_used)):
        
        # update feature count in dict
        feature_counts[features_used[index]] += 1
        
        # handle case when feature is 'hasProfile'
        if(features_used[index] == "Does the reviewer have a profile?"):
            if(threshold_signs[index] == ">"):
                list_of_statements.append("The reviewer has a profile")
            elif(threshold_signs[index] == "<="):
                list_of_statements.append("The reviewer does not have a profile")
            continue
        # handle the cases where the threshold should 
        # be rounded to an integer
        # i.e., when the feature is not max_tfidf or 'hasProfile'
        elif(features_used[index] != "Maximum word importance score (TF-IDF)"):
            
            # if threshold sign is '>', add 0.5 to the value and change sign to '>='
            if(threshold_signs[index] == ">"):
                threshold_values[index] += 0.5
                threshold_signs[index] = ">="
            # if threshold sign is '<=', subtract 0.5 from the value and keep the same sign
            elif(threshold_signs[index] == "<="):
                threshold_values[index] -= 0.5
                
            # if threshold sign is '<=' and value is 0, update sign to be '='
            if(threshold_signs[index] == "<=" and threshold_values[index] == 0):
                threshold_signs[index] = "="

        # create statements based on feature counts
        if(feature_counts[features_used[index]] == 1):
            list_of_statements.append(features_used[index] \
                                      + " " + \
                                      threshold_signs[index] \
                                      + " " + str(int(threshold_values[index])))
        else:
            list_of_statements[unique_features.index(features_used[index])] \
            += " and " + threshold_signs[index] + " " + str(int(threshold_values[index]))

    return list_of_statements, features_not_used

# function 3 - creates a diagram of the decision tree with the decision path 
# for a given review highlighted and saves it as a png
def create_tree_diagram_png(dt, feature_names, class_names, filename, sample):
    
    # get decision tree classifier in DOT (graphviz) format
    dot_data = tree.export_graphviz(dt, out_file=None, 
                                feature_names=feature_names, 
                                    class_names=class_names, filled=True, 
                                    rounded=True, special_characters=True, 
                                    node_ids = False)
    
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    # set all nodes to be of a white colour
    for node in graph.get_node_list():
        node.set_fillcolor('white')
        
    # set colour of leaf nodes depending on their class (fake or genuine)
    for node in graph.get_node_list():
        d = node.get_attributes()
        if("label" in d):
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if(label.startswith('class = ')):
                    # colour nodes depending on their class
                    if('Fake' in label):
                        # set colour of nodes with fake as majority vote
                        node.set_fillcolor('gray75')
                    elif('Genuine' in label):
                        # set colour of nodes with genuine as majority vote
                        node.set_fillcolor('gray97')

    # get decision path for chosen review sample
    decision_path = dt.decision_path(sample)

    # loop through each node in tree and its associated decision-path value, 
    # i.e., if the node is part of the decision path 
    # then it has a value of 1, otherwise, it has a value of 0
    for node_counter, node_value in enumerate(decision_path.toarray()[0]):
        
        # if node is not part of decision path for sample, continue 
        if(node_value == 0):
            continue
        # set colour of node part of decision path for sample
        dot_node = graph.get_node(str(node_counter))[0]
        dot_node.set_fillcolor('darkseagreen1')
        
    # export updated graph to a png
    graph.write_png(filename)

# function 4 - prepares the string to wrap in the 
# HTML document to be displayed to the user
def prepare_string_to_wrap(index, review_text, reviewer_id, 
                           decisions_made, features_not_used, predicted_string):
    
    string_to_wrap = ""

    # add title
    index_string = "<h2>Classification Explanation For Selected Review</h2>"
    string_to_wrap += index_string
    string_to_wrap += "<br/>"
    
    # add review text
    string_to_wrap += "<em>" + review_text + "</em>"
    string_to_wrap += "<br/>"
    
    # add reviewer ID
    string_to_wrap += "- posted by reviewer: " + reviewer_id
    string_to_wrap += "<br/><br/>"
    
    # add classification and accuracy to string
    string_to_wrap += "<mark>" + predicted_string + "</mark>"
    string_to_wrap += "<br/>"
    
    # add line to separate classification from decisions made
    string_to_wrap += "<hr>"
    
    # add decisions made to string
    string_to_wrap += "<br/>"
    string_to_wrap += "This estimation is based on the model's " + \
    "consideration of the following attributes of the review and the reviewer:<br/>"
    string_to_wrap += "<ul>"
    for decision in decisions_made:
        string_to_wrap += "<li>"
        string_to_wrap += decision
        string_to_wrap += "</li>"
    string_to_wrap += "</ul>"
    #string_to_wrap += "<br/>"
    
    # add info about features NOT considered in decisions
    string_to_wrap += "The following features were not " + \
    "considered in the decision-making process for this review:"
    string_to_wrap += "<ul>"
    for feature in features_not_used:
        string_to_wrap += "<li>"
        string_to_wrap += feature
        string_to_wrap += "</li>"
    string_to_wrap += "</ul>"
        
    # add information about visualisation
    string_to_wrap += "This reasoning can also be " + \
    "seen visually in the visualisation below."
    
    return string_to_wrap

# function 5 - wraps decision path text, decision path image 
# and other information in an HTML document and displays it in the users browser
def wrap_output_in_HTML(title, text, image):
    
    current_datetime = datetime.datetime.today().strftime("%d/%m/%Y - %H:%M:%S")

    # create a new HTML file
    filename = title + '.html'
    f = open(filename,'w')
    
    # create basic HTML structure with indications
    # of where strings will be embedded
    wrapper = """<html>
    <head>
    <style>
    body {
        background-color: WhiteSmoke;
    }
    </style>
    <title>%s - %s</title>
    </head>
    <body>
    
    <p>%s</p>
    <a href=%s target="_blank">Click here to open visualisation</a>
    </body>
    </html>"""

    # embed the document title, current datetime, decision path text
    # and tree visualisation file location in the HTML document
    whole = wrapper % (title, current_datetime, text, image)
    f.write(whole)
    f.close()
    
    path = "file://" + str(Path().absolute()) + "/%s"
    path = path % (filename)
    
    return path

####################################################
####################################################


# Main code with user input loop

###
# SET PATH FOR CLASSIFIER AND TEST DATA HERE
path = "/Users/artembutbaev/OneDrive/University of Bath 20-21 " + \
"(Year 4)/CM - Individual Project/2. Code/Model Building/"
###

# load decision tree classifier from path above
with open(path + 'dt_final.pkl', 'rb') as f:
    dt_final = pickle.load(f)
# load test data (features only) from path above
with open(path + 'x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
# load associated review data from path above
with open(path + 'df_test.pkl', 'rb') as f:
    df_test = pickle.load(f)

# set feature names to be displayed in front end
feature_names_readable = ['Does the reviewer have a profile?', 'Posts by reviewer', \
                          '\'Useful\' votes', 'Maximum word importance score (TF-IDF)', \
                          'Length of review (characters)', '\'Cool\' votes', \
                          '\'Funny\' votes', 'Maximum posts by reviewer in a single day', \
                          'Count of numbers in review', 'Count of symbols in review']
# set output class names to be displayed in front end 
class_names_readable = ['Genuine', 'Fake']

# User input loop - user selects a review they would like classified and explained
# an explanation and tree diagram is generated, embedded within an HTML document
# and displayed in the user's web-browser in a new tab
while(True):

    print("Enter a review index from 0 to", len(x_test) - 1, ":")
    user_input = input()
    index_chosen = int(user_input)
    review_text = df_test["reviewContent"].iloc[index_chosen]
    print("Is this the review you want to classify? (y/n)\n")
    print(review_text)
    user_input_2 = input()

    if(user_input_2 == 'y'):
        print("Generating output...")
        
        # 1. get review sample from test data for chosen index
        # and review text and reviewer ID from review data
        sample = x_test.iloc[index_chosen:index_chosen+1]
        review_text = df_test["reviewContent"].iloc[index_chosen]
        reviewer_id = df_test["reviewerID"].iloc[index_chosen]

        # 2. get decision path information (features used and the threshold values)
        features_used, threshold_values, threshold_signs, predicted_string = \
        get_decision_path_info(dt_final, feature_names_readable, sample)
        # write the decision path information 
        # as a series of decisions in text form, detailing
        # the features used and not used in the model's decision making
        decisions_made, not_used = \
        write_decisions(features_used, feature_names_readable, threshold_values, threshold_signs)

        # 3. generate tree diagram with highlighted decision path (save as png)
        create_tree_diagram_png(dt_final, x_test.columns, \
                                 class_names_readable, 'decision_path_1.png', sample)

        # 4. prepare the written explanation to wrap in the HTML document
        string_to_wrap = prepare_string_to_wrap(index_chosen, review_text, reviewer_id, 
                                                decisions_made, not_used, predicted_string)
        image_to_wrap = "decision_path_1.png"
        
        # 5. embed the written explanation and tree diagram in HTML document
        path = wrap_output_in_HTML("output1", string_to_wrap, image_to_wrap)
        # open the prepared HTML document in the user's web browser
        webbrowser.open_new_tab(path)
        
        break