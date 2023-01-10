Author: Artem Butbaev

Date: 30 April 2021

--------------------------------------------------------------------
This code and PDF document (dissertation) was submitted by the author as part of their University dissertation "Detection of fake reviews using a transparent machine learning approach" submitted in 2021.

--------------------------------------------------------------------

This is a brief readme document outlining how to run the code files attached with the dissertation and how to access the Yelp dataset used.

It is recommended to run files 1-3 in the Jupyter Notebook environment, as this was where the files were originally developed, tested and run.

Files 4 (the User Interface) can be run in either Jupyter or in the terminal - both versions have been tested. The classifier and test data has been included for this file, so files 1-3 do not need to be run to test the front end.

! It is important to note that for all files, including Files 4, the paths in the code need to be updated before running the code ! 

For example, in file 4 the path to the classifier and test data need to be updated to the correct folder on the reader's machine.

--------------------------------------------------------------------

For accessing the Yelp dataset used in this dissertation:

Hyperlink: http://liu.cs.uic.edu/download/yelp_filter/
Password needs to be requested from Professor Bing Liu, University of Chicago, the author of the dataset. 
Please note that the dataset is very large, at a size of almost 1GB, hence not being included in the code files.

--------------------------------------------------------------------

Files included in respective folders:


1. data_preparation

	- Notebook 1 - Data Processing.ipynb
	- data_preparation.py

2. feature_engineering

	- Notebook 2 - Feature Engineering.ipynb
	- feature_engineering.py

3. model_evaluation

	- Notebook 3 - Model Evaluation.ipynb
	- model_evaluation.py

4. front_end

	- Notebook 4a - User Interface v1.0.ipynb
	- Notebook 4b - User Interface v2.0.ipynb
	- ui_v1.0.py
	- ui_v2.0.py
	- dt_final.pkl -> the decision tree classifier
	- x_test.pkl -> the test data (features only)
	- df_test.pkl -> associated data for test data (such as the review text)
