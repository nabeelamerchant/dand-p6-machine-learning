# dand-p6-machine-learning
Enron fraud indentification machine learning project from Udacity's Data Analyst Nanodegree

<h4>Summary</h4>
 
The intention of the project is to create a machine learning algorithm that can identify ‘persons of interest’ (POIs) from a list of people who worked at Enron. POIs are people that were involved with the fraud at Enron to some degree. The dataset provided for this project was labeled and contained 18 identified POIs. As a result a supervised learning algorithm was used to train the dataset.

The dataset contained financial and email information about each person. The original dataset contained 146 data points and referenced 21 features. This included features such as people’s salary, exercised stock options, and how many emails they sent and received from POIs. 

<h4>Features</h4>

I included 6 features ('salary', 'total_payments', 'exercised_stock_options', 'expenses', 'frac_to_poi', 'frac_from_poi') in feature_list, which were then passed through the SelectKBest feature selection function. Two of the features in the feature_list, ‘frac_to_poi’ and ‘frac_from_poi’, were generated from existing features. They represent the fraction of emails send to and from POIs to the person, respectively. The new features were a means of checking the relative number of emails that the person sent or received from POIs, instead of the absolute values presented.

No feature scaling was performed as the algorithms used (DecisionTreeClassifier and RandomForestClassifier) are invariant to scaling, unlike SVMs. The feature scores for the SelectKBest automatic future selection are - [10.28104113 9.21965362 26.43169945 0.78313337 7.40448556 1.07200143] respectively. The parameter, k, was selected using the GridSearchCV method, with options from 1 - 6 provided. The GridSearchCV method returned k = 5 as the best parameter for SelectKBest. As a result, the ‘expenses’ feature was not used.

<h4>Parameter Tuning</h4>

I used GridSearchCV to automatically tune the parameters for the algorithm. For the Decision Tree I provided a list of parameters ranging from 2 – 5 to tune min_samples_split. The tuned value came to 2. For Random Forest, which was another classifier I was testing, I tested the parameter ‘max_depth’ between values of 2 – 4, and 4 was the optimized value.

<h4>Validation</h4>

I used the cross validation parameter (cv) in GridSearchCV. This inherently uses k-fold cross validation. The number of folds was set to 13.

<h4>Performance</h4>

I used the F1, precision, and recall scores as a means of testing the algorithm.

I split the dataset into a training and testing set. I used the training set to pass the pipeline to GridSearchCV. I then tested the fit classifier with the test dataset and computed the scores on those predictions and test labels. The results for the scores are listed in the table in the summary pdf.

My F1 score, precision, and recall were all 0.4. High precision but poor recall indicates that the algorithm is good at identifying true positives (POIs) and has a higher false positive count. High recall and low precision mean there’s a lower rate of identifying true positives (POIs) and has a higher false positive count. This equivalence for precision and recall in my results leads me to think that the algorithm is just as good at getting false positives as it is to get false negatives versus true positives (POIs). This is further elucidated in the tester.py evaluation where the number of false positives is 1309 and the number of false negatives in 1365.
