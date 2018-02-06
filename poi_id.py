#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
#sys.path.append("/Users/NabeelaMerchant/Documents/GitHub/ud120-projects/tools/") #fix this
from feature_format import featureFormat, targetFeatureSplit
sys.path.append("../final_project/")
#sys.path.append("/Users/NabeelaMerchant/Documents/GitHub/ud120-projects/final_project/") #fix this
from tester import dump_classifier_and_data, test_classifier
import numpy
import matplotlib.pyplot as plt



'''Functions'''
def poi_feat_split(data): #separates poi from remaining features in data
    poi = data[:][:,0] #first item in feature_list, and thus data, is poi
    feat = data[:][:,1:] #remaining items are features
    return poi, feat

def plot_features(poi, feat): #plots each feature against the other
    for feat1 in range(0,len(feat[0])): #x
        for feat2 in range(0,len(feat[0])): #y
            if feat1 != feat2: #if the features are the same don't plot them
                for ii, pp in enumerate(poi):
                    f1 = feat[ii][feat1] #define x -> f1
                    f2 = feat[ii][feat2] #define y -> f2
                    if pp: #if poi
                        plt.scatter(f1,f2,color='r') #plot in red
                    else: #if not poi
                        plt.scatter(f1,f2,color='b') #plot in blue
                plt.xlabel(features_list[feat1+1]) #add one to account for poi in features_list
                plt.ylabel(features_list[feat2+1]) #add one to account for poi in features_list
                plt.show()
                
def computeFraction(poi_messages, all_messages): #calculates fraction of messages to/from poi
    fraction = 0.
    if not numpy.isnan(float(poi_messages)) or not numpy.isnan(float(all_messages)):
        fraction = float(poi_messages)/float(all_messages)
    return fraction

'''### Task 1: Explore what features you'll use.'''
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# this is the start of an exploration of what features i would like to use.
# actual features used will be listed further down.
features_list = ['poi','salary','total_payments','exercised_stock_options','total_stock_value','expenses','from_this_person_to_poi','from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Explore data_dict
print '\n## Data Exploration ## \n'
# number of people
num_people = len(data_dict)
print 'Number of People:',num_people

# number of features
first_value =  list(data_dict.keys())[0]
num_features = len(data_dict[first_value])
print 'Number of Features:',num_features

# feature names
feature_names = list(data_dict[first_value].keys())
#print feature_names

# example of dictionary values
print 'First Person\'s Features:\n',data_dict[first_value]

# the data has occasional missing values that are represented as NaNs
# I will not be modifying them as the featureFormat function handles the NaNs

# number of POIs and list of names in data_dict (keys)
num_poi = 0
names_list = []

for p in data_dict:
    names_list.append(p)
    if data_dict[p]["poi"]:
        num_poi += 1   
        
num_not_poi = len(data_dict) - num_poi

print 'Number of POIs:',num_poi
print 'Number of non-POIs:',num_not_poi

names_list = sorted(names_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True) #featureFormat should set NaN values to zero

### Explore features
poi, feat = poi_feat_split(data) #split data into poi and remaining features
plot_features(poi, feat) # visualize all features against each other

# based on the results there seems to be one individual with a very high 
# salary, exercised stock options, total stock value, total payments, and 
# expenses that far exceeds anyone elses.

'''### Task 2: Remove outliers'''
print '\n## Outlier Investigation ## \n'
# finding the major outlier
max_salary = max(data[:][:,1]) #picked salary as the feature to identify the outlier over
print 'Max Salary (Outlier):',max_salary

# looking at the max_salary of $26,704,229 and the financial data in the pdf,
# i realised that this was the 'total' row being included in the dataset.
# since it is not a real person, I will remove it from the dataset.

# flattening the dataset to sort it and search it by max_salary 
# to find the index of the outlier
if max_salary == 26704229.0: #prevents other values being deleted as I iterate over the code
    flat = data[:][:,1].flatten()
    flat.sort()
    #print 'Max Salary:', flat[-1], '(last value in sorted index)' #max_salary
    max_salary_index = numpy.where(data[:][:,1]==flat[-1]) #index of max_salary in data
    print 'Index of Max Salary (Outlier):',max_salary_index[0][0]
    print 'Associated Feature Data (Outlier):',data[max_salary_index]
    
    # removes data point that contains max_salary value
    data = numpy.delete(data,max_salary_index,0) 

''' Feature Exploration '''
print '\n## Feature Exploration ## \n'

# re-plot data to see patterns without the outlier
poi, feat = poi_feat_split(data) #split data into poi and remaining features
plot_features(poi, feat) # visualize all features against each other

# taking a look at the results, there appears to be a correlation between 
# exercised_stock_options and total_stock_value. 
# i'm going to investigate this further:

corr_coef = numpy.corrcoef(feat[:][:,2], feat[:][:,3])[0,1]
print 'Pearson\'s r between exercised_stock_options and total_stock_value:',corr_coef

# numpy's correlation coefficient between exercised_stock_options and
# total_stock_value is 0.9638. since they are so highly correlated, i'm going
# to remove one of them (total_stock_value)

'''### Task 3: Create new feature(s)'''

# i'm now going to explore from_poi_to_this_person and from_this_person_to_poi
# in more detail

data_dict_keys = data_dict.keys()

# creating 2 new features that are fractional representations of the messages
# from a poi and to a poi
for keys in data_dict_keys:
    to_poi =  data_dict[keys]['from_this_person_to_poi']
    from_messages =  data_dict[keys]['from_messages']
    frac_to_poi = computeFraction(to_poi,from_messages)
    #print frac_to_poi
    data_dict[keys]['frac_to_poi'] = frac_to_poi
    
    from_poi =  data_dict[keys]['from_poi_to_this_person']
    to_messages =  data_dict[keys]['to_messages']
    frac_from_poi = computeFraction(from_poi,to_messages)
    data_dict[keys]['frac_from_poi'] = frac_from_poi

#print data_dict[data_dict_keys[0]]

data_dict_values = data_dict.values()    
# extracting feature values into a list to help plot
frac_to_poi_list = [float(data_dict_values[i]['frac_to_poi']) for i in range(len(data_dict_values))]
frac_from_poi_list = [float(data_dict_values[i]['frac_from_poi']) for i in range(len(data_dict_values))]

# plotting the two new features against each other
for ii, pp in enumerate(poi):
    f1 = frac_from_poi_list[ii] #define x -> f1
    f2 = frac_to_poi_list[ii] #define y -> f2
    if pp: #if poi
        plt.scatter(f1,f2,color='r') #plot in red
    else: #if not poi
        plt.scatter(f1,f2,color='b') #plot in blue
plt.xlabel('fraction of emails from poi') #add one to account for poi in features_list
plt.ylabel('fraction of emails to poi') #add one to account for poi in features_list
plt.show()

# from the plot we can see that there are no clear clusters between poi and
# non-poi, however, there are sections in the graph where no poi exist. This
# might be useful distinguishing factor for the algorithm and so i'll swap
# the absolute email values for the fractional ones instead.


''' Actual Feature Selection '''
features_list = ['poi','salary','total_payments','exercised_stock_options','expenses','frac_to_poi','frac_from_poi'] # You will need to use more features

# running this list of features with the GaussianNB classifier resulted in a
# poor outcome with a high recall and poor precision --> too many false 
# positives.. going to try some feature selection to improve it and then 
# modify the algorithm and the parameters to see how it can be improved.

### Store to my_dataset for easy export below.
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True) #featureFormat should set NaN values to zero
labels, features = targetFeatureSplit(data) # Split data into features and labels

#univariate selection using selectkbest
from sklearn.feature_selection import SelectKBest
selector = SelectKBest()
k = [1,2,3,4,5,6]
#selector.fit(features_train, labels_train)
#features_train_transformed = selector.transform(features_train)

#print features_train_transformed
#print features_train
'''### Task 4: Try a variety of classifiers'''
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.pipeline import Pipeline

#from sklearn.naive_bayes import GaussianNB
#parameters ={}
#base_clf = GaussianNB() #<<terrible results

#from sklearn.svm import SVC
#parameters ={}
#base_clf = SVC(kernel="linear") #<< too slow! couldn't get a test result even with 1 feature

from sklearn.tree import DecisionTreeClassifier 
min_samples_split=[2,3,4,5]
base_clf = DecisionTreeClassifier()

#from sklearn.ensemble import RandomForestClassifier 
#max_depth = [2,3,4]
#base_clf = RandomForestClassifier()

pipeline = Pipeline(steps=[("skb",selector),("clf",base_clf)])
param_grid = dict(skb__k=k,clf__min_samples_split=min_samples_split) #clf__min_samples_split=min_samples_split #clf__max_depth=max_depth
                  
'''### Task 5: Tune your classifier to achieve better than .3 precision and recall''' 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.metrics import precision_score    
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print '\nTraining Algorithm...'

### Cross Validation: Train Test Split
# splitting the data over here to test the other cross-validation parameters
# for my own curiosity
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Cross Validation: GridSearchCV with KFold cross validation
from sklearn import grid_search

folds = 13 #for cv in gridsearchcv

clf_test = grid_search.GridSearchCV(pipeline, param_grid=param_grid, n_jobs = 1, cv = folds, scoring = 'f1')
clf_test.fit(features_train,labels_train)
pred = clf_test.predict(features_test)

print '...Algorithm has been trained\n'

print clf_test.best_score_
print clf_test.best_params_
print 'F1 score:',f1_score(pred,labels_test)
print 'Precision:',precision_score(pred,labels_test)
print 'Recall:',recall_score(pred,labels_test)

# creating a copy of the classifier to identify the importance of the features
selector_copy = SelectKBest(k=clf_test.best_params_['skb__k'])
selector_copy.fit(features_test,labels_test)
print selector_copy.scores_
features_test_transformed = selector_copy.transform(features_test)
features_train_transformed = selector_copy.transform(features_train)
'''
clf_copy = DecisionTreeClassifier(min_samples_split=clf_test.best_params_['clf__min_samples_split'])
clf_copy.fit(features_train_transformed,labels_train)
print clf_copy.feature_importances_
'''
### Creating the classfier based on the ideal parameters chosen by gridsearchcv to pass to the tester
selector_final = SelectKBest(k=clf_test.best_params_['skb__k'])
#base_clf_final = RandomForestClassifier(max_depth=clf_test.best_params_['clf__max_depth'])
base_clf_final = DecisionTreeClassifier(min_samples_split=clf_test.best_params_['clf__min_samples_split'])
#base_clf_final = GaussianNB()
clf = Pipeline(steps=[("skb",selector_final),("clf",base_clf_final)])


'''### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.'''

dump_classifier_and_data(clf, my_dataset, features_list)
print '\nData dumped!'