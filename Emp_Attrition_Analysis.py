## Import the data from cloud
# pip install pymongo[srv]
import pymongo
import pandas as pd
## Connecting to cloud using the pymongo package
client = pymongo.MongoClient("mongodb+srv://ds_project:ds_project@cluster0.h8ffl.mongodb.net/Project?retryWrites=true&w=majority")

db = client.Project         ## Connecting to the database
print(db)

collection = db.Emp_Details_Updated      ## Extracting the data from database

## Finding the memory location
data = collection.find()
print(data)

## Converting the data into dataframe
attrition1 = pd.DataFrame(list(data))

########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pylab
## attrition = pd.read_excel("C:/Users/Dell/Desktop/Assighnments/Project/Attrition Rate-Dataset.xlsx")
attrition1.shape              ## Shape of the data
attrition1.columns           ## Column names of the data

## Deleting the unwanted data
attrition = attrition1.drop(['_id'], axis = 1)
attrition = attrition.drop(['EmployeeID'], axis = 1)
attrition = attrition.drop(["EmployeeName"], axis = 1)

## Renaming the columns names
attrition.columns = [ "Attrition", "Designation", "Percentage_salary_hike", "Training_in_hrs", "Work_life", "Tenure", "Monthly_salary"]

## Describing the data
data_describe = attrition.describe()  ## Describing the data
attrition.dtypes             ## Data type of the varibles
attrition.isna().sum()       ## Checking the NA values
attrition.isnull().sum()    ## Checking the null values   

### \/\/\/\/\/\/\/\\/\/\/\/\
## EDA
## Value count for Attrition
attrition.Attrition.value_counts()
## Count plot for Attrition
sns.countplot(attrition.Attrition)

## Value count for Desgnatio
attrition.Designation.value_counts()
## Count plot for Desgnetion
sns.countplot(attrition.Designation)

## Value count for Work life
attrition.Work_life.value_counts()
## Count for Worklife
sns.countplot(attrition.Work_life)

## Value count for Monthly_salary
attrition.Training_in_hrs.value_counts()
## Count for Monthly_salary
sns.countplot(attrition.Training_in_hrs)

## Value count for Monthly_salary
attrition.Tenure.value_counts()
## Count for Monthly_salary
sns.countplot(attrition.Tenure)

## Value count for Monthly_salary
attrition.Monthly_salary.value_counts()
## Count for Monthly_salary
sns.countplot(attrition.Monthly_salary)

## Value count for Monthly_salary
attrition.Percentage_salary_hike.value_counts()
## Count for Monthly_salary
sns.countplot(attrition.Percentage_salary_hike)

### First Moment bussiness understanding.
## Mean and Median for attrition dtaa
attrition.mean()
attrition.median()

## Mode for attrition dtaa
attrition.Designation.mode()
attrition.Work_life.mode()

## Variance and Standard deviation
attrition.var()
attrition.std()

## Skewness and kurtossis
attrition.skew()
attrition.kurt()

### \/\/\/\/\/\/\/\/\/\/\/\
## thierd moment of bussiness understanding
## Univariant plots

## Histogram
plt.hist(attrition.Tenure)
plt.hist(attrition.Percentage_salary_hike)
plt.hist(attrition.Work_life)
plt.hist(attrition.Monthly_salary)
plt.hist(attrition.Designation)
plt.hist(attrition.Training_in_hrs)

## Boxplot
plt.boxplot(attrition.Monthly_salary)
plt.boxplot(attrition.Tenure)
plt.boxplot(attrition.Percentage_salary_hike)
plt.boxplot(attrition.Training_in_hrs)

## Data Destribution
stats.probplot(attrition.Percentage_salary_hike, dist="norm",plot=pylab)
stats.probplot(attrition.Training_in_hrs, dist="norm",plot=pylab)
stats.probplot(attrition.Monthly_salary, dist="norm",plot=pylab)

# Normalization function for scaling the attrition.Monthly_salary  data
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
# Normalized data 
attrition.Monthly_salary = norm_func(attrition.Monthly_salary)
attrition.Training_in_hrs = norm_func(attrition.Training_in_hrs)

## labeling the data using the labelencoder for catagorical data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
attrition.Attrition = le.fit_transform(attrition.Attrition)

value = {'Bad': 0,'Good': 1 , 'Best' : 2 }
attrition.Work_life = [value[item] for item in attrition.Work_life]

value1 = {'Software Developer': 0,'Data Engineer': 1 , 'Database Administrator' : 2 , 'Data Administrator' : 3 , 'Data Analyst': 4 , 'Business Analyst': 5 , 'Data Architect': 6 , 'Data Scientist' : 7 ,'Domain Expert': 8 , 'Chief Data Scientist' : 9}
attrition.Designation = [value1[item] for item in attrition.Designation]

## Correlation values between the output and input variables
np.corrcoef(attrition.Attrition, attrition.Monthly_salary)
np.corrcoef(attrition.Attrition, attrition.Percentage_salary_hike)
np.corrcoef(attrition.Attrition, attrition.Tenure)
np.corrcoef(attrition.Attrition, attrition.Work_life)
np.corrcoef(attrition.Attrition, attrition.Training_in_hrs)
np.corrcoef(attrition.Attrition, attrition.Designation)

## Heatmap for the attrition data
sns.heatmap(attrition.corr(), annot = True, fmt = '.0%')

## Dividing the attrition data into input and output data
X_data = attrition.drop(["Attrition"], axis = 1)
Y_data = attrition.Attrition

## Finding the VIF for input variables to checking the Multicolinearity issue
import statsmodels.formula.api as smf
r_des = smf.ols('Designation ~ Percentage_salary_hike + Training_in_hrs + Work_life + Tenure + Monthly_salary', data = X_data).fit().rsquared
vif_des = 1/(1-r_des)

r_per = smf.ols('Percentage_salary_hike ~ Designation  + Training_in_hrs + Work_life + Tenure + Monthly_salary', data = X_data).fit().rsquared
vif_per = 1/(1-r_per)

r_tr = smf.ols('Training_in_hrs ~ Designation  + Percentage_salary_hike + Work_life + Tenure + Monthly_salary', data = X_data).fit().rsquared
vif_tr = 1/(1-r_tr)

r_wrk = smf.ols('Work_life ~ Designation  + Percentage_salary_hike + Training_in_hrs + Tenure + Monthly_salary', data = X_data).fit().rsquared
vif_wrk = 1/(1-r_wrk)

r_tnr = smf.ols('Tenure ~ Designation  + Percentage_salary_hike + Training_in_hrs + Work_life + Monthly_salary', data = X_data).fit().rsquared
vif_tnr = 1/(1-r_tnr)

r_slr = smf.ols('Monthly_salary ~ Designation  + Percentage_salary_hike + Training_in_hrs + Work_life + Tenure', data = X_data).fit().rsquared
vif_slr = 1/(1-r_slr)

d1 = {"variables": ["VIF_desgnation", "VIF_Hike", "VIF_training", "VIF_wrk_life", "VIF_Tnr", "VIF_slr"], "VIF Values": [vif_des, vif_per, vif_tr, vif_wrk, vif_tnr, vif_slr]}
VIF = pd.DataFrame(d1)
VIF
###############

### Over Sampling technic
from imblearn.over_sampling import SMOTE
over_Sampling = SMOTE(0.75)
X_data_res, Y_data_res = over_Sampling.fit_resample(X_data, Y_data)

## Value count and countplot before applying the SMOTE
Y_data.value_counts()
sns.countplot(Y_data)

## Value count and countplot after applying the SMOTE
Y_data_res.value_counts()
sns.countplot(Y_data_res)

## Splitting thee data into train and test data.
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X_data, Y_data, test_size = 0.30, random_state = 0)

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//\\//\/\\/\\\/\/\
## Decission tree
## Building the  Decission tree model
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')

## Model fit on train data
model.fit(train_x, train_y)

## Model predict on test data
pred_test = model.predict(test_x)

## Plot the Decition tree
## conda install python-graphviz
from IPython.display import Image
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.tree import DecisionTreeClassifier,export_graphviz

dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("decisiontree.png")

## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test = confusion_matrix(pred_test, test_y)
cm_test

## Accuracy for train data
acc_test_D = np.mean(pred_test == test_y)
acc_test_D

## Sencitivity(True positive rate)
Sensitivity = cm_test[0,0]/(cm_test[0,0] + cm_test[0,1])
print('sensitivity:', Sensitivity)

## Specificity(True nagative)
specitivity = cm_test[1,1]/(cm_test[1,0] + cm_test[1,1])
print('Specificity:', specitivity)

## Model predict on train data
pred_train = model.predict(train_x)

## Tree plot
dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("decisiontree.png")

## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_train = confusion_matrix(pred_train, train_y)
cm_train

## Accuracy for train data
acc_train_D = np.mean(pred_train == train_y)
acc_train_D

## Sencityvity(True positive rate)
sensitivity_train = cm_train[0,0]/(cm_train[0,0] + cm_train[0,1])
print('sensitivity_train:', sensitivity_train)

## Specityvity(True Nagetive)
Specificity_train = cm_train[1,1]/cm_train[1,1] + cm_train[1,0]
print('Specificity_train:', Specificity_train)

# selecting feature importance
feature_imp = pd.Series(model.feature_importances_, index = X_data.columns).sort_values(ascending=False)
feature_imp
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#### pruning \/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 110, num = 11)]
max_depth.append(None)

max_leaf_nodes = [2, 5, 10, 20,30, 40]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 5, 7]

## Prouning
from sklearn import tree
dtree = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'best', max_leaf_nodes = 10, min_samples_leaf = 3, max_depth= 5)
dtree.fit(train_x, train_y)

## Model predict on test data
pred_test = dtree.predict(test_x)

## Confusion matrics
pd.crosstab(pred_test, test_y)

## Accuracy for test sata
accuracy_test_pr = np.mean(pred_test == test_y)
accuracy_test_pr

## Model predict on the train data
pred_train = dtree.predict(train_x)

## Confusion matrics
pd.crosstab(pred_train, train_y)

## Accuracy for train data
acc_train_pr = np.mean(pred_train == train_y)
acc_train_pr

## #### Grid search on decision tre
#### Grid search on decision tree #####\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//
from sklearn.model_selection import GridSearchCV

criterion = ['gini', 'entropy']
max_depth = [1,3,5,7,9, 11,None]
splitter = ['best', 'splitter']

grid = GridSearchCV(estimator = model , param_grid = dict(criterion = criterion, max_depth = max_depth, splitter = splitter ))
                                                               
grid.fit(train_x, train_y)

print (grid.best_score_)
print (grid.best_params_)


#building the model based on the optimised parameters
model = DT(criterion = 'gini', max_depth = 5, splitter = 'best' )


## Model fit on train data
model.fit(train_x, train_y)

## Model predict on test data
pred_test = model.predict(test_x)

## Plot the Decition tree
from sklearn import tree
dt_dia = tree.DecisionTreeClassifier(random_state = 0)
dt_dia_t = dt_dia.fit(test_x, test_y)
tree.plot_tree(dt_dia_t)

## Confusion matrics
pd.crosstab(pred_test, test_y)

## Accuracy for test sata
accuracy_test_gr = np.mean(pred_test == test_y)
accuracy_test_gr 

## Model predict on the train data
pred_train = model.predict(train_x)

## Plot the Decition tree
from sklearn import tree
dt_dia = tree.DecisionTreeClassifier(random_state = 0)
dt_dia_tr = dt_dia.fit(train_x, train_y)
tree.plot_tree(dt_dia_t)

## Confusion matrics
pd.crosstab(pred_train, train_y)

## Accuracy for train data
acc_train_gr = np.mean(pred_train == train_y)
acc_train_gr

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//\\//\/\\/\\\/\/\
### Random forest

## Buielding the model using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2, n_estimators = 617, min_samples_split = 7, min_samples_leaf = 4, max_features = None, max_depth = None, criterion = 'entropy')
## model fitting on the train data
rf.fit(train_x, train_y)

## Predicting on the test data
pred_test_rf = rf.predict(test_x)

## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test_rf = confusion_matrix(pred_test_rf, test_y)
cm_test_rf

## Sencitivity(True positive rate)
Sensitivity = cm_test_rf[0,0]/(cm_test_rf[0,0] + cm_test_rf[0,1])
print('sensitivity:', Sensitivity)

## Specificity(True nagative)
specitivity = cm_test_rf[1,1]/(cm_test_rf[1,0] + cm_test_rf[1,1])
print('Specificity:', specitivity)

## Accuracy for test data
accuracy_test_RF = np.mean(pred_test_rf == test_y)
accuracy_test_RF

# from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report
fpr, tpr, thresholds = roc_curve(test_y, pred_test_rf)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr or ROC Curve
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])


## AUC curve
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
plt.plot(fpr,tpr,label="data 1, auc="+str(roc_auc))
plt.legend(loc=4)
plt.show()

## Model predict on train data
pred_train_rf = rf.predict(train_x)

## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_train_rf = confusion_matrix(pred_train_rf, train_y)
cm_train_rf

## Accuracy for train data
accuracy_train_RF = np.mean(pred_train_rf == train_y)
accuracy_train_RF

## Sencitivity(True positive rate)
Sensitivity_train = cm_train_rf[0,0]/(cm_train_rf[0,0] + cm_train_rf[0,1])
print('sensitivity_train:', Sensitivity_train)

## Specificity(True nagative)
specitivity_train = cm_train_rf[1,1]/(cm_test_rf[1,0] + cm_test_rf[1,1])
print('Specificity_train:', specitivity_train)

# selecting feature importance
feature_imp = pd.Series(rf.feature_importances_, index = X_data.columns).sort_values(ascending=False)
feature_imp
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Variable Important Features")
plt.legend()
plt.show()


#### Grid search on Random forest################
import numpy as np 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
random_search = {'criterion': ['entropy', 'gini'],
               'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],
               'max_features': ['auto', 'sqrt','log2', None],
               'min_samples_leaf': [4, 6, 8, 12],
               'min_samples_split': [5, 7, 10, 14],
               'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}

clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 80, 
                               cv = 4, verbose= 5, random_state= 101, n_jobs = -1)


model.fit(train_x, train_y)

print (model.best_score_)
print (model.best_params_)

## Bulding randam forest with tuning parameters
rf_tune = RandomForestClassifier(max_depth=935, n_estimators=267, criterion="gini", min_samples_leaf=4, min_samples_split=10, max_features = None)

## model fitting on the train data
rf_tune.fit(train_x, train_y)

## Predicting on the test data
pred_test1 = rf_tune.predict(test_x)

## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test_rf = confusion_matrix(pred_test1, test_y)

## Test accuracy
acc_test_tune = accuracy_score(test_y,pred_test1) 
acc_test_tune

## Model predict on train data
pred_train1 = rf_tune.predict(train_x)

## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_train_rf = confusion_matrix(pred_train1, train_y)

## Accuracy for train data
RF_train_tune = accuracy_score(train_y,pred_train1)
RF_train_tune

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//\\//\/\\/\\\/\/\
#### Stacking #####################
import warnings
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection

Dtree_st = DecisionTreeClassifier(max_depth=4, random_state=1)
Log_st = LogisticRegression()
Nb_st = GaussianNB()
Rf_st = RandomForestClassifier(random_state=1)

sclf = StackingClassifier(classifiers=[Dtree_st, Log_st, Nb_st], use_probas=True, meta_classifier=Rf_st)
                   
print('3-fold cross validation:\n')

for clf, label in zip([Dtree_st, Log_st, Nb_st, sclf], 
                      ['Decision Tree', 
                       'Logistic', 
                       'Naive Bayes',
                       'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, X_data, Y_data, cv=3, scoring='f1_macro')
                                              
print("F1 Scores: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//\\//\/\\/\\\/\/\

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(learning_rate = 0.3, n_estimators = 5000)

## Fitting on train data
ada_clf.fit(train_x, train_y)

## Predicting on test data
test_pred_ada = ada_clf.predict(test_x)  

## Cross table for test data
pd.crosstab(test_y, test_pred_ada)

## Test Accuracy
test_acc_ada = np.mean(test_y == test_pred_ada)
test_acc_ada

## Predicting on train data
train_pred_ada = ada_clf.predict(train_x)

## Predicting on train data
pd.crosstab(train_y, train_pred_ada)

## Train Accuracy
train_acc_ada = np.mean(train_y == train_pred_ada)
train_acc_ada

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\

## eXtreme Gradient Boosting
## Model buielding
## pip install xgboost
import xgboost as xgb
model_xg = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)


## FItting the model to train data
model_xg.fit(train_x, train_y)

## Model evoluting on test data
pred_xg = model_xg.predict(test_x)

## Confusion matrics for test data
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(test_y, pred_xg)

## Accuracy for test data
test_Acc_XG = accuracy_score(test_y, pred_xg)
test_Acc_XG

## Model avolutiong on train data
pred_trn_xg = model_xg.predict(train_x)

## Confusion matrics
confusion_matrix(pred_trn_xg, train_y)

## Accuracy for train data
train_Acc_XG = accuracy_score(train_y, pred_trn_xg)
train_Acc_XG

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\
## SVM
## Buielding the model
from sklearn.svm import SVC
svm = SVC(kernel = "linear")
## Fitting on the train data
svm.fit(train_x, train_y)

## Predict on test data
pred_svm = svm.predict(test_x)

## confussion matrics
con_svm = pd.crosstab(pred_svm, test_y)
con_svm

## Accuracy for test data
acc_test_svm = np.mean(pred_svm == test_y)
acc_test_svm

## predict on train
pred_svm_train = svm.predict(train_x)

## confusion matrics for train data
comn_test = pd.crosstab(pred_svm_train, train_y)
comn_test

## Accuracy for train data
acc_train_svm = np. mean(pred_svm_train == train_y)
acc_train_svm



# Which is the best Model ?
results = pd.DataFrame({
    'Model': ['Decision Tree', 'prouning_DT', 'grid search alg', 'Random forest', 'Ada boosting', 'EXG boosting', 'SVM'],
    'train accuracy': [acc_train_D, acc_train_pr, acc_train_gr , accuracy_train_RF, train_acc_ada, train_Acc_XG, acc_train_svm]})

result_df = results.sort_values(by='train accuracy', ascending=False)
result_df_train = result_df.set_index('train accuracy')
result_df_train

results = pd.DataFrame({
    'Model': ['Decision Tree', 'prouning_DT', 'grid search alg', 'Random forest', 'Ada boosting', "EXG boosting", 'SVM'],
    'test accuracy': [acc_test_D, accuracy_test_pr, accuracy_test_gr, accuracy_test_RF, test_acc_ada, test_Acc_XG, acc_test_svm]})

result_df = results.sort_values(by='test accuracy', ascending=False)
result_df_test = result_df.set_index('test accuracy')
result_df_test

