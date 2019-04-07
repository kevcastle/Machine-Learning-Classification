# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:21:12 2019

@author: Kevin
"""

#Loading Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as smf # logistic regression
from sklearn.model_selection import cross_val_score


#Loading the file

got = pd.read_excel('GOT_character_predictions.xlsx')

##############################################################################
#EDA
##############################################################################

got.info()
"have a total of 1946 observations and 26 variables"

pd.set_option('display.max_columns', None)

got.describe()

"""
more characters are male than female , the book where more characters appear 
is book 4, most mothers are alive while most fathers are dead, an interesting 
variable is age because the mean is negative

"""

got.corr().round(3)

#how many charcaters died?
got['isAlive'].value_counts()
495/1946
"495 characters of the 1946 total characters have died, 25%"

#how many characters do not appear in any books

got['sum_books'] = (got['book1_A_Game_Of_Thrones'] + 
                   got['book2_A_Clash_Of_Kings'] +
                   got['book3_A_Storm_Of_Swords'] +
                   got['book4_A_Feast_For_Crows']+
                   got['book5_A_Dance_with_Dragons'])

got['sum_books'].value_counts()
"272 characters do not appear in any of the books"

#how many of the characters did not appear in any of the books and died?

got['no_book_and_dead']=got['sum_books']+got['isAlive']

got['no_book_and_dead'].value_counts()

"121 of the 495 charcaters who died did not appear in any of the books" 

#changing all the values greater than 0 in no_book_and_dead column to 0s and 1s
got['no_book_and_dead'].values[got['no_book_and_dead'] > 0] = 1

#exploring age variable
got['age'].value_counts()
"Two ages are negative so this variable is confusing right now"

#looking more into detail and explore DOB which is related to age
got['dateOfBirth'].value_counts()

""" 
there are two people whose ages are too high, Doreah and Rhaego, after 
reserach, seems like their DOB 298299 & 278279 seems to be a manual error where
should be 298 and 278
"""

#changing the two date of births manually to the first three numbers
got['dateOfBirth'].values[got['dateOfBirth'] == 298299] = 298
got['dateOfBirth'].values[got['dateOfBirth'] == 278279] = 278

#changing the ages for the same two observations based on research 
got['age'].values[got['age'] == -298001] = 0
got['age'].values[got['age'] == -277980] = 20

#histogram to see popularity ranges where most people die
got_dead = got[got.isAlive == 0]
plt.hist(x='popularity',data=got_dead)
plt.title("Popularity Distribution of Dead Characters")
plt.xlabel('Popularity')
plt.show()

#histogram to compare with Popularity for Alive Characters
got_alive = got[got.isAlive == 1]
plt.hist(x='popularity',data=got_alive)
plt.title("Popularity Distribution of Alive Characters")
plt.xlabel('Popularity')
plt.show()

""" 
More popular characters were classified as dead when comparing the two 
histograms. 
"""

#histogram to see male vs. female for dead
plt.hist(x='male',data=got_dead)
plt.title("Gender Distribution of Dead Characters")
plt.xlabel('Gender')
plt.show()
got_dead['male'].value_counts()
"almost 75% of the people who died were male"

#histogram to see Number of Dead Relations for Dead Characters
plt.hist(x='numDeadRelations',data=got_dead)
plt.title("RelationsDistribution of Dead Characters")
plt.xlabel('NumDeadRelations')
plt.show()

#histogram to compare with Num of Dead Relations for Alive Characters
got_alive = got[got.isAlive == 1]
plt.hist(x='numDeadRelations',data=got_alive)
plt.title("RelationsDistribution of Alive Characters")
plt.xlabel('NumDeadRelations')
plt.show()

#checking to see if there are missing values in NumDeadRelations
got['numDeadRelations'].isnull().sum()

""" 
Seems like most characters dead or alive have 0 Num Dead Relations but 
since there are no missing values maybe 0s can mean actually 0 or unknown
"""

#creating a new variable to show NumDeadRelations greater than 0 to 1
got['anyDeadRelations']= got['numDeadRelations']
got['anyDeadRelations'].values[got['anyDeadRelations'] > 0] = 1

#histogram to see the popularity of the characters who didn't appear in a book
got_no_book = got[got['sum_books'] == 0]
plt.hist(x='popularity',data= got_no_book)
plt.title("Popularity Distribution of Characters in No Books")
plt.xlabel('Popularity')
plt.show()

#histogram to see the distribution of characters who died and if were noble
plt.hist(x='isNoble',data=got_dead)
plt.title("Nobility Distribution of Dead Characters")
plt.xlabel('isNoble')
plt.show()

#histogram to compare with characters who are alive and if were noble
plt.hist(x='isNoble',data=got_alive)
plt.title("Nobility Distribution of Alive Characters")
plt.xlabel('isNoble')
plt.show()

"""
about 50/50 when they died and for the alive characters, less characters are
noble which makes sense so maybe this variable will not help
"""

#missing values
got.isnull().sum()
""" 
Variables such as father, heir,isAliveMother,isAliveFather,isAliveHeir,
isAliveSpouse, age, Date of Birth, spouse and others have a lot of missing 
values, some are more than 90% missing so maybe shouldn't be included in the 
model.
"""

#imputing median for Date of Birth and Age
dob_median = got['dateOfBirth'].median()
got['dateOfBirth'] = got['dateOfBirth'].fillna(dob_median).round(3)

age_median = got['age'].median()
got['age'] = got['age'].fillna(age_median).round(3)

#imputing other numerical variables with -1 to flag as missing 
num_fill = ['isAliveMother', 'isAliveFather', 'isAliveHeir', 'isAliveSpouse']

for col in got[num_fill]:
    if got[col].isnull().astype(int).sum() > 0:
        got[col] = got[col].fillna(-1)

# imputing categorical missing values with unknown
cat_fill =['title', 'culture','mother','father', 'heir', 'house', 'spouse']

for col in got[cat_fill]:
    if got[col].isnull().astype(int).sum() > 0:
        got[col] = got[col].fillna('unknown')

#checking for missing values
got.isnull().sum()

"All missing values were imputed!"

#exploring the house
got['house'].value_counts()

#checking to see which houses have the most dead people

perc = got_dead['house'].value_counts()/got['house'].value_counts()
perc.rename_axis(columns={"house": "perc"},axis=1)
house_list = got['house'].value_counts().sort_index(axis=0)


house = pd.concat([perc,house_list],axis=1)
house.columns = ['perc_dead','house_size']

#creating a new vairable of dangerous houses(house_size >=10 & perc > 0.3)
got['house_danger'] = got['house'].replace({"Night's Watch":"dangerous",                                              
                                             "House Stark":"dangerous",
                                             "House Targaryen":"dangerous",
                                             "House Lannister":"dangerous",
                                             "House Bolton":"dangerous",
                                             "House Greyjoy":"dangerous",
                                             "House Arryn":"dangerous",
                                             "House Whent":"dangerous",
                                             "House Baratheon":"dangerous",
                                             "House Bracken":"dangerous",
                                             "House Tully":"dangerous",
                                             "House Velaryon":"dangerous",
                                             "Brave Companions":"dangerous"})

for row in enumerate(got.loc[: ,'house_danger']):
    
    if row[1] == 'dangerous'  :
        got.loc[row[0], 'house_danger'] = 1
        
    else:
        got.loc[row[0], 'house_danger'] = 0
                
#checking new correlations
got.corr().round(3)

# Export file to excel
got.to_excel('got_plus.xlsx')


###############################################################################
#Modeling 
###############################################################################

###############################################################################
#Logistic Regression with StatModel
###############################################################################

got_data = got.loc[: , ['male', 
                        'popularity',
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'dateOfBirth',
                        'house_danger']]
got_target =  got.loc[: , 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508,
            stratify = got_target)

got_train = pd.concat([X_train, y_train], axis = 1)


model = smf.logit(formula = """isAlive ~ male +
                                       anyDeadRelations +
                                       dateOfBirth +
                                       house_danger +
                                       popularity
                                       """,
                  data = got_train)
 

result = model.fit()

result.summary()

###############################################################################
#Logistic Regression with Sci-Kit Learn
###############################################################################
got_data = got.loc[:,['male',
                      'anyDeadRelations',
                      'house_danger',
                      'dateOfBirth',
                      'popularity']]

got_target = got.loc[: ,'isAlive']

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(got_data,
                                                    got_target,
                                                    test_size = 0.1,
                                                    random_state = 508)

from sklearn.linear_model import LogisticRegression

# Instantiate
lr = LogisticRegression()

# Fit
lr_fit = lr.fit(X_train, y_train)

# Predict[]
lr_pred = lr_fit.predict(X_test)

# Score
y_score_ols = lr_fit.score(X_test, y_test)

print(y_score_ols) 
print('Training Score', lr_fit.score(X_train, y_train).round(4))
print('Testing Score:', lr_fit.score(X_test, y_test).round(4))

###############################################################################
#Decision Tree
###############################################################################


got_data = got.loc[: , ['male', 
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'house_danger',
                        'dateOfBirth',
                        'popularity']]
got_target =  got.loc[: , 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508,
            stratify = got_target)

from sklearn.tree import DecisionTreeClassifier # Classification trees

c_tree = DecisionTreeClassifier(random_state = 508)

c_tree_fit = c_tree.fit(X_train, y_train)


print('Training Score', c_tree_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree_fit.score(X_test, y_test).round(4))

###############################################################################
#KNN Unscaled
###############################################################################

from sklearn.neighbors import KNeighborsClassifier 

got_data = got.loc[: , ['male', 
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'house_danger',
                        'dateOfBirth',
                        'popularity']]
got_target = got.loc[:, 'isAlive']

#with unscaled data
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)

training_accuracy = []
test_accuracy = []



neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)


knn_reg = KNeighborsClassifier(algorithm = 'auto',
                              n_neighbors = 3)



# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_train = knn_reg.score(X_train, y_train)
y_score_knn_optimal = knn_reg.score(X_test, y_test)

print(y_score_knn_optimal)
print('Training Score', y_score_knn_train.round(4))
print('Testing Score:', y_score_knn_optimal.round(4))

cv_lr_3 = cross_val_score(knn_reg,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc')

print(pd.np.mean(cv_lr_3))



###############################################################################
#KNN Scaled
###############################################################################

#importing library to scale data
from sklearn.preprocessing import StandardScaler

got_data = got.loc[: , ['male', 
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'house_danger',
                        'dateOfBirth',
                        'popularity']]
got_target = got.loc[:, 'isAlive']

# Instantiating a StandardScaler() object
scaler = StandardScaler()


# Fitting the scaler with our data
scaler.fit(got_data)


# Transforming our data after fit
got_scaled = scaler.transform(got_data)


# Putting our scaled data into a DataFrame
got_scaled_df = pd.DataFrame(got_scaled)

# Adding labels to our scaled DataFrame
got_scaled_df.columns = got_data.columns

#with scaled data
X_train, X_test, y_train, y_test = train_test_split(
            got_scaled_df,
            got_target,
            test_size = 0.1,
            random_state = 508)

training_accuracy = []
test_accuracy = []



neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)


knn_reg = KNeighborsClassifier(algorithm = 'auto',
                              n_neighbors = 27)



# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_train = knn_reg.score(X_train, y_train)
y_score_knn_optimal = knn_reg.score(X_test, y_test)

print(y_score_knn_optimal)
print('Training Score', y_score_knn_train.round(4))
print('Testing Score:', y_score_knn_optimal.round(4))

cv_lr_3 = cross_val_score(knn_reg,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc')

print(pd.np.mean(cv_lr_3))

###############################################################################
#Random Forest
###############################################################################

#Loading Random Forest Library
from sklearn.ensemble import RandomForestClassifier

got_data = got.loc[: , ['male', 
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'house_danger',
                        'dateOfBirth',
                        'popularity']]
got_target = got.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508,
            stratify = got_target)

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)



# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))

cv_lr_3 = cross_val_score(full_forest_gini,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc')

print(pd.np.mean(cv_lr_3))

cv_lr_3 = cross_val_score(full_forest_entropy,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc')

print(pd.np.mean(cv_lr_3))

###############################################################################
#Gradient Boosted Machines
###############################################################################

from sklearn.ensemble import GradientBoostingClassifier

got_data = got.loc[: , ['male', 
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'house_danger',
                        'dateOfBirth',
                        'popularity']]
got_target = got.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508,
            stratify = got_target)

# Building a gbm
gbm = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
                                  max_depth = 3,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )


gbm_basic_fit = gbm.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))

cv_lr_3 = cross_val_score(gbm,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc')

print(pd.np.mean(cv_lr_3))


#########################
# Hyper Parameter Tuning
#########################

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]


param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

random_grid = RandomForestClassifier(random_state = 508)



# Creating a GridSearchCV object
random_grid_cv = RandomizedSearchCV(random_grid, param_grid, cv = 3)



# Fit it to the training data
random_grid_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned RF Parameter:", random_grid_cv.best_params_)
print("Tuned RF Accuracy:", random_grid_cv.best_score_.round(4))

###############################################################################
#Optimal Random Forest
###############################################################################

got_data = got.loc[: , ['male', 
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'house_danger',
                        'dateOfBirth',
                        'popularity']]
got_target = got.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508,
            stratify = got_target)

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 1400,
                                     criterion = 'gini',
                                     max_depth = 60,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 1400,
                                     criterion = 'entropy',
                                     max_depth = 60,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)



# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()

#Predictions for the model
rf_gini_pred = full_forest_entropy.predict(X_test)
rf_entropy_pred = full_forest_entropy.predict(X_test)


# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))

cv_lr_3 = cross_val_score(full_forest_gini,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc')

print(pd.np.mean(cv_lr_3))

cv_lr_3 = cross_val_score(full_forest_entropy,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc').round(3)

print(pd.np.mean(cv_lr_3))

""" 
After experimenting with different models, entropy random forest with the 
optimal parameters is the best model considering both AUC and training score. 
"""


########################
# Feature importance function
########################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

########################
        
plot_feature_importances(full_entropy_fit,
                         train = X_train,
                         export = False)

###############################################################################
#FINAL MODEL - BEST OVERALL
###############################################################################

"In the end went with entropy random forest due to highest AUC score after CV"

got = pd.read_excel('got_plus.xlsx')

got_data = got.loc[: , ['male', 
                        'anyDeadRelations',
                        'no_book_and_dead',
                        'house_danger',
                        'dateOfBirth',
                        'popularity']]
got_target = got.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508,
            stratify = got_target)

# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 1400,
                                     criterion = 'entropy',
                                     max_depth = 60,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the model
full_entropy_fit = full_forest_entropy.fit(X_train, y_train)

#Predictions for the model
rf_entropy_pred = full_forest_entropy.predict(X_test)

# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))

cv_lr_3 = cross_val_score(full_forest_entropy,got_data, got_target, cv = 3, 
                          scoring =  'roc_auc').round(3)

print(pd.np.mean(cv_lr_3))

# Saving model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'RF_Predicted': rf_entropy_pred})


model_predictions_df.to_excel("Game_of_Thrones.xlsx")
