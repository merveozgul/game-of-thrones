#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:49:36 2019

@author: merveozgul

Purpose of this script is to provide analysis and prediction models to find out
which characters in the epic fantasy novel series Game of Thrones will survive.
 
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
from sklearn.model_selection import GridSearchCV    # hyperparameter optimization
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # Classification trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score # AUC value
from sklearn.metrics import confusion_matrix  

file = 'GOT_character_predictions.xlsx'

got  = pd.read_excel(file)


######################################################################
# Part 1) Initial Data Exploration
######################################################################
got.head(5)

got.tail(5)

with pd.option_context('display.max_rows', 50, 'display.max_columns', 50):
    print(got.describe())
    print(got.info())
    print(got.isnull().sum().sum())

#-----------------------------------------------
#  1. Data Types in the Dataset     
#-----------------------------------------------

list(set(got.dtypes))

"""Following part presents the distribution of two data types, numerical and
non-numerical. """

#-----------------------------------------------
# 1.a)Distribution of Numerical Data         
#-----------------------------------------------

got_num = got.select_dtypes(include = ['float64', 'int64'])
print('There is {} numerical features including:\n{}'.format(len(got_num.columns), got_num.columns.tolist()))

got_num.head()

got_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

#count of missing values
print(got_num.isnull().sum())

#percentage of missing values
print((got_num.isnull().sum()/len(got_num)).round(2))

#-----------------------------------------------
# 1.b) Distribution of Non-Numerical Data         
#-----------------------------------------------

#selecting only categorical variables
got_not_num = got.select_dtypes(include = ['O'])
print('There is {} non numerical features including:\n{}'.format(len(got_not_num.columns), got_not_num.columns.tolist()))

#Unique value counts for categorical variables
for col in got_not_num: 
    print(got_not_num[col].value_counts())
    print(got_not_num[col].describe())

#count of missing values
print(got_not_num.isnull().sum())

#percentage of missing values
print((got_not_num.isnull().sum()/len(got_not_num)).round(2))

#-----------------------------------------------
# 2) Missing Values      
#-----------------------------------------------
"""
Missing value columns for numeric data (%):
    isAliveMother                 0.99
    isAliveFather                 0.99
    isAliveHeir                   0.99    
    isAliveSpouse                 0.86
    age                           0.78 
    dateOfBirth                   0.78
    
Missing value columns for non-numeric data(%):   
    mother     0.99
    father     0.99
    heir       0.99
    spouse     0.86
    culture    0.65
    title      0.52
    house      0.22
"""
#-----------------------------------------------
# 2.a) Flagging Missing Values      
#-----------------------------------------------
for col in got:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)


#-----------------------------------------------
# 2.b) Correcting Some of the information     
#-----------------------------------------------
got.columns

"""After initial exploration, there are some columns that has to be corrected:
    2.b.1) Title & Gender (female titles gender equal to male incorrectly)
    2.b.2) Culture (repetitive, they can be corrected)
    2.b.3) Age & Date Birth (negative values)
"""
#--------------------------
# 2.b.1)Correcting gender
#--------------------------
""" Some male characters were marked as female. Some female characters were marked
as male as well.

*Most of the titles include words "Lord", "King", "Knight" so we can easily correct
gender for those values with "str.contains()".

*Likewise, we can do the same thing for female titles.

"""
#Checking for Males (male = 1)
with pd.option_context('display.max_rows', 800, 'display.max_columns', 50):
    print(got[got['male']==1]['title'].value_counts())
    
#Checking for Females  (male = 0)  
with pd.option_context('display.max_rows', 800, 'display.max_columns', 50):
    print(got[got['male']==0]['title'].value_counts())
    
#checking the unique title names 
print(got['title'].unique())

#------------------------------------------------------------------------------
#Gender & Title (Male)
mtitle_list = ["Ser", 'Septon',"Prince", "Archmaester", 'Lord', 'Maester',  "Bloodrider",
               "Khal", 'King', 'Knight', 'BrotherProctor', 'Cupbearer', 'Steward'] 

#correcting gender for female titles
for val in enumerate(got.loc[ : , 'title']):
 for mtitle in mtitle_list:   
    if val[1] == mtitle:
        got.loc[got['title'].str.contains(mtitle, na=False), 'male'] = 1  
 
#Gender & Title (Female)
ftitle_list = ["Queen", "Princess", 'Lady', 'Sister', 'Wind Witch', 'Wife', 'Septa', 'Sweetsister'] 

#correcting gender for female titles
for val in enumerate(got.loc[ : , 'title']):
 for ftitle in ftitle_list:   
    if val[1] == ftitle:
        got.loc[got['title'].str.contains(ftitle, na=False), 'male'] = 0  

got.loc[got['title'].str.contains('Princess', na=False), 'male'] = 0

#Titles that are noble
got[got['isNoble']==1]['title'].value_counts()  

got[got['isNoble']==0]['isAlive'].sum() / len(got['isNoble'])

#Checking
with pd.option_context('display.max_rows', 800, 'display.max_columns', 50):
    print(got[got['male']==1]['title'].value_counts())
    print(got[got['male']==0]['title'].value_counts())   

#--------------------------
# 2.b.2)Culture 
#--------------------------
#Andals
got.loc[got["culture"] == "Andal","culture"] = "Andals"

#Asshai
got.loc[got["culture"] == "Asshai'i","culture"] = "Asshai"

#Astapori
got.loc[got["culture"] == "Astapori","culture"] = "Astapori"

#Braavosi
got.loc[got["culture"] == "Braavosi","culture"] = "Braavosi"

#Dornishmen
got.loc[got["culture"].isin(["Dorne","Dornish"]),"culture"] = "Dornishmen"

#Ghiscari
got.loc[got["culture"] == "Ghiscaricari","culture"] = "Ghiscari"

#Ironborn
got.loc[got["culture"].isin(['Ironmen', 'ironborn']),"culture"] = "Ironborn"

#Lhazareene
got.loc[got["culture"] == "Lhazareene","culture"] = "Lhazareen"

#Lysene
got.loc[got["culture"].isin(['Lysene', 'Lyseni']),"culture"]  = "Lyseni"

#Meerenese
got.loc[got["culture"] == "Meereen","culture"] = "Meereenese"

#Northmen
got.loc[got["culture"].isin(["Northern mountain clans", "northmen"]),"culture"] = "Northmen"

#Norvoshi
got.loc[got["culture"] == "Norvos","culture"] = "Norvoshi"

#Qarth
got.loc[got["culture"] == "Qarth","culture"] = "Qartheen"

#Reachmen
got.loc[got["culture"].isin(["Reachmen","The Reach"]),"culture"] = "Reach" 
 
#Rivermen
got.loc[got["culture"].isin(['Riverlands', 'Rivermen']),"culture"]  = "Rivermen"

#Stormlanders
got.loc[got["culture"] == "Stormlander","culture"] = "Stormlanders"

#Summer Islander 
got.loc[got["culture"].isin(["Summer Islands","Summer Isles"]),"culture"] = "Summer Islander"

#Valemen
got.loc[got["culture"].isin(["Vale","Vale mountain clans"]),"culture"] = "Valemen"

#Westermen
got.loc[got["culture"].isin(['westermen', 'Westerlands', 'Westerman']),"culture"] = "Westermen"

#Wildlings
got.loc[got["culture"].isin(['Free Folk', 'free folk', 'Wildling']),"culture"] = "Wildlings"

####Counting unique values 
got['culture'].value_counts()


#----------Exploring with plots
culture_plot = sns.countplot(x="culture", hue= "isAlive", data=got)
sns.set_style("ticks", {"xtick.major.size":5, "ytick.major.size":7})
#sns.set_context("notebook", font_scale=0.5, rc={"lines.linewidth":0.3})
#improving the figure size
sns.set(rc={'figure.figsize':(11.7,8.27)})
#rotating the xtick labels
for item in culture_plot.get_xticklabels():
    item.set_rotation(90)
plt.show()
"""
Note: Northmen, Ironborn and Wildlings are the most crowded cultures. 
A lot of people from Ironborn, Northmen, Braavosi, Dornishmen cultures
seem to survive. 
On the otherside, people from Valyrian culture seem to die. That seems like on of the
culture that dead characters are more than alive people.

"""

#--------------------------
# 2.b.3)  Age & Date Birth 
#--------------------------
#---------Correcting date births & negative ages based on the research
#finding negative ages
got[got['age']<0]['age']

#date of births
#Rhaego (died in same year)
got['dateOfBirth']= got['dateOfBirth'].replace(298299, 298) 

#Doreah, died in 299
got['dateOfBirth']= got['dateOfBirth'].replace(278279, 278) 

#age 1: Rhaego (since died in same year, will replace with 0)
got['age']= got['age'].replace(-298001.0, 0)

#age 2: Doreah (299-278=20)
got['age']= got['age'].replace(-277980.0, 20)
 
########Imputing with median for dateofbirth
got['dateOfBirth'].describe()

fill= got['dateOfBirth'].median()

got['dateOfBirth'] = got['dateOfBirth'].fillna(fill)


########Imputing with median for age
got['age'].describe()

fill= got['age'].median()

got['age'] = got['age'].fillna(fill)

#------------------------------------------------------------------------------
#     Replacing Missing Values with 'Unknown' for categorical columns         
#------------------------------------------------------------------------------
#Function to imputing the missing values 
def cat_imputation(df, column, value):
    df.loc[df[column].isnull(),column] = value

"""Ref: https://www.kaggle.com/meikegw/filling-up-missing-values """


#Replace missing values with Unknown for non-numerical data type
for col in got.columns:
    if got[col].dtype == 'O':
        cat_imputation(got, col, 'Unknown')


###############################################################################
# Part 2)Data Preparation: More EDA, Exploratory Insights & Feature Engineering
###############################################################################

#------------------------------------------------------------------------------
#    Gender, Nobility & Survival (EDA Only)  
#------------------------------------------------------------------------------
#----------Exploring with plots
gender_plot = sns.countplot(x="male", hue= "isAlive", data=got)
sns.set_style("ticks", {"xtick.major.size":5, "ytick.major.size":7})
#sns.set_context("notebook", font_scale=0.5, rc={"lines.linewidth":0.3})
#improving the figure size
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.show()

#noble vs. survival
noble_plot = sns.countplot(x="isNoble", hue= "isAlive", data=got)
plt.show()
"""Female characters are more likely to survive."""        

#Gender, Nobility & Survival       
#----------Exploring with plots

female = got[got['male'] == 0]

#Number of Noble females
female[female['isNoble'] == 1]['name'].count() #268

#Number of NOT Noble Females
female[female['isNoble'] == 0]['name'].count() #473

#Plotting the survival rate
noblef_plot = sns.countplot(x="isNoble", hue= "isAlive", data=female)
noblef_plot.set_title("Being Noble & Alive/Dead Ratio (Female)")
plt.show()

#Alive female characters houses
female[female['isAlive'] == 0]['house'].value_counts()

""" There are more not noble female characters, number of alive people
outnumbers the dead characters. 
When we look at the houses and dead female characters, House Targaryen has
the highest number of 20 dead people, and it's followed by Night's Watch
with 11. 
 """
 
#male 
male = got[got['male'] == 1]

#Number of Noble females
male[male['isNoble'] == 1]['name'].count() #268

#Number of NOT Noble Females
male[male['isNoble'] == 0]['name'].count() #473

#Plotting the survival rate
noblem_plot = sns.countplot(x="isNoble", hue= "isAlive", data=male)
noblem_plot.set_title('Being Noble & Alive/Dead Ratio (Male)')
plt.show()

#------------------------------------------------------------------------------
#                 House    (EDA & Feature Engineering)    
#------------------------------------------------------------------------------
#grouping the large-in-size & important houses together
""" Ref: https://gameofthrones.fandom.com/wiki/Great_House """   
#list of important houses 
lst_house_big = [" Night's Watch",
                 "House Targaryen",
                 "House Stark",
                 "House Lannister",
                 "House Frey",
                 "House Arryn",
                 "House Baratheon",
                 "House Baratheon of Dragonstone",
                 "House Baratheon of King's Landi",
                 "House Tyrell",
                 "House Martell",
                 "House Greyjoy"]

#creating a new column 
got['isHousebig'] = 0 

for val in enumerate(got.loc[ : , 'house']):
    for item in lst_house_big: 
        if val[1] == item:
            got.loc[val[0], 'isHousebig'] = 1
            
#Alive & Dead ratio
got[got['isHousebig'] == 1]['isAlive'].sum() / len(got['isHousebig'])

#filtering big houses
bighouses_subset = got[got.house.isin(lst_house_big)].append(got[got['house']=="Night's Watch"])

#Big Houses Dead & Alive Characters 
d1 = bighouses_subset.groupby(["house", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
d1.loc[:, "total"]= d1.sum(axis = 1)
p1 = d1[d1.index != ""].sort_values("total")[[0, 1]].plot.barh(stacked = True, rot = 0, figsize = (14, 12),)
t1 = p1.set(xlabel = "No. of Characters", ylabel = "House"), p1.legend(["Dead", "Alive"], loc = "lower right")
plt.show()
#------------------------------------------------------------------------------
#            Books (EDA & Feature Engineering)
#------------------------------------------------------------------------------

"""Main question I will investigate in this subset: 
    1) How many characters in each book? 
    2) Which characters appear in all books?
    3) Which characters appear in more than 1 book?
    
    Note: A person can be mentioned in a book regardless of being alive or dead."""
for col in got.columns:
    if 'book' in col:
        print(got[col].describe())

#Percentage of Alive members per book
for col in got.columns:
    if 'book' in col:
        print(got[col].name, " ", (got[got[col]==1]['isAlive'].sum()/len(got[col])).round(2))

"""book1_A_Game_Of_Thrones   0.12
   book2_A_Clash_Of_Kings   0.27
   book3_A_Storm_Of_Swords   0.36
   book4_A_Feast_For_Crows   0.5
   book5_A_Dance_with_Dragons   0.3 """


#number of characters in book 1
book_1 = got['book1_A_Game_Of_Thrones'].sum()
#percentage of characters survived in the first book
(got[got['book1_A_Game_Of_Thrones']==1]['isAlive'].sum() / book_1).round(2) #0.62


#number of characters in book 2
book_2 = got['book2_A_Clash_Of_Kings'].sum()
#percentage of characters survived in the second book
(got[got['book2_A_Clash_Of_Kings']==1]['isAlive'].sum() / book_2).round(2) #0.71

#number of characters in book 3
book_3 = got['book3_A_Storm_Of_Swords'].sum()
#percentage of characters survived in the third book
(got[got['book3_A_Storm_Of_Swords']==1]['isAlive'].sum() / book_3).round(2) #0.75

#number of characters in book 4
book_4 = got['book4_A_Feast_For_Crows'].sum()
#percentage of characters survived in the fourth book
(got[got['book4_A_Feast_For_Crows']==1]['isAlive'].sum() / book_4).round(2) #0.84

#number of characters present in the 5th book 
book_5 = got['book5_A_Dance_with_Dragons'].sum()
#percentage of characters survived in the fifth book
(got[got['book5_A_Dance_with_Dragons']==1]['isAlive'].sum() / book_5).round(2) #0.76


#------------------------------------------------------------


#Creating a column for number of books that character has appeared
got["sumbooks"] = got['book1_A_Game_Of_Thrones']+ got['book2_A_Clash_Of_Kings'] 

got["sumbooks"] = got["sumbooks"] + got['book3_A_Storm_Of_Swords'] 

got["sumbooks"] = got['sumbooks'] + got['book4_A_Feast_For_Crows'] 

got["sumbooks"] = got["sumbooks"] + got['book5_A_Dance_with_Dragons']

got["sumbooks"].value_counts()


"""Note: It is possible that a character may not have appeared in 5 books in this
dataset. There are other books that includes further stories from seven kingdoms.
Examples:
    * The Hedge Knight
    * The Sworn Sword (i.e. the character Sam Troops was mentioned in this book
                       but was not mentioned in any of the books in our dataset)
    *The Sons of the Dragon
    * Fire and Blood
    
    Ref: http://www.georgerrmartin.com/book-category/?cat=song-of-ice-and-fire
        https://awoiaf.westeros.org/index.php/The_Sworn_Sword
    For now, I will assume that character appeared in one of the other books, if
    the total appearance in the books is zero. 
    """

#Distribution
got.sumbooks.hist(bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

#Relationship with appearance in total number of books & being alive
sumbookalive = sns.countplot(x="sumbooks", hue='isAlive', data=got)
sumbookalive.set_title("Total No. of Books vs Alive/Dead Ratio")
plt.show()

"""Smaller the difference between alive and death characters for the total books
appeared, it gets harder for us to classify them. """

#sumbook outlier column
sumbooks_low = 1
sumbooks_hi = 4
got['out_sumbooks'] = 0

for val in enumerate(got.loc[ : , 'sumbooks']):
    
    if val[1] < sumbooks_low:
        got.loc[val[0], 'out_sumbooks'] = -1

    if val[1] >= sumbooks_hi:
        got.loc[val[0], 'out_sumbooks'] = 1
           

###########################################################
#  Grouping Cultures based on the continent  & region
###########################################################
got['culture'].value_counts()

#cultures of North Westeros
got['isNWesteros'] = 0
isNWesteros_list = ['Northmen', 'Andal', 'Crannogmen', 'First Men']

for val in enumerate(got.loc[ : , 'culture']):
    for element in isNWesteros_list: 
        if val[1] == element:
            got.loc[val[0], 'isNWesteros'] = 1

"""https://gameofthrones.fandom.com/wiki/The_North
Note: Creating a isWesteros column didn't add extra value for the model as it
was correlated with other variables.So I wanted to group the houses in the North
Westeros. 
"""


#creating a column for people from westeros 
got['isWesteros'] = 0

westerolist = ['Northmen', 'northmen', 'Andal', 'Andals', 'Free folk', 'free folk', 
               'Wildling', 'Wildlings', 'westermen', 'Westerlands', 'Westerman',
               'Crannogmen', 'Valemen', 'Vale', 'Vale mountain clans', 'Stormlands', 
               'Stormlander', 'Ironborn', 'Ironmen', 'ironborn', 'Riverlands', 'Rivermen',
               'Dornishmen', 'Dornish', 'Dorne','Reachmen', 'Reach', 'The Reach'] 


for val in enumerate(got.loc[ : , 'culture']):
    for element in westerolist: 
        if val[1] == element:
            got.loc[val[0], 'isWesteros'] = 1

#Westeros survival rate
(got[got['isWesteros']==1]['isAlive'].sum()/len(got[got['isWesteros']==1]['isAlive'])).round(2)


#-------Ironborn
#Creating a binary col for Ironborn
got['isIronborn'] = 0

for val in enumerate(got.loc[ : , 'culture']):
    
    if val[1] == 'Ironborn' or val[1] == 'Ironmen' or val[1] == 'ironborn':
        got.loc[val[0], 'isIronborn'] = 1


#-----Essosi
#list of cultures in Essos
essos_list = ['Braavosi', 'Braavos', 'Lysene', 'Lyseni', 'Tyroshi', 
              'Qohor', 'Norvoshi', 'Norvos', 'Myrish','Pentoshi']
        
        
#Creating a binary col for Essosis
got['isEssosi'] = 0

for val in enumerate(got.loc[ : , 'culture']):
    for element in essos_list: 
        if val[1] == element:
            got.loc[val[0], 'isEssosi'] = 1

#percentage of alive in Essos
(got[got['isEssosi']==1]['isAlive'].sum()/len(got[got['isEssosi']==1]['isAlive'])).round(2)

####################################################            
# Deciding on the Features I will use
###################################################

"""Decision criteria to eliminate:
    1) Columns that are non-numerical and column 'S.No' 
    2) Columns that have missing value more than 80%
    3) Columns that are highly correlated with each other, avoiding collinearity """

#1) dropping non-numerical columns
got2 = got.select_dtypes(include = ['float64', 'int64'])
got2 = got2.drop(['S.No'], axis=1)

# 2)Columns that have missing value more than 80% 
got2 = got2.dropna(thresh=0.8*len(got), axis=1)


# 3) Correlation Matrix
####################################################
got_corr = got2.corr().round(2)
print(got_corr)

corr2 = got_corr.iloc[:, :]
pd.options.display.float_format = '{:,.4f}'.format

#plotting correlation that is above 0.6
corr2[np.abs(corr2) < 0.5] = 0
plt.figure(figsize=(16,10))
sns.heatmap(corr2, annot=True, cmap='YlGnBu')
plt.show()

"""Note: Although correlation matrix will not capture the full relationship with
variables, we can still use it as a reference point. """

got['popularity'].corr(got['isAlive'])

###############################################################################
# Part 3 & 4) Model Building & Evaluation 
###############################################################################

"""This part consists of 2 parts:
    A) Single Models
        -Logistic Regression
        -KNN Classifier
        -DecisionClassifier Tree
    B)Ensemble Model
        -Random Forest

Evaluation metrics for the models:
    -AUC 
    -Confusion Matrix 
    -Precision & Recall metrics
    -Accuracy
    
Ref:https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
        
Note: It is important to check for class imbalance for the response variable."""


#---Dropping more features
got_train = got2.drop(['isAlive'], axis=1)


#creating the target variable
got_target = got2['isAlive']

#checking for class imbalance
sns.countplot(x="isAlive", data=got2)
plt.show()
"""There is a class imbalance for our response variable. We should take care of
the class imbalance in our train test split by stratified sampling. """


# train_test_split (stratified response variable)
X_train, X_test, y_train, y_test = train_test_split(got_train,
                                                    got_target,
                                                    test_size = 0.1,
                                                    random_state = 508,
                                                    stratify = got_target)

#-----------------------------------------------------------------------------
#      1. Logistic Regression
#-----------------------------------------------------------------------------
#Checking for class imbalance for train & test sets
# Original Value Counts
got_target.value_counts()

got_target.sum() / got_target.count()


# Training set value counts
y_train.value_counts()

y_train.sum() / y_train.count()


# Testing set value counts
y_test.value_counts()

y_test.sum() / y_test.count()

"""Ratios of alive vs. dead characters between train and test sets match. So 
we don't have to control for class imbalance. """


log_reg = LogisticRegression(random_state = 508)

#fitting the model
logreg_fit = log_reg.fit(X_train, y_train)

#prediction
logreg_optimal_pred_train = logreg_fit.predict(X_train)
logreg_optimal_pred_test = logreg_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

# AUC:
print('Training AUC Score', roc_auc_score(y_train, logreg_optimal_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, logreg_optimal_pred_test).round(4))


#---- Tuning LR
# Creating a hyperparameter grid
C_space      = pd.np.arange(0.001, 2, 0.1)


param_grid = {'C': C_space}


# Building the model object one more time
logreg_object = LogisticRegression(solver = 'lbfgs',
        random_state = 508)



# Creating a GridSearchCV object
logreg_grid = GridSearchCV(logreg_object,
                           param_grid,
                           cv = 3,
                           scoring = 'roc_auc',
                           return_train_score = False)



# Fit it to the training data
logreg_grid.fit(X_train, y_train) 


print("Tuned Logistic Regression Parameter:", logreg_grid.best_params_)
print("Tuned Logistic Regression Accuracy:",  logreg_grid.best_score_.round(4))

#----Logistic Regression Optimal Model

# Building our optimal model object
logreg_optimal = LogisticRegression(C = 0.501,
                                    solver = 'lbfgs',
                                    random_state = 508)

"""https://chrisalbon.com/machine_learning/logistic_regression/handling_imbalanced_classes_in_logistic_regression/ """

#fitting the model
logreg_optimal.fit(X_train, y_train)

#prediction
logreg_optimal_pred_train = logreg_optimal.predict(X_train)
logreg_optimal_pred_test = logreg_optimal.predict(X_test)


# Accuracy: Let's compare the testing score to the training score.
print('Training Score', logreg_optimal.score(X_train, y_train).round(4))
print('Testing Score:', logreg_optimal.score(X_test, y_test).round(4))

# AUC:
print('Training AUC Score', roc_auc_score(y_train, logreg_optimal_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, logreg_optimal_pred_test).round(4))

#---- Cross-validation
# Cross-Validating the model with three folds
cv_logreg_optimal = cross_val_score(logreg_optimal,
                                      got_train,
                                      got_target,
                                      cv = 3)

print('\nAverage: ',
      pd.np.mean(cv_logreg_optimal).round(3),
      '\nMinimum: ',
      min(cv_logreg_optimal).round(3),
      '\nMaximum: ',
      max(cv_logreg_optimal).round(3))

#-----
# Confusion Matrix
#-----
print(confusion_matrix(y_true = y_test,
                       y_pred = logreg_optimal_pred_test))
 

labels = ['isDead', 'isAlive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = logreg_optimal_pred_test)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            )


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


#-----------------------------------------------------------------------------
#      2. KNN Classifier
#-----------------------------------------------------------------------------
"""We don't have to worry about class imbalance in KNN Classifiers, as it
handles it internally. The first KNN model below is with the full model. """

# Running the neighbor optimization code with a small adjustment for classification
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


#Looking for the highest test accuracy
print(test_accuracy)

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

knn_clf = KNeighborsClassifier(n_neighbors = 5)

# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


# Generating Predictions based on the optimal KNN model
knn_clf_train = knn_clf_fit.predict(X_train)
knn_clf_pred = knn_clf_fit.predict(X_test)

# Let's compare the testing score to the training score.
print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))
#Training Score 0.7613
#Testing Score: 0.7487


# AUC:
print('Training AUC Score', roc_auc_score(y_train, knn_clf_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, knn_clf_pred).round(4))


"""
   Training AUC Score 0.8127
   Testing AUC Score 0.7024
Comment: It looks like the model is overfitting."""

#---- Cross-validation
# Cross-Validating the model with three folds
cv_logreg_optimal = cross_val_score(knn_clf,
                                      got_train,
                                      got_target,
                                      cv = 3)

print('\nAverage: ',
      pd.np.mean(cv_logreg_optimal).round(3),
      '\nMinimum: ',
      min(cv_logreg_optimal).round(3),
      '\nMaximum: ',
      max(cv_logreg_optimal).round(3))

#-----
# Confusion Matrix
#-----
print(confusion_matrix(y_true = y_test,
                       y_pred = knn_clf_pred))
 

labels = ['isDead', 'isAlive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = knn_clf_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            )


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the KNN classifier')
plt.show()

##############################################################################
#KNN with selected features
##############################################################################


got2 = got.select_dtypes(include = ['float64', 'int64'])
got2 = got2.drop(['S.No'], axis=1)

# 2)Columns that have missing value more than 80% 
got2 = got2.dropna(thresh=0.8*len(got), axis=1)

#creating the target variable
got_target = got2['isAlive']

got2.columns
got_train = got2.loc[: , ['book1_A_Game_Of_Thrones', 
                          'dateOfBirth', 'book2_A_Clash_Of_Kings', 
                          'age','book4_A_Feast_For_Crows', 'm_title', 
                          'isNWesteros','popularity', 'sumbooks', 'out_sumbooks',
                          'male', 'isEssosi', 'isIronborn', 'm_spouse', 'isNoble']]

#train test split
X_train, X_test, y_train, y_test = train_test_split(got_train,
                                                    got_target,
                                                    test_size = 0.1,
                                                    random_state = 508)



##############################################################################
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


#Looking for the highest test accuracy
print(test_accuracy)

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

knn_clf = KNeighborsClassifier(n_neighbors = 5)

# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

print('Training Score', knn_clf_fit.score(X_train, y_train).round(4)) 
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


# Generating Predictions based on the optimal KNN model
knn_clf_train = knn_clf_fit.predict(X_train)
knn_clf_pred = knn_clf_fit.predict(X_test)

# Let's compare the testing score to the training score.
print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


# AUC:
print('Training AUC Score', roc_auc_score(y_train, knn_clf_train).round(4)) #0.8033
print('Testing AUC Score', roc_auc_score(y_test, knn_clf_pred).round(4)) #0.7655

#---- Cross-validation
# Cross-Validating the model with three folds
cv_logreg_optimal = cross_val_score(knn_clf,
                                      got_train,
                                      got_target,
                                      cv = 3)

print('\nAverage: ',
      pd.np.mean(cv_logreg_optimal).round(3),
      '\nMinimum: ',
      min(cv_logreg_optimal).round(3),
      '\nMaximum: ',
      max(cv_logreg_optimal).round(3))

#-----
# Confusion Matrix
#-----
print(confusion_matrix(y_true = y_test,
                       y_pred = knn_clf_pred))
 

labels = ['isDead', 'isAlive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = knn_clf_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            )


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the KNN classifier')
plt.show()

###############################################################################
# 3) Building Classification Tree
###############################################################################
#got.columns

got2 = got.select_dtypes(include = ['float64', 'int64'])
got2 = got2.drop(['S.No'], axis=1)

# 2)Columns that have missing value more than 80% 
got2 = got2.dropna(thresh=0.8*len(got), axis=1)

#creating the target variable
got_target = got2['isAlive']

got_train = got2.loc[: , ['numDeadRelations', 'book1_A_Game_Of_Thrones', 
                          'dateOfBirth','book3_A_Storm_Of_Swords', 'book2_A_Clash_Of_Kings', 
                          'age','book4_A_Feast_For_Crows', 'm_title', 
                          'isNWesteros', 'isHousebig', 'isWesteros',
                          'popularity', 'sumbooks', 'out_sumbooks', 'male', 'isEssosi', 
                          'isIronborn', 'm_spouse']]


#train test split
X_train, X_test, y_train, y_test = train_test_split(
            got_train,
            got_target,
            test_size = 0.1,
            random_state = 508)



#Building the Classifier
c_tree = DecisionTreeClassifier(random_state = 508,
                                class_weight ='balanced')

#fitting the classifier
c_tree_fit = c_tree.fit(X_train, y_train)

#accuracy scores
print('Training Score', c_tree_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree_fit.score(X_test, y_test).round(4))

###############################################################################
# Hyperparameter Tuning with GridSearchCV
###############################################################################
# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 500)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space}


# Building the model object one more time
c_tree_2_hp = DecisionTreeClassifier(random_state = 508)

# Creating a GridSearchCV object
c_tree_2_hp_cv = GridSearchCV(c_tree_2_hp, param_grid, cv = 3)

# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Classifier Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Classifier Accuracy:", c_tree_2_hp_cv.best_score_.round(4))

#building the classifier with optimal parameters
c_tree2 = DecisionTreeClassifier(random_state = 508,
                                max_depth = 5,
                                min_samples_leaf = 10,
                                class_weight = 'balanced')

#fitting the model
c_tree2_fit = c_tree2.fit(X_train, y_train)

#Accuracy
print('Training Score', c_tree2_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree2_fit.score(X_test, y_test).round(4))

#prediction
ctree2_pred_train = c_tree2_fit.predict(X_train)
ctree2_pred_test = c_tree2_fit.predict(X_test)

#AUC scores
print('Training AUC Score', roc_auc_score(y_train, ctree2_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, ctree2_pred_test).round(4))   

#---- Cross-validation
# Cross-Validating the model with three folds
cv_ctree_2 = cross_val_score(c_tree2,
                                      got_train,
                                      got_target,
                                      cv = 3)

print('\nAverage: ',
      pd.np.mean(cv_ctree_2).round(3),
      '\nMinimum: ',
      min(cv_ctree_2).round(3),
      '\nMaximum: ',
      max(cv_ctree_2).round(3))


#-----
# Confusion Matrix
#-----
print(confusion_matrix(y_true = y_test,
                       y_pred = ctree2_pred_test))
 

labels = ['isDead', 'isAlive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = ctree2_pred_test)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            )


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the Decision Tree Classifier')
plt.show()



#defining function for important features
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

"""
plot_feature_importances(c_tree2,
                         train = X_train,
                         export = False) 
"""

###############################################################################
# 4) Building Random Forest
###############################################################################
got.columns


got2 = got.select_dtypes(include = ['float64', 'int64'])
got2 = got2.drop(['S.No'], axis=1)

# 2)Columns that have missing value more than 80% 
got2 = got2.dropna(thresh=0.8*len(got), axis=1)

#creating the target variable
got_target = got2['isAlive']

got2.columns
got_train = got2.loc[: , ['numDeadRelations', 'book1_A_Game_Of_Thrones', 
                          'dateOfBirth','book3_A_Storm_Of_Swords', 'book2_A_Clash_Of_Kings', 
                          'age','book4_A_Feast_For_Crows', 'm_title', 
                          'isNWesteros', 'isHousebig', 'isWesteros',
                          'popularity', 'sumbooks', 'out_sumbooks', 'male', 
                          'isEssosi', 'isIronborn', 'm_spouse']]


#train test split
X_train, X_test, y_train, y_test = train_test_split(got_train,
                                                    got_target,
                                                    test_size = 0.1,
                                                    random_state = 508)

#Random Forest Classifier
rf_clf=RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508, 
                                     class_weight= 'balanced')



#Train the model using the training sets y_pred=clf.predict(X_test)
rf_clf.fit(X_train,y_train)

# Fitting the models
forest_fit = rf_clf.fit(X_train, y_train)

# Scoring
print('Training Score', forest_fit.score(X_train, y_train).round(4))
print('Testing Score:', forest_fit.score(X_test, y_test).round(4))

#predict on test
rf_pred_test = rf_clf.predict(X_test)
rf_pred_train = rf_clf.predict(X_train)

#AUC Score
print('Training AUC Score', roc_auc_score(y_train,rf_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, rf_pred_test).round(4))

#---- Cross-validation
# Cross-Validating the model with three folds
cv_rf_clf = cross_val_score(rf_clf,
                                      got_train,
                                      got_target,
                                      cv = 3)

print('\nAverage: ',
      pd.np.mean(cv_rf_clf).round(3),
      '\nMinimum: ',
      min(cv_rf_clf).round(3),
      '\nMaximum: ',
      max(cv_rf_clf).round(3))


#-----
# Confusion Matrix
#-----

print(confusion_matrix(y_true = y_test,
                       y_pred = rf_pred_test))
 

labels = ['isDead', 'isAlive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = rf_pred_test)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            )


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the Random Forest classifier')
plt.show()

"""
#plot feature importance for random forest
plot_feature_importances(rf_clf,
                         train = X_train,
                         export = True)

It overrides on random forest's confusion matrix, when you run everything from 
the beginning. So if you want use the feature importance, please comment it out.
"""

##########################################
# GridsearchCV Hyperparameter Tuning
##########################################
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : pd.np.arange(4, 10, 2),
    'criterion' :['gini', 'entropy'],
    'min_samples_leaf': pd.np.arange(1, 150, 15), 
    'bootstrap': [True, False]
}


# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)

# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)


# Fit it to the training data
full_forest_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Random Forest Parameter:", full_forest_cv.best_params_)
print("Tuned Random Forest Accuracy:", full_forest_cv.best_score_.round(4))



#-------Optimal Model
rf_optimal = RandomForestClassifier(bootstrap = True,
                                    criterion = 'gini',
                                    min_samples_leaf = 1,
                                    max_depth = 8,
                                    n_estimators = 500,
                                    max_features = 'auto', 
                                    class_weight = 'balanced')


#fitting the model
rf_optimal.fit(X_train, y_train)


#predict on train
rf_optimal_pred_train = rf_optimal.predict(X_train)

#predict on test
rf_optimal_pred = rf_optimal.predict(X_test)


rf_optimal_predp = rf_optimal.predict_proba(X_test)

#Accuracy
print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))

#saving the scores
rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)

# AUC:
print('Training AUC Score', roc_auc_score(y_train,rf_optimal_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, rf_optimal_pred).round(4))

"""Interesting Result: Not tuned version gave me a better result than the optimal
model. """


#-----
# Confusion Matrix
#-----

print(confusion_matrix(y_true = y_test,
                       y_pred = rf_pred_test))
 

labels = ['isDead', 'isAlive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = rf_optimal_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            )


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the Random Forest classifier')
plt.show()

"""
#plot feature importance for random forest
plot_feature_importances(rf_optimal,
                         train = X_train,
                         export = False)
"""



#---- Cross-validation
# Cross-Validating the model with three folds
cv_rf_optimal = cross_val_score(rf_optimal,
                            got_train,
                            got_target,
                            cv = 3)

print('\nAverage: ',
      pd.np.mean(cv_rf_optimal).round(3),
      '\nMinimum: ',
      min(cv_rf_optimal).round(3),
      '\nMaximum: ',
      max(cv_rf_optimal).round(3))



