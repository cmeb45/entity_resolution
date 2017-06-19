from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from dateutil.parser import parse
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from scipy.stats import randint, expon
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report
from sys import argv, exit, stderr

# Custom Functions
def time_standardize(in_string):
    """Standardizes time by extracting hours and minutes, and then converting to minutes
    """
    
    # If input is not string, return input
    if isinstance(in_string, basestring) is False:
        return in_string
    
    # Use regular expression to parse hours and minutes, or minutes
    time_search = re.search("(([0-9]) ho?u?rs?[,.] )?([0-9]{1,3}) min\.?u?t?e?s?",
                            in_string, flags=re.IGNORECASE)
    if time_search is not None:
        hours = time_search.group(2)
        if hours is not None:
            hours = int(hours)
        else:
            hours = 0
        minutes = int(time_search.group(3))
        return (hours*60)+minutes
    else:
        # Use regular expression to parse hours alone
        time_search = re.search("([0-9]) ho?u?rs?[,.]", in_string, flags=re.IGNORECASE)
        if time_search is not None:
            hours = int(time_search.group(1))
            return hours*60
        else:
            return in_string

def is_date(string):
    try: 
        parse(string)
        return True
    except ValueError:
        return False
    except AttributeError:
        return False

def time_diff(row):
    return abs(row['time_x'] - row['time_y'])

def dir_diff(row):
    return normalized_damerau_levenshtein_distance(row['director_x'],row['director_y'])

def jaccard_similarity(query, document):
    query = query.split()
    document = document.split()
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return 1.0*len(intersection)/len(union)

def star_diff(row):
    query = row['star']
    doc = row['star1']+' '+row['star2']+' '+row['star3']+' '+row['star4']+' '+row['star5']+' '+row['star6']
    return jaccard_similarity(query, doc)

def prepare_datasets(amazon,rot,test_data):
    test_amazon = pd.merge(test_data,amazon, how='inner', left_on='id1', right_on='id')
    X = pd.merge(test_amazon,rot,how='left',left_on='id 2', right_on='id')
    X = X.drop(['id_x','id_y'], axis=1)
    X['time_diff'] = X.apply(time_diff, axis = 1)
    X['dir_diff'] = X.apply(dir_diff, axis = 1)
    X['star_diff'] = X.apply(star_diff, axis = 1)
    X = X.drop(['time_x','director_x','star','time_y','director_y',
                       'star1','star2','star3','star4','star5','star6'], axis=1)
    return(X)

if __name__ == '__main__':
    if len(argv) < 2:
        print("Usage: python %s TESTFILE" %
              argv[0], file=stderr)
        exit(1)

    # Load Data
    amazon = pd.read_csv('amazon.csv')
    rot = pd.read_csv('rotten_tomatoes.csv')
    train = pd.read_csv('train.csv')
    holdout = pd.read_csv('holdout.csv')
    test = pd.read_csv(argv[1])


    # Data Cleaning
    ## Standardize Time Variables
    amazon['time'] = amazon['time'].apply(time_standardize)
    rot['time'] = rot['time'].apply(time_standardize)


    ## Clean Time Values that are Dates or nan
    rot.loc[rot['time'].isnull(),'time'] = 0
    amazon.loc[(amazon['time'].isnull() | amazon['time'].apply(is_date)),'time'] = 0


    ## Remove NaN from Star Variables
    star_vars = ['star1','star2','star3','star4','star5','star6']
    for var in star_vars:
        rot[var] = rot[var].replace(np.nan, '', regex=True)

    amazon['star'] = amazon['star'].replace(np.nan, '', regex=True)


    ## Remove Extra Variables
    amazon = amazon.drop(['cost'], axis=1)
    rot = rot.drop(['rotten_tomatoes','audience_rating',
                    'review1','review2','review3','review4','review5','year'], axis=1)

    # Join Training Datasets
    train_amazon = pd.merge(train,amazon, how='inner', left_on='id1', right_on='id')
    X = pd.merge(train_amazon,rot,how='inner',left_on='id 2', right_on='id')

    ## Extract data labels
    y = X.ix[:, 'gold'].values
    y = y.astype(int)
    X = X.drop(['id_x','id_y','gold'], axis=1)

    ## Apply Time Difference Function
    X['time_diff'] = X.apply(time_diff, axis = 1)

    ## Apply Levenshtein Distance Function on Directors
    X['dir_diff'] = X.apply(dir_diff, axis = 1)

    ## Apply Jaccard Distance Function on Stars
    X['star_diff'] = X.apply(star_diff, axis = 1)

    ## Drop Extra Variables
    X = X.drop(['time_x','director_x','star','time_y','director_y',
                           'star1','star2','star3','star4','star5','star6'], axis=1)

    feature_set = ['time_diff','dir_diff','star_diff']


    # Partition into Training and Holdout Data
    X_train, X_hold, y_train, y_hold = train_test_split(
         X, y, test_size=0.1, random_state=42)


    # Create a Random Forest Classifier
    ## Run Randomized Search for Hyperparameter Optimization
    cv_call = StratifiedKFold(y_train,n_folds=10)
    # Specify cross-validation settings
    param_dist = {"n_estimators": randint(5, 500),
                 "class_weight": ["balanced","balanced_subsample"]}
    n_iter_search = 30
    clf = RandomForestClassifier(random_state=42,n_jobs=-1)
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search,cv=cv_call,
                                       scoring='f1')

    random_search = random_search.fit(X_train[feature_set], y_train)

    ## Retrieve Optimal Hyperparameter Values from Random Search
    best_parameters, score, _ = max(random_search.grid_scores_, key=lambda x: x[1])
    clf = RandomForestClassifier(random_state=42,n_jobs=-1,
                    n_estimators=187,#best_parameters["n_estimators"],
                    class_weight="balanced_subsample")#best_parameters["class_weight"])

    # best_parameters["n_estimators"]=187,best_parameters["class_weight"]="balanced_subsample"

    ## Run Model with Optimized Parameters on Entire Training Dataset
    clf = clf.fit(X[feature_set], y)

    # Join Test Datasets
    X_test = prepare_datasets(amazon,rot,test)
    preds_test = clf.predict(X_test[feature_set])
    X_test_preds = pd.DataFrame(preds_test)
    X_test_preds = X_test_preds.rename(columns = {0:'gold'})
    X_test_preds.to_csv("gold.csv", index = False)