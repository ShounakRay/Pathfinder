# @Author: Shounak Ray <Ray>
# @Date:   28-Jul-2020 08:07:08:089  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: GlencoeModeling_SALES_ML.py
# @Last modified by:   Ray
# @Last modified time: 23-Feb-2021 16:02:04:046  GMT-0700
# @License: [Private IP]


import itertools
import math
# Import packages
import random
import sys
import time as t
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf
from autocorrect import Speller
from lime import lime_tabular
from numpy import random as rand_np
from numpy.random import choice
from sklearn import (decomposition, ensemble, feature_selection, linear_model,
                     metrics, model_selection, preprocessing)

_ = """
#######################################################################################################################
########################################################### MASTER CONTROL VARIABLES ##################################
####################################################################################################################"""
df_feature_list = ['member_name', 'status', 'date', 'item_group', 'item_name', 'total', 'service_provider']
df_contigencies = []

# Self-explanatory, recognize the data-type


def utils_recognize_type(dtf, col, max_cat=20):
    if ((dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat)):
        return "cat"
    else:
        return "num"

# Visualize a heatmap of the datatype, organized by NaN, categorical, and numeric values


def visDatatypes(dtf):
    plt.figure(figsize=(17, 13))
    dic_cols = {col: utils_recognize_type(df_MEMBERSHIP, col, max_cat=20) for col in dtf.columns}
    heatmap = dtf.isnull()
    for k, v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, cbar=False, cmap='flag').set_title('Dataset Overview')
    print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")
    plt.show()

# One-hot encode the data to be usable for machine learning


def dummy_check(df, features):
    df = df[features]
    if len(list(df.select_dtypes(include='object').columns)) != 0:
        dff = pd.get_dummies(df, dummy_na=False)
    else:
        dff = df
    return dff


alphabet = 'abcdefghijklmnopqrstuvwxyz'


def autocorrect(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


_ = """
#######################################################################################################################
################################################## DATA PROCESSING ####################################################
####################################################################################################################"""

df_SALES = pd.read_csv("Data/SALES.csv", low_memory=False)

# REMOVE MOST NON-FINAL COLUMNS
df_SALES = df_SALES[df_feature_list + df_contigencies]
# DROP ALL NA ROWS
df_SALES.dropna(inplace=True)
# UPPER-CASE ALL VALUES
df_SALES['member_name'] = df_SALES['member_name'].str.upper()
df_SALES['status'] = df_SALES['status'].str.upper()
df_SALES['item_group'] = df_SALES['item_group'].str.upper()
df_SALES['item_name'] = df_SALES['item_name'].str.upper()
df_SALES['service_provider'] = df_SALES['service_provider'].str.upper()
# SET COL TYPE OF NUMBER VALUES
df_SALES['total'] = df_SALES['total'].astype(float)
# Converts date col to datetime for sorting and future comparison purposes
df_SALES['date'] = pd.to_datetime(df_SALES['date'])
df_SALES.sort_values(by='date', inplace=True)
df_SALES = df_SALES.reset_index(inplace=False).drop('index', axis=1, inplace=False)
# AUTOCORRECT GRAMMAR TO POSSIBLY ELIMINATE/REDUCE REPETITION --> Very Difficult

df_SALES.drop(['member_name', 'date', 'item_name', 'total'], 1, inplace=True)

set(df_SALES['status'].to_list())
# Dimensionality reduction (feature engineering) (split member_type into DC and GC)
for row_index in list(df_SALES.index):
    if("DC" in df_SALES['service_provider'][row_index]):
        df_SALES['service_provider'].iloc[row_index] = 0  # DC Provider
    else:
        df_SALES['service_provider'].iloc[row_index] = 1  # Private

    if("ACTIVE - NO CHARGE" in df_SALES['status'][row_index]):
        df_SALES['status'].iloc[row_index] = 0
    elif("ACTIVE" in df_SALES['status'][row_index]):
        df_SALES['status'].iloc[row_index] = 1
    elif("BILLED" in df_SALES['status'][row_index]):
        df_SALES['status'].iloc[row_index] = 2
    elif("INACTIVE" in df_SALES['status'][row_index]):
        df_SALES['status'].iloc[row_index] = 3
    else:
        df_SALES['status'].iloc[row_index] = 4

    if('FOOD' in df_SALES['item_group'][row_index]):
        df_SALES['item_group'].iloc[row_index] = 0  # Food
    elif('WINE' in df_SALES['item_group'][row_index]
         or 'LIQUOR' in df_SALES['item_group'][row_index]
         or 'BEER' in df_SALES['item_group'][row_index]
         or 'BEVERAGES' in df_SALES['item_group'][row_index]
         or 'BEER' in df_SALES['item_group'][row_index]):
        df_SALES['item_group'].iloc[row_index] = 1  # Drinks
    elif('FITNESS' in df_SALES['item_group'][row_index]
         or 'TENNIS' in df_SALES['item_group'][row_index]
         or 'CURLING' in df_SALES['item_group'][row_index]
         or 'SWIMMING' in df_SALES['item_group'][row_index]
         or 'SWIM' in df_SALES['item_group'][row_index]
         or 'SQUASH' in df_SALES['item_group'][row_index]
         or 'SKATING' in df_SALES['item_group'][row_index]
         or 'BODYWORK' in df_SALES['item_group'][row_index]
         or 'BOWLING' in df_SALES['item_group'][row_index]
         or 'BADMINTON' in df_SALES['item_group'][row_index]
         or 'AQUATICS' in df_SALES['item_group'][row_index]
         or 'SPORTS' in df_SALES['item_group'][row_index]
         or 'RACQUETS' in df_SALES['item_group'][row_index]
         or 'CLIMBING' in df_SALES['item_group'][row_index]):
        df_SALES['item_group'].iloc[row_index] = 2  # Sports
    elif('PHYSIOTHERAPY' in df_SALES['item_group'][row_index]
         or 'SPA' in df_SALES['item_group'][row_index]
         or 'HEALTH' in df_SALES['item_group'][row_index]
         or 'MASSAGE' in df_SALES['item_group'][row_index]):
        df_SALES['item_group'].iloc[row_index] = 3  # Relaxation
    else:
        df_SALES['item_group'].iloc[row_index] = 4  # Merchandise/Other

# Storing frequencies of unique values in df
uniques = []
for col in df_SALES.columns:
    uniques.append((col, len(set(df_SALES[col])), df_SALES[col].value_counts().to_dict()))


_ = """
#######################################################################################################################
################################################### MACHINE LEARNING ##################################################
####################################################################################################################"""

df_SALES_ML = dummy_check(df_SALES, list(df_SALES.columns))
classification_class = 'item_group'
regression_class = 'total'

############
model = linear_model.LinearRegression()
model_lasso = linear_model.Lasso(alpha=0.1)
############
model_tree = sk.tree.DecisionTreeClassifier()
model_cent = sk.neighbors.NearestCentroid()
############
# Split and format the total dataset into training and testing sets
df_SALES_training, df_SALES_testing = np.split(df_SALES, [int(.7 * len(df_SALES))])
df_SALES_testing = df_SALES_testing.reset_index(drop=True)

# Split and format testing sets into regressors and actual values
df_SALES_testing_regressors = df_SALES_testing.drop(classification_class, 1).values
df_SALES_testing_actual = df_SALES_testing[classification_class].to_list()

# Split and format training sets into regressors and predictors (to be fed into model)
df_SALES_training_regressors = df_SALES_training.drop(classification_class, 1).values
df_SALES_training_predictor = df_SALES_training[classification_class].to_list()

###################################
########## CLASSIFICATION #########
###################################
model_tree.fit(df_SALES_training_regressors, set(df_SALES_training_predictor))
model_cent.fit(df_SALES_training_regressors, list(df_SALES_training_predictor))
model_cent.score(df_SALES_testing_regressors, df_SALES_testing_actual)
###################################
########## REGRESSION #############
###################################
model_lasso.fit(df_SALES_training_regressors, df_SALES_training_predictor)
model.fit(df_SALES_training_regressors, df_SALES_training_predictor)
r2 = model.score(df_SALES_training_regressors, df_SALES_training_predictor)
r2_lasso = model_lasso.score(df_SALES_training_regressors, df_SALES_training_predictor)
y_pred = list(model.predict(df_SALES_testing_regressors))
y_pred_lasso = list(model_lasso.predict(df_SALES_testing_regressors))
print("MEAN ABSOLUTE ERROR")
sk.metrics.mean_absolute_error(df_SALES_testing_actual, y_pred_lasso)
sk.metrics.r2_score(df_SALES_testing_actual, y_pred_lasso)
print("PREDICTED $$$")
plt.hist(y_pred_lasso, bins=30, range=[-10, 300])
print("ACTUAL $$$")
plt.hist(df_SALES_testing_actual_lasso, bins=30, range=[-10, 300])
print("DIFFERENCE IN $$$")
diff = np.array(y_pred) - np.array(df_SALES_testing_actual)
scipy.stats.describe(diff)
plt.hist(diff, bins=30, range=[-20, 20])
list(np.sort(diff))
list(-np.sort(-diff))
###################################
###################################
###################################

_ = """
#######################################################################################################################
################################################# NEAT VISUALIZATIONS #################################################
####################################################################################################################"""
# Fun visualization of average transactions per day
days = np.sort(list(set(list(df_SALES['date']))))
avg_transactions_per_day = []
total_people = []
unique_people = []
for day in days:
    df_SALES_exp = df_SALES[df_SALES['date'] == pd.to_datetime(day)]
    unique_people.append(len(set(df_SALES_exp['member_name'])))
    total_people.append(len(df_SALES_exp['date'].to_list()))

# avg_transactions_per_day = [t / u for t, u in zip(total_people, unique_people)]
# plt.figure(figsize = (10,5))
# plt.plot(days, avg_transactions_per_day)
# plt.show()
# ########


# unique_nodes = [node_name.replace('"','\\"') for node_name in unique_nodes]
# PRINT FINAL GRAPH TO JSON FILE
# data = nx.readwrite.json_graph.node_link_data(G, dict(source='from', target='to', name='id', links='label'))
# with open('GlencoeGraphData.json', 'w') as f:
#     json.dump(data, f, indent = 4, sort_keys = True)


# # ADD THE WEIGHTED EDGES TO GRAPH AND PLOT
# G.add_weighted_edges_from(edge_weights_3t)
# G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
# G.graph['graph'] = {'scale': '3'}
# A = nx.drawing.nx_agraph.to_agraph(G)
# A.draw('GlencoeSALES_full.png', prog = 'dot')
#
# # CONDENSE THE GRAPH AND PLOT
# cutoff = np.sort(list(set(list(edge_weights.values()))))[::-1][top_CONST - 1]
# top = [edge for edge in G.edges(data = True)
#        if edge[2]['weight'] > cutoff]
# G.clear()
# G.add_edges_from(top)
# G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
# G.graph['graph'] = {'scale': '3'}
# A = nx.drawing.nx_agraph.to_agraph(G)
# A.draw('GlencoeSALES_condensed.png', prog = 'dot')


# node_values = [{w:node_values[w]} for w in unique_nodes]
