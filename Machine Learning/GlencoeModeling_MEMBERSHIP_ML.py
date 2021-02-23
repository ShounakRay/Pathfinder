# @Author: Shounak Ray <Ray>
# @Date:   19-Jul-2020 18:07:77:777  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: GlencoeModeling_MEMBERSHIP_ML.py
# @Last modified by:   Ray
# @Last modified time: 23-Feb-2021 16:02:75:757  GMT-0700
# @License: [Private IP]


import datetime
import itertools
import math
# Import packages
import random
import sys
import time as t

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lime import lime_tabular
from numpy import random as rand_np
from numpy.random import choice
from sklearn import (decomposition, ensemble, feature_selection, linear_model,
                     metrics, model_selection, preprocessing)

_ = """
#######################################################################################################################
########################################################### MASTER CONTROL VARIABLES ##################################
####################################################################################################################"""
# pd.set_option('display.expand_frame_repr', False)                               # Avoid low_memory warning
df_feature_list = ['gender', 'member_type', 'marital_status', 'membership_tenure', 'age', 'member_status']
df_contigencies = ['member_since', 'date_of_birth']
# Some useful statements:

# np.array(list(df_MEMBERSHIP.isna().sum()))
# # Iterate through all uniques in each column, get a sense of all unique values in cleaned DataFrame
# range = len(list(df_MEMBERSHIP.gender.unique()))
# for col_name in df_MEMBERSHIP.columns:
#     uniques = set(list(df_MEMBERSHIP['gender']))

# # Identify frequency of columns and delete ones with extremely high values (dimensionality reduction)
# names = []
# frequencies = []
# for i in range(4, len(df_MEMBERSHIP_ML.columns) - 8):
#     if(df_MEMBERSHIP_ML[df_MEMBERSHIP_ML.columns[i]][0] != 'object'):
#         curr_sum = df_MEMBERSHIP_ML[df_MEMBERSHIP_ML.columns[i]].sum()
#         if(curr_sum <= 100):
#             names.append(df_MEMBERSHIP_ML.columns[i])
#         else:
#             frequencies.append(curr_sum)
# df_MEMBERSHIP_ML = df_MEMBERSHIP_ML.drop(names, 1).reset_index(drop = True)
# sns.distplot(frequencies, bins = 100)
# df_MEMBERSHIP['member_type'].nunique()

# for index in range(len(df_MEMBERSHIP.columns)):
#     if(list(df_MEMBERSHIP.dtypes)[index] == 'object'):
#         df_MEMBERSHIP[df_MEMBERSHIP.columns[index]] = df_MEMBERSHIP[df_MEMBERSHIP.columns[index]].str.upper()
# Manually remove all columns KNOWN to have no impact, from all datasets
# df_MEMBERSHIP.drop(['primary_email', 'billing_cycle', 'statement_delivery_method', 'salutation_prefix',
#                     'secondary_email', 'activation_date', 'deactivation_date', 'slip_rate', 'address_name',
#                     'company', 'card_template', 'deactivation_date', 'activation_date', 'employer', 'business_phone',
#                     'member_number', 'prefix', 'suffix', 'nick_name', 'home_phone', 'cell_phone', 'source',
#                     'address_line_1', 'address_line_2', 'address_line_3', 'billing_email', 'card_holder_name',
#                     'first_name', 'country', 'middle_name', 'last_name', 'city', 'state', 'zip'],
#                    inplace=True, axis=1)

_ = """
#######################################################################################################################
########################################################### FUNCTIONS FOR MAIN FILE ###################################
####################################################################################################################"""

# 1-inactivated: Removes all columns which have only one distinct value. Reasoning: not useful for AffinityNet linkages
# 2-inactivated: Identify columns with NaN values, remove them from df (NOT INDIVIDUAL ROWS). Reasoning: useless, fatal


def clean_df(df):
    for col in df.columns:
        if (len(df[col].unique()) == 1):
            df.drop(col, inplace=True, axis=1)
    df = df.dropna(inplace=False)
    # na_values_in_df = np.array(list(df.isna().sum()))
    # while (len(list(filter((0).__ne__, na_values_in_df))) > 1):
    #     index_of_highest_na = np.argmax(na_values_in_df)
    #     df = df.drop(df.columns[index_of_highest_na], inplace = False, axis = 'columns')
    #     na_values_in_df = np.array(list(df.isna().sum()))
    return df

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


def forward_selected(data, response):
    """
    Linear model designed by forward selection.
    Parameters:
    > data : pandas DataFrame with all possible predictors and response
    > response: string, name of response column in data
    Returns:
    > model: an "optimal" fitted statsmodels linear model
            with an intercept
            selected by forward selection
            evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


_ = """
#######################################################################################################################
######################################################## MAIN FILE: CLEANING AND ANALYTICS ############################
####################################################################################################################"""


# Specify type of each column in this line...
df_EVENTS = pd.read_csv("Data/EVENTS.csv", low_memory=False)
df_MEMBERSHIP = pd.read_csv("Data/MEMBERSHIP.csv", low_memory=False)
df_RESERVATIONS = pd.read_csv("Data/RESERVATIONS.csv", low_memory=False)
df_SALES = pd.read_csv("Data/SALES.csv", low_memory=False)

# REMOVE USELESS ROWS
df_MEMBERSHIP.dropna(subset=df_feature_list + df_contigencies, inplace=True)

# Assign col types. Dependent on final ML feature set -- FIRST OCCURRENCE
df_MEMBERSHIP['gender'] = df_MEMBERSHIP['gender'].astype(str)
df_MEMBERSHIP['age'] = df_MEMBERSHIP['age'].astype(int)
df_MEMBERSHIP['membership_tenure'] = df_MEMBERSHIP['membership_tenure'].astype(int)
df_MEMBERSHIP['member_type'] = df_MEMBERSHIP['member_type'].astype(str)
df_MEMBERSHIP['marital_status'] = df_MEMBERSHIP['marital_status'].astype(str)
df_MEMBERSHIP['member_status'] = df_MEMBERSHIP['member_status'].astype(str)

_ = """
#######################################################################################################################
########################################################### DATA CLEANING PROCEDURES ##################################
################################################################# MEMBERSHIP ##########################################
####################################################################################################################"""

# REMOVE MOST NON-FINAL COLUMNS
df_MEMBERSHIP = df_MEMBERSHIP[df_feature_list + df_contigencies]

# Make most import applicable strings uppercase (to avoid formatting-based repetitions)
df_MEMBERSHIP['member_type'] = df_MEMBERSHIP['member_type'].str.upper()
df_MEMBERSHIP['marital_status'] = df_MEMBERSHIP['marital_status'].str.upper()
df_MEMBERSHIP['member_status'] = df_MEMBERSHIP['member_status'].str.upper()
df_MEMBERSHIP['gender'] = df_MEMBERSHIP['gender'].str.upper()

# Delete useless, placeholder member_since rows
df_MEMBERSHIP = df_MEMBERSHIP[df_MEMBERSHIP.member_since != '1900-01-01']
df_MEMBERSHIP = df_MEMBERSHIP[df_MEMBERSHIP.member_since != '1899-12-30']
df_MEMBERSHIP = df_MEMBERSHIP[df_MEMBERSHIP.member_since != '1901-01-01']
df_MEMBERSHIP = df_MEMBERSHIP[df_MEMBERSHIP.member_since != '1905-01-01']

# Re-structure membership tenure according to current datetime
today = datetime.datetime.now()
df_MEMBERSHIP['member_since'] = pd.to_datetime(df_MEMBERSHIP['member_since'])
df_MEMBERSHIP['date_of_birth'] = pd.to_datetime(df_MEMBERSHIP['date_of_birth'])
df_MEMBERSHIP['membership_tenure'] = [int((today - x).days / 365.25) for x in df_MEMBERSHIP['member_since']]
df_MEMBERSHIP['age'] = [int((today - x).days / 365.25) for x in df_MEMBERSHIP['date_of_birth']]
df_MEMBERSHIP.drop(df_contigencies, inplace=True, axis=1)

# Set str.compatible columns as strings
df_MEMBERSHIP['gender'] = df_MEMBERSHIP['gender'].astype(str)
df_MEMBERSHIP['age'] = df_MEMBERSHIP['age'].astype(int)
df_MEMBERSHIP['membership_tenure'] = df_MEMBERSHIP['membership_tenure'].astype(int)
df_MEMBERSHIP['member_type'] = df_MEMBERSHIP['member_type'].astype(str)
df_MEMBERSHIP['marital_status'] = df_MEMBERSHIP['marital_status'].astype(str)
df_MEMBERSHIP['member_status'] = df_MEMBERSHIP['member_status'].astype(str)

df_MEMBERSHIP.reset_index(inplace=True)

# Dimensionality reduction (feature engineering) (split member_type into DC and GC)
for row_index in list(df_MEMBERSHIP.index):
    if("DC" in df_MEMBERSHIP['member_type'][row_index]):
        df_MEMBERSHIP['member_type'].iloc[row_index] = "DC"
    elif("GC" in df_MEMBERSHIP['member_type'][row_index]):
        df_MEMBERSHIP['member_type'].iloc[row_index] = "GC"

# Drop auto-added index column, not required.
df_MEMBERSHIP.drop('index', 1, inplace=True)

# One-hot encode the membership data, structure the data so its usable for machine learning
df_MEMBERSHIP_ML = dummy_check(df_MEMBERSHIP, list(df_MEMBERSHIP.columns))
df_MEMBERSHIP_ML.drop(['marital_status_0'], 1, inplace=True)
df_MEMBERSHIP_ML.drop(['marital_status_DOMESTIC PARTNERSHIP', 'marital_status_SINGLE',
                       'member_status_ACTIVE - NO CHARGE', 'member_status_SUSPENDED'], 1, inplace=True)

# # DEBUGGING PRUPOSES ONLY, CALCULATE FREQUENCY OF TYPE OF FEATURE
# df_MEMBERSHIP_ML['gender_FEMALE'].sum()
# df_MEMBERSHIP_ML['gender_MALE'].sum()
# df_MEMBERSHIP_ML['member_type_DC'].sum()
# df_MEMBERSHIP_ML['member_type_GC'].sum()
# df_MEMBERSHIP_ML['marital_status_DOMESTIC PARTNERSHIP'].sum() # 1
# df_MEMBERSHIP_ML['marital_status_MARRIED'].sum()
# df_MEMBERSHIP_ML['marital_status_SINGLE'].sum() # 75
# df_MEMBERSHIP_ML['member_status_ACTIVE'].sum()
# df_MEMBERSHIP_ML['member_status_ACTIVE - NO CHARGE'].sum() # 33
# df_MEMBERSHIP_ML['member_status_SUSPENDED'].sum() # 6

_ = """
#######################################################################################################################
################################################# MACHINE LEARNING PROCESS FOR ALL DATASETS ###########################
####################################################################################################################"""

# PCA, K-mean clustering, dimensionality reduction techniques (look at features inside the feature)
# Initialize the linear model
model = linear_model.LinearRegression()
model_lasso = linear_model.Lasso(alpha=-4.8)
# Split and format the total dataset into training and testing sets
df_MEMBERSHIP_training, df_MEMBERSHIP_testing = np.split(df_MEMBERSHIP_ML, [int(.7 * len(df_MEMBERSHIP_ML))])
df_MEMBERSHIP_testing = df_MEMBERSHIP_testing.reset_index(drop=True)

# Split and format testing sets into regressors and actual values
df_MEMBERSHIP_testing_regressors = df_MEMBERSHIP_testing.drop('membership_tenure', 1).values
df_MEMBERSHIP_testing_actual = df_MEMBERSHIP_testing['membership_tenure'].to_list()

# Split and format training sets into regressors and predictors (to be fed into model)
df_MEMBERSHIP_training_regressors = df_MEMBERSHIP_training.drop('membership_tenure', 1).values
df_MEMBERSHIP_training_predictor = df_MEMBERSHIP_training['membership_tenure'].to_list()

model_stepwise_regression = forward_selected(df_MEMBERSHIP_training, 'membership_tenure')
model_lasso.fit(df_MEMBERSHIP_training_regressors, df_MEMBERSHIP_training_predictor)
model.fit(df_MEMBERSHIP_training_regressors, df_MEMBERSHIP_training_predictor)

r2 = model.score(df_MEMBERSHIP_training_regressors, df_MEMBERSHIP_training_predictor)
r2_lasso = model_lasso.score(df_MEMBERSHIP_training_regressors, df_MEMBERSHIP_training_predictor)

y_pred = list(model.predict(df_MEMBERSHIP_testing_regressors))
y_pred_lasso = list(model_lasso.predict(df_MEMBERSHIP_testing_regressors))

print("MEAN ABSOLUTE ERROR")
sk.metrics.mean_absolute_error(df_MEMBERSHIP_testing_actual, y_pred_lasso)
print("PREDICTED TENURE")
sns.distplot(y_pred_lasso)
print("ACTUAL TENURE")
sns.distplot(df_MEMBERSHIP_testing_actual)
print("DIFFERENCE IN TENURES")
sns.distplot(np.array(y_pred_lasso) - np.array(df_MEMBERSHIP_testing_actual), bins=8)

# AffinityNet Processing:
# 1: Identify connection between person and event/loc based on time attributes (this event/loc then that event/loc).
#    Events = nodes
# 2: Identify connection between reservation <-> cancelation (res. then res // res then cancel).

# >>> DEBUGGING INFORMATION
# Need more information about the time columns, and event number/name/location relationship
len(list(df_MEMBERSHIP.gender.unique()))
len(list(df_SALES.member_name.unique()))
