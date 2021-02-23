# @Author: Shounak Ray <Ray>
# @Date:   24-Jul-2020 16:07:88:888  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: GlencoeModeling_SALES_AffinityNet.py
# @Last modified by:   Ray
# @Last modified time: 23-Feb-2021 16:02:29:296  GMT-0700
# @License: [Private IP]


import collections
import os
import webbrowser
# Import packages
# from datetime import datetime, timedelta
# from itertools import chain
from pathlib import Path

import pandas as pd
from G_to_J import graph_preproccessing, graph_to_json

# Big Picture Plan:
# 1. Dataframe as input
# 2. Make Dataframe
# 3. Create a card
# Mini Next Steps:
# 1. Make much more specific groups (item_group AND (SEPARATE) service_provider further stratification)
# 2. Generalize Data Processing Section
# 3. Think if you should make cross-group lines black and dotted (what do they actually mean)
# 4. Take screenshots and annotate them (send to Rob just for his thoughts)
# 5. Organize all files and folders in ZIP and send to Rob

# Tracker of all HTML Files with no nodes (not enough data)
out = []


def populate_files(browser_open, df, edge_class, strat_class, top_CONST, VIS_parameters, shape_name, VIS_location,
                   init_strat_name='', prim_dir_name=''):
    HTML_FILES = []
    HTML_FILE_NAMES = []
    PRIMARY_DIRECTORY_NAMES = []

    # Make HTML Files and Directory Names
    for feature_num in range(len(edge_class)):
        for isolation_num in range(len(top_CONST)):
            dict_stratClass_edgeClass, edge_weights_3t, d_keys_ind, d_values_ind, color_rules = graph_preproccessing(
                df, edge_class[feature_num], strat_class[feature_num],
                'member_name', top_CONST[isolation_num], 'detailed')
            if(dict_stratClass_edgeClass == 0):
                out.append((prim_dir_name, edge_class[feature_num], strat_class, top_CONST[isolation_num]))
                continue
            HTML_OUTPUT = graph_to_json(edge_class[feature_num], strat_class[feature_num], top_CONST[isolation_num],
                                        VIS_parameters, shape_name, dict_stratClass_edgeClass, d_keys_ind,
                                        d_values_ind, edge_weights_3t, color_rules, VIS_location, init_strat_name)
            HTML_NAME = "Glencoe_AUTO-" + \
                str(edge_class[feature_num]) + "##" + str(strat_class[feature_num]) + \
                "-" + str(top_CONST[isolation_num]) + ".html"
            DIRECTORY_NAME = str(edge_class[feature_num]) + " + " + str(strat_class[feature_num])

            HTML_FILES.append(HTML_OUTPUT)
            HTML_FILE_NAMES.append(HTML_NAME)
            PRIMARY_DIRECTORY_NAMES.append(DIRECTORY_NAME)

    # Create directories, write to file, open in default broswer [chrome]
    for iter_num in range(len(HTML_FILES)):
        most_specific_path = str(prim_dir_name) + \
            str(PRIMARY_DIRECTORY_NAMES[iter_num]) + "/" + str(HTML_FILE_NAMES[iter_num])
        if not os.path.exists(str(prim_dir_name + PRIMARY_DIRECTORY_NAMES[iter_num])):
            os.makedirs(str(prim_dir_name + PRIMARY_DIRECTORY_NAMES[iter_num]))
        if not Path(most_specific_path).is_file():
            with open(most_specific_path, "w") as file:
                file.write(HTML_FILES[iter_num])
        if(browser_open):
            webbrowser.open_new_tab("file://" + os.path.realpath(most_specific_path))

    return 0


_ = """
#######################################################################################################################
########################################################### MASTER CONTROL VARIABLES ##################################
####################################################################################################################"""
df_feature_list = ['member_name', 'status', 'item_group', 'item_name', 'service_provider']
df_contigencies = ['age', 'date', 'member_number']
# how many top nodes reported (isolation)
top_CONST = [100, 75, 50, 25, 10, 0]
# edge_class = ['item_group', 'item_name', 'item_name', 'item_group']             # node/edge-connection attributes
# strat_class = ['service_provider', 'item_group', 'service_provider', 'status']  # class for color stratification
buttons = False                 # clustering options for the graph
groups_attribute = True         # specify group for each node for auto-coloring
groups_option = True           # Default coloring already takes place, extra clustering functions are absent/inactive
physics_option = True           # specify physics for graph motion
node_labels = True              # specify labels for each node
manipulation_option = True      # specify manipulation values
interaction_option = True       # specify interaction values
layout_option = True            # specify layout appearance
shape_name = 'dot'              # shape of each node
shapes = True                   # specify if shape should be set
node_values = False             # specify node size (won't do anything though)
heading_format = True           # Heading of file showing details of Graph
VIS_parameters = [buttons, groups_attribute, groups_option, physics_option, node_labels,
                  manipulation_option, interaction_option, layout_option, shapes, node_values, heading_format]
VIS_location = '/Users/Ray/Documents/Python/Glencoe/'
OPEN_TAB = False                 # Open page in chrome?


edge_class = ['service_provider']
strat_class = ['item_group']
_ = """
#######################################################################################################################
################################################## DATA PROCESSING ####################################################
####################################################################################################################"""
# >>> DF_SALES PROCESSING
df_SALES = pd.read_csv("Data/SALES.csv", low_memory=False)
# REMOVE MOST NON-FINAL COLUMNS
df_SALES = df_SALES[df_feature_list + df_contigencies]
# DROP ALL NA ROWS
df_SALES.dropna(inplace=True)
df_SALES.isna().sum()
# STRIP ALL STRINGS
df_obj = df_SALES[['member_name', 'service_provider']]
df_SALES[df_obj.columns] = df_obj.apply(lambda x: x.str.strip().replace('  ', ' '))
# df_SALES.to_csv('Semi-processed_SALES.csv')
# UPPER-CASE ALL VALUES (TO ENSURE MINIMAL CLASS OVERLAP)
df_SALES['member_name'] = df_SALES['member_name'].str.upper()
df_SALES['status'] = df_SALES['status'].str.upper()
df_SALES['item_group'] = df_SALES['item_group'].str.upper()
df_SALES['item_name'] = df_SALES['item_name'].str.upper()
df_SALES['service_provider'] = df_SALES['service_provider'].str.upper()
# Converts date col to datetime for sorting and future comparison purposes (do i need this?)
df_SALES['date'] = pd.to_datetime(df_SALES['date'])
df_SALES.sort_values(by='date', inplace=True)
df_SALES = df_SALES.reset_index(inplace=False).drop('index', axis=1, inplace=False)


_ = """
#######################################################################################################################
################################################## AFFINITY NET PREPARATION ###########################################
####################################################################################################################"""

# For the whole dataset (GENERICS)
populate_files(OPEN_TAB, df_SALES, edge_class, strat_class, top_CONST,
               VIS_parameters, shape_name, VIS_location, '', 'GENERICS/')


# EXT-STRAT
mins_maxes = [[0.0, 6.0], [7.0, 18.0], [19.0, 45.0], [46.0, 60.0], [61.0, max(df_SALES['age'])]]

# init_segreg_names = ['service_provider', 'item_group', 'item_group']
init_segreg_names = ['service_provider']
for segreg_num in range(len(init_segreg_names)):
    strat_ft_name = init_segreg_names[segreg_num]
    strat_one_values = dict(collections.Counter(list(df_SALES[strat_ft_name])))
    strat_one_values = {k: v for k, v in sorted(strat_one_values.items(), key=lambda item: item[1], reverse=True)}
    if(segreg_num == 0):
        edge_class = ['item_name']
        strat_class = ['item_group']
    elif(segreg_num == 1):
        edge_class = ['item_name']
        strat_class = ['service_provider']
    elif(segreg_num == 2):
        edge_class = ['item_name']
        strat_class = ['item_group']
    for strat_num in range(len(strat_one_values)):
        top_CONST = [0, 10, 25, 50, 75, 100]
        for age_range in mins_maxes:
            curr_strat_one_value = list(strat_one_values.keys())
            df_SALES_filtered = df_SALES[(df_SALES[strat_ft_name] == curr_strat_one_value[strat_num]) &
                                         ((df_SALES['age'] >= age_range[0]) &
                                          (df_SALES['age'] <= age_range[1]))].reset_index().drop('index', 1)
            populate_files(OPEN_TAB, df_SALES_filtered, edge_class, strat_class, top_CONST, VIS_parameters,
                           shape_name, VIS_location, str(strat_ft_name.replace('_', ' ').upper() + ': ' +
                                                         curr_strat_one_value[strat_num] + ", " + str(age_range) +
                                                         '; '),
                           str("EXT-STRAT" + "/" + strat_ft_name + "/" + str(strat_num) + " - " +
                               list(strat_one_values.keys())[strat_num] + "/" + str(age_range) + "/"))

_ = """
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
####################################################################################################################"""
