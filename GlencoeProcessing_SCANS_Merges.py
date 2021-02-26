# @Author: Shounak Ray <Ray>
# @Date:   23-Feb-2021 15:02:90:903  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: scan_analysis.py
# @Last modified by:   Ray
# @Last modified time: 25-Feb-2021 16:02:36:365  GMT-0700
# @License: [Private IP]

import functools
import math
import re
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd

group_relations = {'GC': 'Golf Club',
                   'AFF': 'Affiliate',
                   'INT': 'Internal (INT)',
                   'GL': 'Internal (GL)',
                   'DCGuest': 'Downtown Guest',
                   'LG': 'League',
                   'CORP': 'Corporate'}


def util_search_df(df, *conditions):
    def conjunction(*conditions):
        return functools.reduce(np.logical_and, conditions)
    return df[conjunction(*conditions)]


def lambda_dep_type(row):
    if(row['Dependent']):
        seg = int(row['lettered_ID'].split('-')[1])
        if(seg == 1):
            return 'Spousal'
        elif(seg > 1):
            return 'Non-Spousal'
        else:
            return 'Unclassified'
    else:
        return None


def lambda_mem_type(row, GROUP_RELATIONS=group_relations):
    print([row['Mem_Category_Abbrev'], type(row['Mem_Category_Abbrev'])])
    if(row['Mem_Category_Abbrev']):
        if(type(row['Mem_Category_Abbrev']) == float):
            if(not math.isnan(row['Mem_Category_Abbrev'])):
                return GROUP_RELATIONS.get(row['Mem_Category_Abbrev']) or 'Non-Classified'
        else:
            return GROUP_RELATIONS.get(row['Mem_Category_Abbrev']) or 'Non-Classified'
    else:
        return None

# Counter(df_RESERVATIONS['start_time'].isnull())
# Counter(df_SALES['start_time'].isnull())
# df_MEMEBRSHIP['activation_date']

# TODO: Data Requirement
# Do ID's show rough order in the membership data? But we still need actual
# SALES exact time
# Times for the reservations (NAN values for reservation times)
# <><> How many times do you sign up for an event
# Duration of program times and activities, reservations


df_MANSCANS = pd.read_csv("Data/MANUAL_SCANS.csv", low_memory=False)
df_SCANS = pd.read_csv("Data/SCANS.csv", low_memory=False)
df_EVENTS = pd.read_csv("Data/EVENTS.csv", low_memory=False)
df_MEMEBRSHIP = pd.read_csv("Data/MEMBERSHIP.csv", low_memory=False)
df_RESERVATIONS = pd.read_csv("Data/RESERVATIONS.csv", low_memory=False)
df_SALES = pd.read_csv("Data/SALES.csv", low_memory=False)
dfs = [df_MANSCANS, df_SCANS, df_EVENTS, df_MEMEBRSHIP, df_RESERVATIONS, df_SALES]

# TODO: How do you tag 123GC (Corporate or Golf Club)
# TODO: Test Feature engineering across all cases

# Aggregate and accumulate all member numbers across all the datasets
store = []
for d in dfs:
    store.append(set(d['member_number']))
# Reformat data to DataFrame and optionally filter to only include those that include letters (ignoring nans)
mids = [x for x in list(set(list(chain.from_iterable(store)))) if x == x]
mids = [mid for mid in mids if(type(mid) == str and any(let.isalpha() for let in mid))]
df_mids = pd.DataFrame(mids, columns=['lettered_ID'])

# Determine Regex Pattern to process Family ID
pat_partial = '('
for k in group_relations.keys():
    pat_partial += k + '|'
pat = pat_partial[:-1] + ')' + r'|-[0-9]{1,2}'
# Apply Regex
df_mids['Family_ID'] = df_mids['lettered_ID'].apply(lambda x: re.sub(pat, '', x))
df_mids['Mem_Category_Abbrev'] = df_mids['lettered_ID'].apply(lambda x:
                                                              re.sub(r'[0-9]', '', x).replace('-', '').strip())
df_mids.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_mids['Mem_Category_Type'] = df_mids.apply(lambda_mem_type, axis=1)
df_mids.drop('Mem_Category_Abbrev', axis=1, inplace=True)
df_mids['Dependent'] = df_mids['lettered_ID'].apply(lambda x: True if('-' in x) else False)
df_mids['Dependent_Type'] = df_mids.apply(lambda_dep_type, axis=1)
df_mids = df_mids.sort_values('Family_ID').reset_index(drop=True)

# [x for x in df_mids['lettered_ID'] if 'INT' in x]
# df_mids.to_html('df_mids.html')

# DATA PROCESSING
df_MANSCANS['member_name'] = df_MANSCANS['member_name'].str.upper()
df_SCANS = df_SCANS[['card_holder', 'time', 'date', 'location', 'member_number']]
df_SCANS['card_holder'] = df_SCANS['card_holder'].str.upper()
df_SCANS['location'] = df_SCANS['location'].str.upper()
# STRIP ALL STRINGS
str_list = ['card_holder', 'location']
for x in str_list:
    df_SCANS[x] = [str(i).replace('  ', ' ') for i in list(df_SCANS[x])]

# >>> DF_SALES W/ ORDER PROCESSING
# Compare overlap across scans and sales datasets
for mem_name, mem_num in list(zip(df_SALES['member_name'], df_SALES['member_number'])):
    matches = df_SCANS[(df_SCANS['card_holder'] == mem_name) | (df_SCANS['member_number'] == mem_num)]

# df_SCANS['card_holder'].isin(['LITTLE, DOROTHY']).unique()

# len(set(list(df_SCANS['card_holder']) + list(df_MANSCANS['member_name'])).intersection(df_SALES['member_name']))
# len(set(list(df_SCANS['card_holder']) + list(df_MANSCANS['member_name'])))
# len(set(df_SALES['member_name']))
#
# len(set(list(df_SCANS['member_number']) + list(df_MANSCANS['member_number'])).intersection(df_SALES['member_number']))
# len(set(list(df_SCANS['member_number']) + list(df_MANSCANS['member_number'])))
# len(set(df_SALES['member_number']))
