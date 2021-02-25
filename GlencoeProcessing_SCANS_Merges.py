# @Author: Shounak Ray <Ray>
# @Date:   23-Feb-2021 15:02:90:903  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: scan_analysis.py
# @Last modified by:   Ray
# @Last modified time: 24-Feb-2021 22:02:27:277  GMT-0700
# @License: [Private IP]

import re
from collections import Counter
from itertools import chain

import pandas as pd

# '1741GC-1'.split('-')


def dep_type(row):
    if(row['Dependent']):
        seg = int(row['lettered_ID'].split('-')[1])
        print(seg)
        if(seg == 1):
            return 'Spousal'
        elif(seg > 1):
            return 'Non-Spousal'
        else:
            return 'Unclassified'
    else:
        return None


group_relations = {'GC': 'Golf Club',
                   'AFF': 'Affiliate',
                   'INT': 'Internal (INT)',
                   'GL': 'Internal (GL)',
                   'DCGuest': 'Downtown Guest',
                   'LG': 'League',
                   'CORP': 'Corporate'}

df_MANSCANS = pd.read_csv("Data/MANUAL_SCANS.csv", low_memory=False)
df_SCANS = pd.read_csv("Data/SCANS.csv", low_memory=False)
df_EVENTS = pd.read_csv("Data/EVENTS.csv", low_memory=False)
df_MEMEBRSHIP = pd.read_csv("Data/MEMBERSHIP.csv", low_memory=False)
df_RESERVATIONS = pd.read_csv("Data/RESERVATIONS.csv", low_memory=False)
df_SALES = pd.read_csv("Data/SALES.csv", low_memory=False)
dfs = [df_MANSCANS, df_SCANS, df_EVENTS, df_MEMEBRSHIP, df_RESERVATIONS, df_SALES]

store = []
for d in dfs:
    store.append(set(d['member_number']))
unique_mids = list(set(list(chain.from_iterable(store))))
lettered_mids = [mid for mid in unique_mids if(type(mid) == str and any(let.isalpha() for let in mid))]
lmids = pd.DataFrame(lettered_mids, columns=['lettered_ID'])
lmids['Mem_Category_Abbrev'] = lmids['lettered_ID'].apply(lambda x: re.sub(r'[0-9]', '', x).replace('-', '').strip())
lmids['Mem_Category_Type'] = [chg for chg in lmids['Mem_Category_Abbrev'].replace(group_relations)
                              if(chg in group_relations.keys()) else None]
lmids['Dependent'] = lmids['lettered_ID'].apply(lambda x: True if('-' in x) else False)
lmids['Dependent_Type'] = lmids.apply(dep_type, axis=1)

Counter(lmids['Mem_Category_Type'])
Counter(lmids['Mem_Category_Abbrev'])

mid = unique_mids[0]


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
