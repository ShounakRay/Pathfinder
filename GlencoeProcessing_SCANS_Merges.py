# @Author: Shounak Ray <Ray>
# @Date:   23-Feb-2021 15:02:90:903  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: scan_analysis.py
# @Last modified by:   Ray
# @Last modified time: 23-Feb-2021 16:02:84:840  GMT-0700
# @License: [Private IP]


import pandas as pd

# >>> DF_SCANS PROCESSING
df_MANSCANS = pd.read_csv("Data/MANUAL_SCANS.csv", low_memory=False)
df_MANSCANS['member_name'] = df_MANSCANS['member_name'].str.upper()
df_SCANS = pd.read_csv("Data/SCANS.csv", low_memory=False)
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

df_SCANS['card_holder'].isin(['LITTLE, DOROTHY']).unique()

len(set(list(df_SCANS['card_holder']) + list(df_MANSCANS['member_name'])).intersection(df_SALES['member_name']))
len(set(list(df_SCANS['card_holder']) + list(df_MANSCANS['member_name'])))
len(set(df_SALES['member_name']))

len(set(list(df_SCANS['member_number']) + list(df_MANSCANS['member_number'])).intersection(df_SALES['member_number']))
len(set(list(df_SCANS['member_number']) + list(df_MANSCANS['member_number'])))
len(set(df_SALES['member_number']))
