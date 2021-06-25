# @Author: Shounak Ray <Ray>
# @Date:   23-Feb-2021 15:02:90:903  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: scan_analysis.py
# @Last modified by:   Ray
# @Last modified time: 24-Jun-2021 19:06:88:883  GMT-0600
# @License: [Private IP]

import functools
import math
import re
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from collections import Counter
# from itertools import chain
pd.set_option('display.max_columns', None)

# Full form of abbreviations found in member_number (constant).
group_relations = {'GC': 'Golf Club',
                   'AFF': 'Affiliate',
                   'INT': 'Internal (INT)',
                   'GL': 'Internal (GL)',
                   'DCGuest': 'Downtown Guest',
                   'LG': 'League',
                   'CORP': 'Corporate'}

# Preferred order of MultiIndexes (global constant)
global_order = ['Identification', 'Time', 'Demographics', 'Core', 'Transactions', 'Preferences', 'Miscellaneous']

# MULTI-INDEX the columns for better referencing and understanding
raw_tuple_assignment = [('prefix', 'Demographics'),
                        ('statement_delivery_method', 'Preferences'),
                        ('card_holder_name', 'Demographics'),
                        ('charge_item', 'Transactions'),
                        ('secondary_email', 'Demographics'),
                        ('item_total', 'Transactions'),
                        ('card_holder', 'Demographics'),
                        ('badge', 'Identification'),
                        ('age_bin', 'Demographics'),
                        ('last_name', 'Demographics'),
                        ('card_template', 'Miscellaneous'),
                        ('home_phone', 'Demographics'),
                        ('event', 'Core'),
                        ('quantity', 'Transactions'),
                        ('event_name', 'Core'),
                        ('source', 'Core'),
                        ('entries_in_a_month', 'Miscellaneous'),
                        ('salutation_prefix', 'Demographics'),
                        ('business_phone', 'Demographics'),
                        ('start_time', 'Time'),
                        ('seqnumber', 'Miscellaneous'),
                        ('address_line_2', 'Demographics'),
                        ('address_line_3', 'Demographics'),
                        ('service_provider', 'Core'),
                        ('gender', 'Demographics'),
                        ('email', 'Demographics'),
                        ('member_number', 'Identification'),
                        ('member_since', 'Demographics'),
                        ('date_of_birth', 'Demographics'),
                        ('price', 'Transactions'),
                        ('membership_tenure', 'Demographics'),
                        ('status', 'Demographics'),
                        ('marital_status', 'Demographics'),
                        ('middle_name', 'Demographics'),
                        ('sub_total', 'Transactions'),
                        ('nick_name', 'Demographics'),
                        ('party_size', 'Miscellaneous'),
                        ('table', 'Miscellaneous'),
                        ('first_name', 'Demographics'),
                        ('zip', 'Demographics'),
                        ('entries_in_a_day', 'Miscellaneous'),
                        ('reservation_type', 'Core'),
                        ('member_type', 'Demographics'),
                        ('check_number', 'Core'),
                        ('billing_email', 'Demographics'),
                        ('meal_period', 'Core'),
                        ('event_end', 'Time'),
                        ('fsa', 'Miscellaneous'),
                        ('date', 'Time'),
                        ('address_line_1', 'Demographics'),
                        ('trainer_end', 'Time'),
                        ('member_status', 'Demographics'),
                        ('item_group', 'Core'),
                        ('email_address', 'Demographics'),
                        ('company', 'Demographics'),
                        ('deactivation_date', 'Demographics'),
                        ('trainer', 'Core'),
                        ('employer', 'Core'),
                        ('phone_number', 'Demographics'),
                        ('billing_cycle', 'Demographics'),
                        ('slip_rate', 'Miscellaneous'),
                        ('trainer_start', 'Time'),
                        ('item_name', 'Core'),
                        ('event_number', 'Core'),
                        ('notes', 'Miscellaneous'),
                        ('reservation', 'Core'),
                        ('number_of_guests', 'Core'),
                        ('location', 'Core'),
                        ('age', 'Demographics'),
                        ('created_on', 'Time'),
                        ('member_name', 'Demographics'),
                        ('created_via', 'Miscellaneous'),
                        ('city', 'Demographics'),
                        ('players', 'Miscellaneous'),
                        ('country', 'Demographics'),
                        ('id', 'Identification'),
                        ('end_time', 'Time'),
                        ('activity', 'Core'),
                        ('booked_by', 'Core'),
                        ('cell_phone', 'Demographics'),
                        ('primary_email', 'Demographics'),
                        ('address_name', 'Demographics'),
                        ('total', 'Transactions'),
                        ('created_date', 'Time'),
                        ('suffix', 'Demographics'),
                        ('comments', 'Miscellaneous'),
                        ('time', 'Time'),
                        ('activation_date', 'Demographics'),
                        ('class_code', 'Core'),
                        ('state', 'Demographics')]
eng_tuple_assignment = [('Family_ID', 'Identification'),
                        ('Mem_Category_Type', 'Identification'),
                        ('Dependent', 'Identification'),
                        ('Dependent_Type', 'Identification')]
tuple_assignment = [t[::-1] for t in raw_tuple_assignment + eng_tuple_assignment]


def util_search_df(df, *conditions):
    """Utility Function: Returns DataFrame that meets certain conditions

    Parameters
    ----------
    df : DataFrame
        The DataFrame to search and filter.
    *conditions : args iterable
        Conditions to iterate through

    Returns
    -------
    DataFrame :
        Returns filtered DataFrame.

    """
    # Local function to reduce iterable of conditions into one condition (higher order)
    def conjunction(*conditions):
        return functools.reduce(np.logical_and, conditions)
    return df[conjunction(*conditions)]


def util_drop_level(df, level_label=0):
    """Utility Function: Remove MultiIndex format if preferred.

    Parameters
    ----------
    df : DataFrame
        The data with MultiIndexes.
    level_label : int
        The name/label of the specific MultiIndex level. Default to int(0).

    Returns
    -------
    DataFrame
        The DataFrame without MultiIndexes, rather only the regular column names.

    """
    # Drops the first level of the DataFrame, assuming it is 0
    df.columns = df.columns.droplevel(0)
    return df


def util_set_MI(df, tuple_assignment=tuple_assignment):
    """Utility Function: Sets a MultiIndex to the inputted DataFrame based on global key.

    Parameters
    ----------
    df : DataFrame
        The inputted DataFrame (without multiIndex).
    tuple_assignment : list (of tuples)
        Structure that associates column names to categories.

    Returns
    -------
    DataFrame
        The same DataFrame but with a new, MultiIndex level.

    """
    existing_cols = df.columns
    keys = [tup for tup in tuple_assignment if tup[1] in existing_cols]
    df.columns = pd.MultiIndex.from_tuples(keys)
    return df


def util_reorder_MI(df, order=global_order):
    """Utility Function: reorders MultiIndex according to constant order.

    Parameters
    ----------
    df : DataFrame
        DataFrame with unordered MultiIndexes.
    order : list
        Preferred order of MultiIndexes (global constant)

    Returns
    -------
    DataFrame
        The original DataFrame but with reordered MultiIndexes.

    """
    # Filter global order to include relevant columns found in inputted df.
    local_order = [lvl for lvl in order if lvl in lvl in set([tup[0] for tup in df.columns])]
    # Perform reindexing of columns by transposing inputted df twice.
    # TODO: Optimize reindexing process, very slow and inefficient
    df = df.T.reindex(local_order, level=0).T
    return df


def lambda_dep_type(row, mnum_col='member_number'):
    """Assigns the dependent status based on DataFrame values.

    Parameters
    ----------
    row : pd.Series
        A row of the DataFrame.

    Returns
    -------
    string
        Whether or not the dependent is Spousal, or non-classified if not set.

    """
    # This condition is True if row['Dependent'] evaluates to a something other than None (expected string).
    #  Otherwise it's False.
    if(row['Dependent']):
        # Get the "dependent number" at the tail of the member_number
        seg = int(row[mnum_col].split('-')[1])
        if(seg == 1):
            return 'Spousal'
        elif(seg > 1):
            return 'Non-Spousal'
        else:
            return 'Non-Classified'
    else:
        return None


def lambda_mem_type(row, GROUP_RELATIONS=group_relations):
    """Assigns the member type based on constant key.

    Parameters
    ----------
    row : pd.Series
        A row of the DataFrame.
    GROUP_RELATIONS : dict
        Full form of abbreviations found in member_number (global constant).

    Returns
    -------
    string
        The full-form of the abbreviated, or non-classified if not set.

    """
    # This condition is True if row['Mem_Category_Abbrev'] evaluates to a something other than None.
    #  Otherwise it's False.
    if(row['Mem_Category_Abbrev']):
        if(type(row['Mem_Category_Abbrev']) == float):
            if(not math.isnan(row['Mem_Category_Abbrev'])):
                return GROUP_RELATIONS.get(row['Mem_Category_Abbrev']) or 'Non-Classified'
        else:
            return GROUP_RELATIONS.get(row['Mem_Category_Abbrev']) or 'Non-Classified'
    else:
        return 'Standard'


def engineer_member_id(df, mnum_col='member_number', GROUP_RELATIONS=group_relations):
    """Feature Engineers new member groupings based on provided member number column.

    Parameters
    ----------
    df : DataFrame
        Original DataFrame (not MultiIndexed) to be engineered.
    mnum_col : string
        The column name containing member numbers.
    GROUP_RELATIONS : dict
        Full form of abbreviations found in member_number (global constant).

    Returns
    -------
    DataFrame
        The engineering DataFrame based on `mnum_col`.

    """
    # Determine Regex Pattern to process Family ID
    pat_partial = '('
    for k in GROUP_RELATIONS.keys():
        pat_partial += k + '|'
    pat = pat_partial[:-1] + ')' + r'|-[0-9]{1,2}'
    # Apply Regex
    df['Family_ID'] = df[mnum_col].apply(lambda x: re.sub(pat, '', x) if x == x else None)
    df['Mem_Category_Abbrev'] = df[mnum_col].apply(lambda x: re.sub(r'[0-9]', '', x).replace('-', '').strip()
                                                   if x == x else None)
    df['Mem_Category_Type'] = df.apply(lambda_mem_type, axis=1)
    df.drop('Mem_Category_Abbrev', axis=1, inplace=True)
    df['Dependent'] = df[mnum_col].apply(lambda x: (True if('-' in x) else False) if x == x else None)
    df['Dependent_Type'] = df.apply(lambda_dep_type, axis=1)

    return df


# TODO: Data Requirement

# TODO: Integrate Bryce's Source Table Code w/ S3 Pull

# Import all the base tables/sources of data, and then store in a dict for reference purposes
# df_MANSCANS = pd.read_csv("Data/MANUAL_SCANS.csv", low_memory=False)
# df_SCANS = pd.read_csv("Data/SCANS.csv", low_memory=False)
df_EVENTS = pd.read_csv("Data/EVENTS.csv", low_memory=False)
df_MEMEBRSHIP = pd.read_csv("Data/MEMBERSHIP.csv", low_memory=False)
df_RESERVATIONS = pd.read_csv("Data/RESERVATIONS.csv", low_memory=False)
df_SALES = pd.read_csv("Data/SALES.csv", low_memory=False)
dfs = {  # 'MANSCANS': df_MANSCANS,
         # 'SCANS': df_SCANS,
    'EVENTS': df_EVENTS,
    'MEMBERSHIP': df_MEMEBRSHIP,
    'RESERVATIONS': df_RESERVATIONS,
    'SALES': df_SALES}

# Feature Engineer the Member IDs
for df_name, df in dfs.items():
    # dfs[df_name] = engineer_member_id(df).infer_objects()
    if df_name == 'MEMBERSHIP':
        continue
    print(f'DATASET: {df_name}')
    date_series = pd.to_datetime(df['date'].copy().dropna())
    plt.figure(figsize=(12, 8))
    date_series.hist(bins=100)
    print(f"Earliest Date: {min(date_series)}")
    print(f"Latest Date: {max(date_series)}")
    print('\n\n')

df = dfs['EVENTS'].copy()
list(df)
df['event_name'].unique()

plt.figure(figsize=(30, 8))
df['seqnumber'].hist(bins=10)
df['reservation'].hist(bins=50)
df[['date', 'created_on', 'event_end', 'start_time', 'end_time', 'reservation']]
Counter(df['date'] == df['reservation'])
df[df['date'] == '2022-09-02']
df['location']

df_SALES['service_provider'].unique()
df_SALES[df_SALES['service_provider'] == 'DC Sports - Aquatics'].head(50)

df_SALES[df_SALES['price'] == 0]

df_EVENTS.drop(['end_time', 'seqnumber'])

# TODO: EVENTS, pick start time and end time
# TODO: EVENTS, why are there 5 date columns
# QUESTION: `reservation` is `event_end` or `created_on`

# Set the MultiIndex on all the DataFrames and format aesthetically
dfs_MI = {}
for df_name in dfs.keys():
    dfs_MI[df_name] = util_reorder_MI(util_set_MI(dfs.get(df_name))).infer_objects()


#

#

#
