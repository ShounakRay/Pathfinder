# @Author: Shounak Ray <Ray>
# @Date:   18-Jun-2021 14:06:16:161  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: core_execution.py
# @Last modified by:   Ray
# @Last modified time: 22-Jun-2021 13:06:08:082  GMT-0600
# @License: [Private IP]

import math
import re
from datetime import datetime

import numpy as np
import pandas as pd

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""

group_relations = {'GC': 'Golf Club',
                   'AFF': 'Affiliate',
                   'INT': 'Internal (INT)',
                   'GL': 'Internal (GL)',
                   'DCGuest': 'Downtown Guest',
                   'LG': 'League',
                   'CORP': 'Corporate'}


_ = """
#######################################################################################################################
####################################################   FUNCTIONS   ####################################################
#######################################################################################################################
"""

_ = """
#####################################
########   DATA INGESTION   #########
#####################################
"""


_ = """
#####################################
############   STAGE 1   ############
#####################################
"""


def engineer_member_data(df, mnum_col='member_number', GROUP_RELATIONS=group_relations):
    """Feature Engineers new member groupings based on provided member number column.
       Also condenses available columns to minimze entropy/confusion.

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

    final_drop = []

    # NOTE: Looking at Member IDs
    # Determine Regex Pattern to process Family ID
    pattern = '(' + '|'.join(list(GROUP_RELATIONS)) + ')' + r'|-[0-9]{1,2}'
    df['Family_ID'] = df[mnum_col].apply(lambda x: re.sub(pattern, '', x) if x == x else None)
    df['Mem_Category_Abbrev'] = df[mnum_col].apply(lambda x: re.sub(r'[0-9]', '', x).replace('-', '').strip()
                                                   if x == x else None)
    df['Mem_Category_Type'] = df.apply(lambda_mem_type, axis=1)
    final_drop += ['Mem_Category_Abbrev']
    df['Dependent'] = df[mnum_col].apply(lambda x: (True if('-' in x) else False) if x == x else None)
    df['Dependent_Type'] = df.apply(lambda_dep_type, axis=1)

    # NOTE: Looking at raw columns to condense
    df['Member_Name'] = (df['prefix'] + ' '
                         + df['salutation_prefix'] + ' '
                         + df['first_name'] + ' '
                         + df['middle_name'] + ' '
                         + df['last_name'] + ' '
                         + df['suffix']).str.strip()
    final_drop += ['prefix', 'first_name', 'middle_name', 'last_name', 'suffix']
    for time_col in ['member_since', 'date_of_birth', 'activation_date', 'deactivation_date']:
        df[time_col] = pd.to_datetime(df[time_col])
    df['membership_tenure'] = (datetime.now() - df['activation_date']).dt.days
    df['address'] = (df['address_line_1'].replace(pd.NA, ' ' + ' ') +
                     df['address_line_2'].replace(pd.NA, ' ' + ' ') +
                     df['address_line_3'].replace(pd.NA, ' ')).str.strip()
    final_drop += ['address_line_1', 'address_line_2', 'address_line_3']

    df['age'] = (datetime.now() - df['date_of_birth']).dt.days / 365.25
    age_classes = list(set(tuple([int(i) for i in x.split('-')])
                           for x in [i if (i is not np.nan and '+' not in i) else '61-999'
                                     for i in df['age_bin'].unique()]))
    df['age_bin'] = df['age'].apply(lambda age: [age_class for age_class in age_classes
                                                 if age_class[0] <= age <= age_class[1] + 1])
    df['age_bin'] = df['age_bin'].apply(lambda i: None if not (type(i) == list and len(i) == 1) else str(i[0]))
    df.loc[df[df['gender'] != df['gender']].index, 'gender'] = 'UNKNOWN'

    # NOTE: Only dropping these next columns since they're consistently empty
    final_drop += ['salutation_prefix', 'deactivation_date', 'card_template', 'slip_rate',  'address_name', 'company']
    _ = [df.drop(col, inplace=True, axis=1) for col in final_drop]

    return df   # Not required since all operations are inplace


def retrieve_selective_ids(df, prefix=None, salutation_prefix=None):
    None


_ = """
#####################################
############   STAGE 2   ############
#####################################
"""


def checks_formatting(df):
    df.reset_index(inplace=True)
    df.columns = [c.replace(' ', '_') for c in df.columns if c != 'index'] + ['last_temp']
    df['Check_Server'] = df['Check_Server'] + ' ' + df['last_temp']
    df.drop('last_temp', axis=1, inplace=True)

    return df   # Not required since all operations are inplace


_ = """
#####################################
############   STAGE 3   ############
#####################################
"""


_ = """
#######################################################################################################################
#########################################   MODULE 1 – MEMBER DEMOGRAPHICS   ##########################################
#######################################################################################################################
"""
member_df = engineer_member_data(pd.read_csv('Data/MEMBERSHIP.csv')).infer_objects()
list(member_df)
member_df['marital_status'].unique()
member_df[member_df['marital_status'] == '0']

# QUESTION What does a marital status of 0 mean?

_ = """
#######################################################################################################################
###########################################   MODULE 2 – ACTIVITY CATALOG   ###########################################
#######################################################################################################################
"""
# TEMP ingestion -> Should ultimately be a different dataset
checks_df = checks_formatting(pd.read_csv('Data/checks.csv')).infer_objects()


_ = """
#######################################################################################################################
#####################################   MODULE 3 – TIME/ACTIVTIY CONSIDERATION   ######################################
#######################################################################################################################
"""

_ = """
#######################################################################################################################
#########################################   APPLICATION 1 – APPLY FILTERING   #########################################
#######################################################################################################################
"""

_ = """
#######################################################################################################################
###########################################   APPLICATION 2 – MAKE GRAPH   ############################################
#######################################################################################################################
"""
