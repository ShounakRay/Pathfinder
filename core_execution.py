# @Author: Shounak Ray <Ray>
# @Date:   18-Jun-2021 14:06:16:161  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: core_execution.py
# @Last modified by:   Ray
# @Last modified time: 24-Jun-2021 19:06:27:273  GMT-0600
# @License: [Private IP]

import math
import re
from collections import Counter, defaultdict
from datetime import datetime

import holidays as hds
import networkx as nx
import numpy as np
import pandas as pd
from suntime import Sun

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


def pull_from_S3(access_key):
    pass


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

    for phn_col in ['home_phone', 'business_phone', 'cell_phone']:
        df[phn_col] = df[phn_col].apply(lambda x: str(x))

    # QUESTION What does a `marital status` of 0 mean?
    # QUESTION What does nan `marital_status` value mean?
    # QUESTION What does nan `billing cycle` mean? Forgot to record or defaults to something?
    # QUESTION What does nan `source` value mean?

    # NOTE: Only dropping these next columns since they're consistently empty
    final_drop += ['salutation_prefix', 'deactivation_date', 'card_template', 'slip_rate',  'address_name', 'company']
    _ = [df.drop(col, inplace=True, axis=1) for col in final_drop]

    return df   # Not required since all operations are inplace


def retrieve_selective_membership(df, ids, prefix=None, salutation_prefix=None):
    # df = member_df.copy()
    subset = df[df['member_number'].isin(ids)]

    return subset


_ = """
#####################################
############   STAGE 2   ############
#####################################
"""


def checks_formatting(df):
    # Fix input columns before any further pre-processing
    df.reset_index(inplace=True)
    df.columns = [c.replace(' ', '_') for c in df.columns if c != 'index'] + ['last_temp']
    df['Check_Server'] = df['Check_Server'] + ' ' + df['last_temp']
    df.drop('last_temp', axis=1, inplace=True)

    for col in ['Check_Creation_Date']:
        df[col] = pd.to_datetime(df[col])

    for col in ['Check_Open_Time', 'Check_Close_Time']:
        df[col] = df[col].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time() if x == x else np.nan)

    # Macro, Holidays
    CAD_Holidays = hds.CAN(years=list(set(df['Check_Creation_Date'].apply(lambda x: x.date().year))))
    df['Holiday_Name'] = df['Check_Creation_Date'].apply(lambda d: CAD_Holidays.get(d))
    df['Holiday'] = df['Holiday_Name'].apply(lambda name: True if type(name) == str else False)

    # Macro, Weekdays
    day_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['Weekday_Number'] = df['Check_Creation_Date'].apply(lambda d: d.weekday())
    df['Weekend'] = df['Weekday_Number'].apply(lambda x: True if x in [5, 6] else False)
    df['Weekday'] = ~df['Weekend']
    df['Day_Name'] = df['Weekday_Number'].apply(day_dict.get)
    df['Day_Number_in_Year'] = df['Check_Creation_Date'].apply(lambda d: d.timetuple().tm_yday)

    # Micro, Time of Day
    sun = Sun((latitude := 51.0447), (longitude := -114.0719))
    sr, ss = sun.get_local_sunrise_time(), sun.get_local_sunset_time()

    return df   # Not required since all operations are inplace


def retrieve_selective_activities(df, service_providers=None, item_groups=None, item_names=None):
    kwargs = locals().copy()
    kwargs.pop('df')

    # # NOTE: For SALES data, testing...
    # df = df_SALES.copy()

    # NOTE: For checks.csv data, testing...
    df = checks_df.copy().rename(columns={'Open_On_Terminal': 'service_provider'})

    # Dynamic filtering
    for var_name, filts in kwargs.items():
        if filts is not None:
            var_name = var_name[:-1]
            print(f'Filtering on {var_name}...')
            df = df[df[var_name].isin(filts)]

    # TODO: Ultimately, CHECK and SALES data should be merged into one dataframe

    return df


_ = """
#####################################
############   STAGE 3   ############
#####################################
"""


def retrieve_selective_transactions(df, start_date=None, end_date=None, holidays=None,
                                    weekdays=None, repeating_days=None):
    # # NOTE: For SALES data, testing...
    # df = df_SALES.copy()

    # NOTE: For checks.csv data, testing...
    df = checks_df.copy()

    # Filter by simple range
    if not bool(start_date):
        end_date = df['Check_Creation_Date'].min()
    if not bool(end_date):
        end_date = df['Check_Creation_Date'].max()
    bounds = [start_date, end_date]
    for d in bounds:
        bounds[bounds.index(d)] = datetime.strptime(d, '%Y-%m-%d')
    df = df[(bounds[0] <= df['Check_Creation_Date']) & (df['Check_Creation_Date'] <= bounds[1])]

    # Filter by holidays


_ = """
#####################################
############   STAGE 4   ############
#####################################
"""


def restructure_to_connections(df, dict_repl='default'):
    if dict_repl == 'default':
        dict_repl = {'Open_On_Terminal': 'to',
                     'Close_On_Terminal': 'from',
                     'Check_Open_Time': 'open_time',
                     'Check_Close_Time': 'close_time'}
    # TODO: Include membership information in meta data (adjust groupby operation accordingly)
    storage = df.rename(columns=dict_repl).groupby(['Member_Number',
                                                    'Check_Creation_Date']
                                                   )[['to',
                                                      'from',
                                                      'open_time',
                                                      'close_time']].apply(dict).apply(lambda d: {k: v.to_list()
                                                                                                  for k, v in
                                                                                                  d.items()})

    return storage


_ = """
#######################################################################################################################
#########################################   MODULE 1 – MEMBER DEMOGRAPHICS   ##########################################
#######################################################################################################################
"""
member_df = engineer_member_data(pd.read_csv('Data/MEMBERSHIP.csv')).infer_objects()

_ = """
#######################################################################################################################
###########################################   MODULE 2 – ACTIVITY CATALOG   ###########################################
#######################################################################################################################
"""
# TEMP ingestion -> Should ultimately be a different dataset
checks_df = checks_formatting(pd.read_csv('Data/checks.csv')).infer_objects()

list(checks_df)

sprov_catalog = set(list(checks_df['Open_On_Terminal'].unique()) + list(checks_df['Close_On_Terminal'].unique()))
serve_catalog = set(checks_df['Check_Server'])
linkage = defaultdict(list)
for terminal, server in checks_df.set_index('Open_On_Terminal')['Check_Server'].items():
    linkage[terminal].append(server)
    linkage[terminal] = list(set(linkage[terminal]))

_ = """
#######################################################################################################################
#####################################   MODULE 3 – TIME/ACTIVTIY CONSIDERATION   ######################################
#######################################################################################################################
"""

# HIGHER RESOLUTION: SERVICE PROVIDER, ITEM GROUP, ITEM NAME
df_SALES = pd.read_csv("Data/SALES.csv", low_memory=False)
# LOWER RESOLUTION: SERVICE PROVIDER (PROXY)

_ = """
#######################################################################################################################
#########################################   APPLICATION 1 – APPLY FILTERING   #########################################
#######################################################################################################################
"""
# TODO: Given start and end times per day, determine location path per member number
# MEMBER FILTERS
member_ids = ['1002', '1002-1', '1002-2']

# ACTIVITY FILTERS
# NOTE: Current absence of item_group and item_name in the SALES data limits resolution
# NOTE: However, client flow can still be mapped using open and close terminals
#       (although you don't know what they bought in this time)
service_providers = ['DC Cafe29 Self Serve', 'DC Cafe29 Stir Fry', 'DC Cafe29 Salad Bar', 'DC Point After S']
item_groups = ['DC Fitness - Merchandise', 'DC Fitness - Personal Training', 'DC Badminton - Private Lessons']
item_names = ['Kickboxing Tank Top', 'Kickboxing T-shirt', 'Kickboxing Shin Pads', 'Kickboxing Shorts Premium']

# TIME FILTERS
start_date = '2021-01-01'
end_date = '2021-05-29'
holidays = True
weekdays = True
repeating_days = ['Monday', 'Tuesday', 'Friday']
time_of_day = ['Morning', 'Afternoon', 'Evening']

# NOTE: Get specific member information following inputted requirements
selective_members = retrieve_selective_membership(member_df, ids=member_ids)
# NOTE: Get filtered occurrence data following inputted specific event/activity/transaction requirements
selective_occurences = retrieve_selective_activities(checks_df,
                                                     service_providers=service_providers,
                                                     item_groups=None,
                                                     item_names=None)
# NOTE: Get specific check data following inputted time requirements
final_occurrences = retrieve_selective_transactions(selective_occurences,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    holidays=holidays,
                                                    weekdays=weekdays,
                                                    repeating_days=repeating_days)
# NOTE: Merge transactions and member data
final_subset = pd.merge(selective_members, final_occurrences, on='member_number').infer_objects()

_ = """
#######################################################################################################################
###########################################   APPLICATION 2 – MAKE GRAPH   ############################################
#######################################################################################################################
"""

# Re-structure so it's easy to make graphs
connections = restructure_to_connections(final_subset)

# Make networkx Graph
internal_graph = nx.DiGraph()
c = 0
for key, data in connections.items():
    c += 1
    # TODO: Add edges (nodes implicity added)


# TODO: Convert networkx graph to VIS.JS Graph

# GLENCOE IDEAS:
# IDEA: Detecting high traffic times, spacing them out
#       > Staffing proxy
# IDEA: Bundling pricing, classes/reservations VERSUS. Balance with traffic due to popularity
#       > How do you analyze popularity trends for specific groups/providers and balance traffic?
# IDEA: More insightful member lookup with buying patterns
# TEMP: Bring up Forms...

# TRAFFIC <-> BUNDLING <-> MARKETING
# Level of specificity: Service Provider, Age Groups, Time/Day, Membership

# NOTE: November 2019 -> June 2020
# NOTE: All of 2019 (Seasonanility, Times)

# EOF

# EOF
