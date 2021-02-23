# @Author: Shounak Ray <Ray>
# @Co-author: Bryce Howes <Howes>
# @Date:   09-Dec-2020 15:12:00:000  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: GC_TimeLine_Algo.py
# @Last modified by:   Ray
# @Last modified time: 23-Feb-2021 16:02:61:615  GMT-0700
# @License: [Private IP]

'''
MM'"""""`MM M""MMMMMMMM MM""""""""`M M"""""""`YM MM'""""'YMM MMP"""""YMM MM""""""""`M
M' .mmm. `M M  MMMMMMMM MM  mmmmmmmM M  mmmm.  M M' .mmm. `M M' .mmm. `M MM  mmmmmmmM
M  MMMMMMMM M  MMMMMMMM M`      MMMM M  MMMMM  M M  MMMMMooM M  MMMMM  M M`      MMMM
M  MMM   `M M  MMMMMMMM MM  MMMMMMMM M  MMMMM  M M  MMMMMMMM M  MMMMM  M MM  MMMMMMMM
M. `MMM' .M M  MMMMMMMM MM  MMMMMMMM M  MMMMM  M M. `MMM' .M M. `MMM' .M MM  MMMMMMMM
MM.     .MM M         M MM        .M M  MMMMM  M MM.     .dM MMb     dMM MM        .M
MMMMMMMMMMM MMMMMMMMMMM MMMMMMMMMMMM MMMMMMMMMMM MMMMMMMMMMM MMMMMMMMMMM MMMMMMMMMMMM
'''

# pd.set_option('display.max_columns', None)

import pandas as pd

# import inflection
# import re
#
#
# def convert_to_snake(input):
#     out = inflection.underscore(input)
#     out = out.lstrip().replace(' ','_')
#     out = re.sub('\W+','', out)
#     return out

# member_demo.columns = [convert_to_snake(x) for x in member_demo.columns]


# MEMBERSHIP = pd.read_csv('Data/MEMBERSHIP.csv')
# SALES = pd.read_csv('Data/SALES.csv')
# EVENTS = pd.read_csv('Data/EVENTS.csv')
# RESERVATIONS = pd.read_csv('Data/RESERVATIONS.csv')
# SCANS = pd.read_csv('Data/SCANS.csv')
# MANUAL_SCANS = pd.read_csv('Data/MANUAL_SCANS.csv')


def gc_member_search(MEMBERSHIP, SALES, EVENTS, RESERVATIONS, SCANS, MANUAL_SCANS, id):
    id_list = []
    id_list.append(id)
    check_dep = id.split('-')
    if len(check_dep) > 1:
        id_list.append(check_dep[0] + 'GC' + '-' + check_dep[1])
    else:
        id_list.append(id + 'GC')
    df1 = MEMBERSHIP[MEMBERSHIP['member_number'].isin(id_list)].drop_duplicates()
    df2 = SALES[SALES['member_number'].isin(id_list)].drop_duplicates()
    df2['category'] = 'sale'
    df3 = EVENTS[EVENTS['member_number'].isin(id_list)].drop_duplicates()
    df3['category'] = 'event'
    df4 = RESERVATIONS[RESERVATIONS['member_number'].isin(id_list)].drop_duplicates()
    df4['category'] = 'reservation'
    df5 = SCANS[SCANS['member_number'].isin(id_list)].drop_duplicates()
    df5['category'] = 'scan'
    df6 = MANUAL_SCANS[MANUAL_SCANS['member_number'].isin(id_list)].drop_duplicates()
    df6['category'] = 'manual scan'
    df7 = df2[['member_number', 'date', 'category']].append([df3[['member_number', 'date', 'category']],
                                                             df4[['member_number', 'date', 'category']],
                                                             df5[['member_number', 'date', 'category']],
                                                             df6[['member_number', 'date', 'category']]])

    return df1, df2, df3, df4, df5, df6, df7


# df1,df2,df3,df4,df5,df6,df7 = gc_member_search(MEMBERSHIP,SALES,EVENTS,RESERVATIONS,SCANS, '1006')
