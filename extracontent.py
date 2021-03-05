# @Author: Shounak Ray <Ray>
# @Date:   23-Feb-2021 15:02:41:415  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: extracontent.py
# @Last modified by:   Ray
# @Last modified time: 05-Mar-2021 11:03:71:718  GMT-0700
# @License: [Private IP]


# df_1 = pd.read_csv('/Users/Ray/Documents/Python/Glencoe/All Cardholders, by Last Name + GlencoeID.csv')
# df_2 = pd.read_csv('/Users/Ray/Documents/Python/Glencoe/Access Denial, Granted' +
#                    ' and Other Badge Events, by Reader + GlencoeID.csv')
# out_1, out_2 = [], []
#
# # Drop any duplicate rows from datasets
# df_1.drop_duplicates(inplace = True)
# df_2.drop_duplicates(inplace = True)

# # Delete any features that don't have multiple unique values
# for col in df_1.columns:
#     if(len(df_1[col].unique()) == 1):
#         out_1.append(col)
#         df_1.drop(col, 1, inplace = True)
# # Delete feature tagged as `0` in pandas, doesn't seem to be helpful
# df_1.drop('0', 1, inplace = True)
# # Delete any features that don't have multiple unique values
# for col in df_2.columns:
#     if(len(df_2[col].unique()) == 1):
#         out_2.append(col)
#         df_2.drop(col, 1, inplace = True)
#
# # Check which values in this specific column (anticipated `member number` don't begin with 0)
#
# # For Dataset 2, determine relative frequency of anticipated non-member IDs
# expected = 0
# tb_deleted = 0
# for member_number in list(df_1['0115101']):
#     if str(member_number)[0] == '0':
#         expected += 1
#     elif(str(member_number)[0] == 'E' or str(member_number)[0] == 'C'):
#         tb_deleted += 1
# # Number of Non-0 IDs = 1078 for the first dataset
# print('Dataset 1: Percentage of anticipated GlencoeIDs that refer to contractors or employees: ' +
#       str((tb_deleted/df_1.shape[0])*100)[:5])
#
# # For Dataset 2, determine relative frequency of anticipated non-member IDs
# expected = 0
# tb_deleted = 0
# for member_number in list(df_2['E21805']):
#     if str(member_number)[0] == '0':
#         anomaly += 1
#     elif(str(member_number)[0] == 'E' or str(member_number)[0] == 'C'):
#         tb_deleted += 1
# # Number of Non-0 IDs = 1078 for the first dataset
# print('Dataset 2: Percentage of anticipated GlencoeIDs that refer to contractors or employees: ' +
#       str((tb_deleted/df_2.shape[0])*100)[:5])
#
# # For `All Cardholders, by Last Name + GlencoeID.csv`
# df_1.columns = ['Name', '']
# len(df_1['0115101'].unique())
# len(df_1[' AMUNDRUD,  JOHN'].unique())

# <><><><><><><><>
# [x for x in df_mids['lettered_ID'] if 'INT' in x]

# <><><><><><><>
# >> Manually input categorizations, Flip list of tuples key, and store
# all_cols = set(list(chain.from_iterable([list(d.columns) for d in dfs])))
# raw_tuple_assignment = []
# for col in all_cols:
#     raw_tuple_assignment.append((col, input(str(all_cols.index(col)/len(all_cols)) +
#                                         ' Input Category for ' + col + ':')))

# ><><><><><><>
# >>> Testing Member Number Engineering using info from all source tables
# Aggregate and accumulate all member numbers across all the datasets
# store = []
# for d in dfs.values():
#     store.append(set(d['member_number']))
# Reformat data to DataFrame and optionally filter to only include those that include letters (ignoring nans)
# mids = [x for x in list(set(list(chain.from_iterable(store)))) if x == x]
# mids = [mid for mid in mids if(type(mid) == str and any(let.isalpha() for let in mid))]
# df_mids = pd.DataFrame(mids, columns=['lettered_ID'])
#
# # Determine Regex Pattern to process Family ID
# pat_partial = '('
# for k in group_relations.keys():
#     pat_partial += k + '|'
# pat = pat_partial[:-1] + ')' + r'|-[0-9]{1,2}'
# # Apply Regex
# df_mids['Family_ID'] = df_mids['lettered_ID'].apply(lambda x: re.sub(pat, '', x))
# df_mids['Mem_Category_Abbrev'] = df_mids['lettered_ID'].apply(lambda x:
#                                                               re.sub(r'[0-9]', '', x).replace('-', '').strip())
# df_mids['Mem_Category_Type'] = df_mids.apply(lambda_mem_type, axis=1)
# df_mids.drop('Mem_Category_Abbrev', axis=1, inplace=True)
# df_mids['Dependent'] = df_mids['lettered_ID'].apply(lambda x: True if('-' in x) else False)
# df_mids['Dependent_Type'] = df_mids.apply(lambda_dep_type, axis=1)
# df_mids = df_mids.sort_values('Family_ID').reset_index(drop=True)


# # Basic Data Re-formatting
# df_MANSCANS['member_name'] = df_MANSCANS['member_name'].str.upper()
# df_SCANS = df_SCANS[['card_holder', 'time', 'date', 'location', 'member_number']]
# df_SCANS['card_holder'] = df_SCANS['card_holder'].str.upper()
# df_SCANS['location'] = df_SCANS['location'].str.upper()
# # STRIP ALL STRINGS
# str_list = ['card_holder', 'location']
# for x in str_list:
#     df_SCANS[x] = [str(i).replace('  ', ' ') for i in list(df_SCANS[x])]
#
# # >>> DF_SALES W/ ORDER PROCESSING
# # Compare overlap across scans and sales datasets
# for mem_name, mem_num in list(zip(df_SALES['member_name'], df_SALES['member_number'])):
#     matches = df_SCANS[(df_SCANS['card_holder'] == mem_name) | (df_SCANS['member_number'] == mem_num)]

# df_SCANS['card_holder'].isin(['LITTLE, DOROTHY']).unique()

# len(set(list(df_SCANS['card_holder']) + list(df_MANSCANS['member_name'])).intersection(df_SALES['member_name']))
# len(set(list(df_SCANS['card_holder']) + list(df_MANSCANS['member_name'])))
# len(set(df_SALES['member_name']))
#
# len(set(list(df_SCANS['member_number']) + list(df_MANSCANS['member_number'])).intersection(df_SALES['member_number']))
# len(set(list(df_SCANS['member_number']) + list(df_MANSCANS['member_number'])))
# len(set(df_SALES['member_number']))

# %timeit functest
# %timeit functest2
#
#
# def functest2():
#     out = _github('https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'.format(owner=owner,
#                                                                                             repo=repo,
#                                                                                             branch=branch,
#                                                                                             path=path))
