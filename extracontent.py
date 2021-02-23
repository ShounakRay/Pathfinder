# @Author: Shounak Ray <Ray>
# @Date:   23-Feb-2021 15:02:41:415  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: extracontent.py
# @Last modified by:   Ray
# @Last modified time: 23-Feb-2021 15:02:80:805  GMT-0700
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
