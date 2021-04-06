# @Author: Shounak Ray <Ray>
# @Date:   01-Apr-2021 21:04:31:316  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: xml_to_csv.py
# @Last modified by:   Ray
# @Last modified time: 03-Apr-2021 22:04:06:069  GMT-0600
# @License: [Private IP]

# CONVERSION RUN-TIME: 3.5 SECONDS (EXCLUDING OPTIONAL DATETIME TYPE CHECKS)


import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


def t_s(t):
    return (t.hour * 60 + t.minute) * 60


# Import XML File and extract Relevant Lines
the_xml = [line.strip() for line in open('Data/Report-Data-Glencoe.xml', 'r').readlines()]
the_xml = the_xml[the_xml.index('<ss:Table>') + 1:the_xml.index('</ss:Table>') - 1]

# Retrive rows and headers (short-circuited, very optimized)
rows, headers = [], []
for line in the_xml:
    '"s27"' in line and headers.append(re.findall(r'>(.*?)<', line)[1])
    ('<ss:Cell>' in line or '"s21"' in line) and rows.append(re.findall(r'>(.*?)<', line)[1])
rows = [rows[len(headers) * start:len(headers) * (start + 1)] for start in range(int(len(rows) / len(headers)))]

# Transform Parsed Data
df = pd.DataFrame(rows, columns=headers).apply(pd.to_numeric, errors='ignore')

# This part is optional, I personally like converting dates from string to datetime format
df['Check Creation Date'] = pd.to_datetime(df['Check Creation Date'])
for c in ['Check Open Time', 'Check Close Time']:
    df[c] = [t.time() for t in pd.to_datetime(df[c])]

df.to_csv('Data/Check_HighRes.csv')

df['Check_Open_Duration'] = (df['Check Close Time'].apply(t_s) - df['Check Open Time'].apply(t_s)) / 60.0

# Counter(df['Check_Open_Duration'] >= 0)
# Counter(df[df['Check_Open_Duration'] < 0]['Check Close Time'])
# plt.plot(df[df['Check_Open_Duration'] < 0]['Check Close Time'].apply(t_s).reset_index(drop=True))
#
#
# df['Check_Open_Duration']
# plt.figure(figsize=(30, 15))
# plt.plot(df[df['Check_Open_Duration'] >= 0]['Check_Open_Duration'])
# plt.plot(df[df['Check_Open_Duration'] < 0]['Check_Open_Duration'])
# plt.plot(df[(df['Check_Open_Duration'] < 0) & (df['Check Open Time'].apply(t_s) < 0)]['Check_Open_Duration'])
# Counter(df['Check Open Time'].apply(t_s) != 0)
# dir(df['Check Close Time'][0])
#
# ((df['Check Close Time'][0].hour * 60) + df['Check Close Time'][0].minute) * 60

if __name__ == '__main__':
    from collections import Counter

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Check Activity Over Time
    _time, _freq = (zip(*sorted(dict(Counter(df['Check Creation Date'])).items())))
    time_freq = pd.Series(_freq, _time)
    fig, ax = plt.subplots(figsize=(15, 20), nrows=3)
    ax[0].set_title('Check Activity on Day of Week\n(Sunday > Saturday)')
    sns.boxplot((time_freq.index.dayofweek + 1) % 7 + 1, time_freq, ax=ax[0])
    ax[1].set_title('Check Activity on Week of Year\nNOTE: Week 48 – 52 is from 2019, Week 1 – 5 is from 20')
    sns.boxplot(time_freq.index.weekofyear, time_freq, ax=ax[1])
    ax[2].set_title('Check Activity in Given Data')
    ax[2].bar(_time, _freq)
