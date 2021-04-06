# @Author: Shounak Ray <Ray>
# @Date:   01-Apr-2021 21:04:31:316  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: xml_to_csv.py
# @Last modified by:   Ray
# @Last modified time: 06-Apr-2021 13:04:09:091  GMT-0600
# @License: [Private IP]

# CONVERSION RUN-TIME: 3.5 SECONDS (EXCLUDING OPTIONAL DATETIME TYPE CHECKS)


import re
from collections import Counter

import adjustText
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
    '"s27"' in line and headers.append(re.findall(r'>(.*?)<', line)[1].replace(' ', '_'))
    ('<ss:Cell>' in line or '"s21"' in line) and rows.append(re.findall(r'>(.*?)<', line)[1])
rows = [rows[len(headers) * start:len(headers) * (start + 1)] for start in range(int(len(rows) / len(headers)))]

# Transform Parsed Data
df = pd.DataFrame(rows, columns=headers).apply(pd.to_numeric, errors='ignore').infer_objects()

# This part is optional, I personally like converting dates from string to datetime format
df['Check_Creation_Date'] = pd.to_datetime(df['Check_Creation_Date'])
for c in ['Check_Open_Time', 'Check_Close_Time']:
    df[c] = [t.time() for t in pd.to_datetime(df[c])]

df = df.replace('', None)

# Format Object-Typed Features
rep_check = {}
for col in [c for c in df.select_dtypes(object).columns
            if not any([True if d in c else False for d in ['Time', 'Date']])]:
    df[col] = [re.sub(r'\s{2,}', ' ', e.replace('&amp;', '&').replace(',', ', ')).strip() for e in df[col]]
    rep_check[col] = dict(Counter(df[col]).most_common())

df.to_csv('Data/Check_HighRes.csv')

# Feature Engineering
df['Check_Open_Duration'] = (df['Check_Close_Time'].apply(t_s) - df['Check_Open_Time'].apply(t_s)) / 60.0

# _temp = df[(df['Check_Open_Duration'] < 0) & (df['Check_Close_Time'].apply(t_s) != 0)]

# Visualizations (Check Times)
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(15, 15))
    _temp = df[df['Check_Open_Duration'] >= 0]['Check_Open_Duration']
    _ = ax.scatter(_temp.index, _temp,
                   s=3, c='green', label='Close_Time after Open_Time (Makes Sense)')
    _temp = df[df['Check_Open_Duration'] < 0]['Check_Open_Duration']
    _ = ax.scatter(_temp.index, _temp,
                   s=10, c='orange', label='Close_Time before Open_Time (No Sense)')
    _temp = df[(df['Check_Open_Duration'] < 0) & (df['Check_Close_Time'].apply(t_s) != 0)]
    _ = ax.scatter(_temp['Check_Open_Duration'].index, _temp['Check_Open_Duration'],
                   s=50, c='red', label='Close_Times Don\'t make sense')
    _ = ax.legend()
    _ = ax.set_xlabel('Table Row Number (proportional to Time)')
    _ = ax.set_ylabel('Check_Open_Duration')
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.minorticks_on()
    ax.tick_params(labeltop=True, labelright=True)
    texts = []
    for i in _temp.index:
        y_pos = _temp['Check_Open_Duration'][i]
        texts.append(ax.text(i, y_pos, 'Row ' + str(i) + ' @ ' + str(int(round(y_pos))) + ' mins', size=7))
        # ax.annotate('Row ' + str(i), (i + 2000, _temp['Check_Open_Duration'][i] + 20), fontsize=6)
    _ = adjustText.adjust_text(texts)
    fig.tight_layout()

# Visualizations (Other)
if __name__ == '__main__':
    from collections import Counter

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Check Activity Over Time
    _time, _freq = (zip(*sorted(dict(Counter(df['Check_Creation_Date'])).items())))
    time_freq = pd.Series(_freq, _time)
    fig, ax = plt.subplots(figsize=(15, 20), nrows=3)
    ax[0].set_title('Check Activity on Day of Week\n(Sunday > Saturday)')
    sns.boxplot((time_freq.index.dayofweek + 1) % 7 + 1, time_freq, ax=ax[0])
    ax[1].set_title('Check Activity on Week of Year\nNOTE: Week 48 – 52 is from 2019, Week 1 – 5 is from 20')
    sns.boxplot(time_freq.index.weekofyear, time_freq, ax=ax[1], order=time_freq.index.weekofyear.unique())
    ax[2].set_title('Check Activity in Given Data')
    ax[2].bar(_time, _freq)
