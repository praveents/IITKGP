import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


#missingno is a python library for nice visulaisation of missing number in the data
import missingno as msno


#load data frame
teamData = pd.read_csv("scrapped_data_large.csv")

print(teamData.nunique())

teamData = teamData.replace('-', np.NaN)

# teamData.fillna(value=0, axis=1, inplace=True)

# Nullity or missing values by columns
msno.matrix(df=teamData.iloc[:,2:18], figsize=(15, 6))

# get the number of missing data points per column
missing_values_count = teamData.isnull().sum()

# look at the # of missing points in the first ten columns
print(missing_values_count[0:17])

teamDataDropRowWithAllNA = teamData.copy()
teamDataDropRowWithAllNA.dropna(how='all',inplace=True)

msno.matrix(df=teamDataDropRowWithAllNA.iloc[:,2:18], figsize=(12, 6), color=(0.42, 0.1, 0.05))

teamDataDropRowWithThres = teamData.copy()
teamDataDropRowWithThres.dropna(thresh=10,inplace=True)

msno.matrix(df=teamDataDropRowWithThres.iloc[:,2:18], figsize=(12, 6), color=(0.42, 0.1, 0.05))

print(teamData.dtypes)

for col in teamData.columns:
    if str.isnumeric((teamData[col])):
        teamData[col] = pd.to_numeric(teamData[col], errors='coerce')

print(teamData.dtypes)

teamData['teams'] = teamData['teams'].str.strip()

teamData['teams'] = teamData['teams'].str.lower()

print(teamData.loc[teamData['teams'].str.contains('india')])


