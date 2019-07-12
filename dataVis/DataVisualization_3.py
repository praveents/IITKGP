#!/usr/bin/env python
# coding: utf-8

# # Data Visualization Tutorial 3

# ## Load and Setup the Data

# In[1]:


# import required library functions
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# load data, skip the top 20 and bottom 2 rows as they do not contain relevant data
df_canada = pd.read_excel('canada.xlsx',
                          sheet_name = 'Canada by Citizenship',
                          skiprows = range(20),
                          skipfooter = 2)

# conversion index and columns to lists
df_canada.columns.tolist()
df_canada.index.tolist()

# remove unnecessary columns
# in pandas axis=0 re|presents rows (default) and axis=1 represents columns.
df_canada.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

# rename some columns to make better sense
df_canada.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

# convert all column names to strings
df_canada.columns = list(map(str, df_canada.columns))

# full range of the time series
years = list(map(str, range(1980, 2014)))

# add Total column
df_canada['Total'] = df_canada.sum(axis=1)

# index data by country
df_canada.set_index('Country', inplace=True)


# ## Visualize Correlation (2 variables)
# 
# ### Visualize total immigration into Canada from 1980 to 2013

# In[3]:


# we can use the sum() method to get the total population per year
df_total = pd.DataFrame(df_canada[years].sum(axis=0))

# change the years to type int (useful for regression later on)
df_total.index = map(int, df_total.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_total.reset_index(inplace = True)

# rename columns
df_total.columns = ['year', 'total']

# view the final dataframe
df_total.head()


# In[4]:


# plot the data points
df_total.plot(kind='scatter',
              x='year',
              y='total',
              figsize=(12, 6)
             )

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

plt.show()


# ### Compute and Visualize Regression Line

# In[5]:


# compute line fit y = ax + b
x = df_total['year']      # year on x-axis
y = df_total['total']     # total on y-axis
fit = np.polyfit(x, y, deg=1)

fit


# In[6]:


# plot the data points
df_total.plot(kind='scatter',
              x='year',
              y='total',
              figsize=(12, 6)
             )

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

# plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

plt.show()


# ## Visualize Regression with Seaborn

# In[7]:


# import library
import seaborn as sns


# In[ ]:


# we can use the sum() method to get the total population per year
df_total = pd.DataFrame(df_canada[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_total.index = map(float, df_total.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_total.reset_index(inplace=True)

# rename columns
df_total.columns = ['year', 'total']

# view the final dataframe
df_total.head()


# ### Plot regression line

# In[ ]:


plt.figure(figsize=(12, 6))
ax = sns.regplot(x='year', y='total', data=df_total, color='green', marker='+')


# ### Plot LOESS curve

# In[ ]:


plt.figure(figsize=(12, 6))
ax = sns.regplot(x='year', y='total', data=df_total, lowess=True, color='green', marker='+')


# ## Visualize Correlation (3 variables)
# 
# ### Arrange coutry wise yearly data

# In[ ]:


# transposed dataframe 
df_can_t = df_canada[years].transpose()

# cast the Years (the index) to type int
df_can_t.index = map(int, df_can_t.index)

# let's label the index. This will automatically be the column name when we reset the index
df_can_t.index.name = 'Year'

# reset index to bring the Year in as a column
df_can_t.reset_index(inplace=True)

# view the changes
df_can_t.head()


# ### Compute normalized immigration data

# In[ ]:


# normalize China data
norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())

print(norm_china)


# ### Visualize using Bubble Plot

# In[ ]:


ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='China',
                    figsize=(15, 8),
                    alpha=0.5,                  # transparency
                    color='green',
                    s=norm_china * 2000 + 10,  # pass in weights as size
                    xlim=(1978, 2015)
                   )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from China')
ax0.legend(['China'], loc='upper left', fontsize='x-large')


# In[ ]:




