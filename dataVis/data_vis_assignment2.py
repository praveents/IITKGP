# import required library functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data, skip the top 20 and bottom 2 rows as they do not contain relevant data
immigrationCanada = pd.read_excel('canada.xlsx',
                                  sheet_name='Canada by Citizenship',
                                  skiprows=range(20),
                                  skipfooter=2)

# remove unnecessary columns
immigrationCanada.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)

# rename some columns to make better sense
immigrationCanada.rename(columns={'OdName': 'Country', 'AreaName': 'Continent', 'RegName': 'Region'}, inplace=True)

# convert all column names to strings
immigrationCanada.columns = list(map(str, immigrationCanada.columns))

# full range of the time series
immigrationYears = list(map(str, range(1980, 2014)))

immigrationCanadaCopy = immigrationCanada.copy()

# add Total of one countries for all years
immigrationCanada['Total'] = immigrationCanada.sum(axis=1)

# add Total of all countries for each year
immigrationCanadaCopy.loc['Total_year'] = immigrationCanada[immigrationCanada.columns[4:]].sum(axis=0)

# Extract dataframe for continent Asia
immigrationAsia = immigrationCanada.loc[immigrationCanada['Continent'] == 'Asia']

# sort based on the Total column values
immigrationAsia.sort_values(by=['Total'], inplace=True, ascending=False)

# top countries in Asia continent having highest immigrants to Canada
immigrationAsia = immigrationAsia.head()

print('Top 5 Asian Countries having Immigrants to Canada')
print(immigrationAsia[['Country', 'Continent', 'Total']])

# we can use the sum() method to get the total population per year
topAsiaTotals = pd.DataFrame(immigrationAsia[immigrationYears].sum(axis=0))

# change the years to type int (useful for regression later on)
topAsiaTotals.index = map(int, topAsiaTotals.index)

# reset the index to put in back in as a column in the df_tot dataframe
topAsiaTotals.reset_index(inplace=True)

# rename columns
topAsiaTotals.columns = ['year', 'total']

# view the final dataframe
topAsiaTotals.head()


# plot the total of all 5 countries against each year
topAsiaTotals.plot(kind='bar',
                   x='year',
                   y='total',
                   figsize=(12, 6)
                   )

plt.title('Total Immigration to Canada from 1980 - 2013 for sum of top 5 counties from Asia')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()

# drop column which are not required
immigrationAsia.drop(['Continent', 'Region', 'DevName'], axis=1, inplace=True)

# line types
lineStyles = ['-', '--', '-.', ':', '--']

# go through each country in list and plot graph for each
plt.figure(figsize=(16, 12))
for i in range(5):
    countryData = list(immigrationAsia.iloc[i].values)
    plt.plot(immigrationYears, countryData[1:-1], label=countryData[0], linestyle=lineStyles[i], linewidth=3)
plt.title('Top Asian Country Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.yticks(np.arange(0, 50000, step=5000))
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# How will you show the relative contribution of each of above 5
# countries as a fraction of total immigration from all countries over
# each year from 1980 through 2013 on the same plot ?
# we can use the sum() method to get the total population per year
immigrationAsia.drop('Total', inplace=True, axis=1)

# get total of all countries for each year
totalCountryYear = list(immigrationCanadaCopy.iloc[-1].values)

asiaRelativeContributions = immigrationAsia.drop('Country', axis=1)

# calculate the relative percentage w.r.t total immigrant count
asiaRelativeContributions = asiaRelativeContributions.div(totalCountryYear[4:]) * 100

print(asiaRelativeContributions)

# insert the country back
asiaRelativeContributions.insert(0, 'Country', immigrationAsia['Country'].values)

# go through each country in list and plot graph for each
immigrationYears = np.arange(1980, 2014)
plt.figure(figsize=(16, 12))
for i in range(5):
    countryData = list(asiaRelativeContributions.iloc[i].values)
    y_x = [x + i * .16 for x in immigrationYears]
    plt.bar(y_x, countryData[1:], width=0.16, label=countryData[0])
plt.title('Asian Country Immigration vs All to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('% Immigrants relative to total')
plt.xticks(np.arange(1980, 2014, step=1))
plt.xticks(rotation=90)
plt.yticks(np.arange(0, 25, step=1))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

