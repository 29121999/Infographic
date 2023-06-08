import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec



pd.set_option('display.max_columns', None)

weoDF = pd.read_csv(r'C:\Users\Sahil Khan\OneDrive\Desktop\WEO_Data.csv')

int_col = ['WEO Country Code']
num_col = [str(i) for i in range(2000,2029)]

for col in weoDF.columns:
    if col in num_col:
        weoDF[col] = pd.to_numeric(weoDF[col], errors='coerce')  # this will convert non-numeric values to NaN
        weoDF[col] = weoDF[col].fillna(0)  # replace NaN with 0
        weoDF[col] = weoDF[col].replace([np.inf, -np.inf], 0)  # replace inf with 99999
        weoDF[col] = weoDF[col].astype('float')  # convert to integer
    elif col in int_col:
        weoDF[col] = pd.to_numeric(weoDF[col], errors='coerce')  # this will convert non-numeric values to NaN
        weoDF[col] = weoDF[col].fillna(0)  # replace NaN with 0
        weoDF[col] = weoDF[col].replace([np.inf, -np.inf], 0)  # replace inf with 99999
        weoDF[col] = weoDF[col].astype('int')  # convert to integer
    else:
        weoDF[col] = weoDF[col].astype('object')
        
weoDF.rename(columns = lambda x: x.strip().replace(' ', '_').lower(), inplace=True)

for col in num_col:
    weoDF[col] = pd.np.where(weoDF['scale'] == 'Millions', weoDF[col]*1000000,
                       pd.np.where(weoDF['scale'] == 'Billions', weoDF[col]*1000000000,
                       pd.np.where(weoDF['scale'] == 'Units', weoDF[col]*1,
                       weoDF[col])))


# Set up the dashboard/infographics
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

# Getting GDP DATA
gdp_df = weoDF.loc[(weoDF['subject_descriptor'] == 'Gross domestic product, constant prices') & (weoDF['units'] == 'National currency')]

# Dropping all irrevelant columns
gdp_df = gdp_df[['country'] + num_col]


diff_df = gdp_df[['country','2010','2020']]
diff_df['diff'] = diff_df['2020'] - diff_df['2010']

# Plot top 5 countries
top_5 = diff_df.sort_values('diff', ascending=False).head(5)
top5GDP = top_5.country.to_list()
top5df = gdp_df[gdp_df['country'].isin(top5GDP)]
top5df.set_index('country', inplace=True)
ax1 = fig.add_subplot(gs[0, 0])
top5df.T.plot(ax=ax1, figsize=(10, 10))
ax1.set_xlabel('Year')
ax1.set_ylabel('GDP (in USD)')
ax1.set_title('Top 5 Countries with Highest GDP Growth')



# Plot bottom 5 countries
bottom_5 = diff_df.sort_values('diff', ascending=True).head(5)
bottom5GDP = bottom_5.country.to_list()
bottom5df = gdp_df[gdp_df['country'].isin(bottom5GDP)]
bottom5df.set_index('country', inplace=True)
ax2 = fig.add_subplot(gs[0, 1])
bottom5df.T.plot(ax=ax2, figsize=(10, 10))
ax2.set_xlabel('Year')
ax2.set_ylabel('GDP (in USD)')
ax2.set_title('Bottom 5 Countries with Lowest GDP Growth')


govRev_df = weoDF.loc[(weoDF['subject_descriptor'] == 'General government revenue') & (weoDF['units'] == 'National currency')]
govRev_df = govRev_df[['country'] + num_col]
govRev_df['total'] = govRev_df.iloc[:, 1:].sum(axis=1)
govRev_df_sorted = govRev_df.sort_values('total', ascending=False).head(10)



# Create a figure and axis object
ax3 = fig.add_subplot(gs[1, 0])
sns.barplot(data=govRev_df_sorted, x='country', y='total', ax=ax3)
ax3.set_title('Total by Country')
ax3.set_xlabel('Country')
ax3.set_ylabel('Total')
# plt.xticks(rotation=90)
# Display the plot
# plt.show()
# ax3



top_5_countries = govRev_df.sort_values('total', ascending=False).head(5)['country'].tolist()
govRev_df['country_category'] = govRev_df['country'].apply(lambda x: x if x in top_5_countries else 'Other')

# govRev_df['country_category'] = govRev_df['country'].apply(lambda x: x if x in top_5_countries else 'Other')

grouped_data = govRev_df.groupby('country_category')['total'].sum()

# ax3 = figSize.add_subplot(gs[1, 1])

# ax.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=90)
# ax.set_title('Total by Country Category')
# # plt.show()

plt.subplot(gs[1,1])
plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%')
plt.title('Distribution of Top 10 Industries')


fig.savefig("abc.png", dpi=300)
plt.show()




