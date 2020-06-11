import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import the data directly into a pandas dataframe
nba = pd.read_csv("ppg2008.csv", index_col='Name  ')
# remove index title
nba.index.name = ""
print(type(nba))
# normalize data columns
nba_norm = (nba - nba.mean()) / (nba.max() - nba.min())
# relabel columns
labels = ['Games', 'Minutes', 'Points', 'Field goals made', 'Field goal attempts', 'Field goal percentage', 'Free throws made',
          'Free throws attempts', 'Free throws percentage','Three-pointers made', 'Three-point attempt', 'Three-point percentage',
          'Offensive rebounds', 'Defensive rebounds', 'Total rebounds', 'Assists', 'Steals', 'Blocks', 'Turnover', 'Personal foul']
nba_norm.columns = labels
print(nba_norm)
# set appropriate font and dpi
sns.set(font_scale=1.2)
sns.set_style({"savefig.dpi": 100})
# plot it out
ax = sns.heatmap(nba_norm, cmap=plt.cm.Blues, linewidths=.1)
# set the x-axis labels on the top
ax.xaxis.tick_top()
# rotate the x-axis labels
plt.xticks(rotation=90)
plt.yticks(rotation=90)
# get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
fig = ax.get_figure()
# specify dimensions and save
fig.set_size_inches(15, 20)
fig.savefig("nba.png")
plt.show()