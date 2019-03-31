from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## import the data
# Axon:
df = pd.read_csv('axon16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')
df.index = df['file']
#
typ = df['name']
df = df.iloc[:,0:-2]
for k in list(df):
  df[k] = ( df[k] - mean(df[k]) ) / std(df[k])

clr={'Martinotti':'red','basket':'hotpink','bitufted':'orange','chandelier':'deepskyblue','double-bouquet':'lime','neurogliaform':'navy'}
row_colors = typ.map(clr)
del row_colors.index.name

fig = plt.figure()
cg = sns.clustermap(df.iloc[:,0:-2],row_cluster=True,col_cluster=True, row_colors=row_colors,  metric="cosine", linewidths=0.2,figsize=(26, 32))
HHM = 1.15 # The hight of the heatmap
hm = cg.ax_heatmap.get_position()
cg.ax_heatmap.set_position([hm.x0+0.0, hm.y0, hm.width*0.82, hm.height*HHM])
cl_rw = cg.ax_row_colors.get_position()
cg.ax_row_colors.set_position([cl_rw.x0+(cl_rw.width*0.7) , cl_rw.y0 ,cl_rw.width*0.3 ,cl_rw.height*HHM] )
cg.ax_row_colors.set_axis_off()
row = cg.ax_row_dendrogram.get_position()
cg.ax_row_dendrogram.set_position([row.x0-(row.width*2.0)+(cl_rw.width*0.7), row.y0, row.width*3.0, row.height*HHM])

cg.ax_col_dendrogram.set_position([hm.x0+0.0, (hm.height*HHM)+hm.y0+0.002 , hm.width*0.82 ,0.07])

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), fontsize=20, rotation=0)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), fontsize=24, rotation=90)
plt.setp(cg.cax.yaxis.get_majorticklabels(), fontsize=20, rotation=0)

cg.cax.set_position([.855, hm.y0+hm.height*HHM*0.91, .017, hm.height*HHM*0.09])

# To print the interneuronal types and colors:
typ2 = ['Martinotti','basket','bitufted','chandelier','double-bouquet','neurogliaform']
k=0
for i in linspace(0.8,2.4,6)*-1:
    plt.text(1.4,i,typ2[k], color='k',fontsize=30)
    plt.text(0,i, r'$\blacksquare$' , color=clr[typ2[k]],fontsize=40,ha='left')
    k=k+1

plt.savefig('Unsupervised.pdf')

