import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

names = ['Martinotti','basket','bitufted','chandelier','double-bouquet','neurogliaform']
lbl_x = ['MC','BC','BTC','CHC','DBC','NGF']

# draw confusion matrixes - Morphology (Figure *)
conf_mat = pd.read_csv('conf_mat_axon.csv', header=0, index_col=0)
conf_mat.columns = names
conf_mat.index = names
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
del MATs.index.name
F1 = 2*(MATs*MATp)/(MATs+MATp)
F1[np.isnan(F1)]=0

conf_mat = pd.read_csv('conf_mat_dend.csv', header=0, index_col=0)
conf_mat.columns = names
conf_mat.index = names
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
del MATs.index.name
F2 = 2*(MATs*MATp)/(MATs+MATp)
F2[np.isnan(F2)]=0

conf_mat = pd.read_csv('conf_mat_axon_dend.csv', header=0, index_col=0)
conf_mat.columns = names
conf_mat.index = names
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
del MATs.index.name
F3 = 2*(MATs*MATp)/(MATs+MATp)
F3[np.isnan(F3)]=0


fig, (ax1,ax2,ax3 ) = plt.subplots(1, 3, sharey=False, figsize=(17,6))
## A
ax1.imshow(F1,zorder=1, vmin=0, vmax=1 ,cmap=plt.cm.Blues)
lbl = list(F1.index)
ax1.yaxis.set_ticks(range(6))
ax1.xaxis.set_ticks(range(6))
ax1.set_xticklabels(lbl_x,size=14)
ax1.set_yticklabels(lbl_x,size=14)
ax1.xaxis.tick_top()
for (j,i),label in np.ndenumerate(F1):
    if (label>0.5):
      ax1.text(i,j,round(label, 3),ha='center',va='center',color='white',size=14)
    else:
      ax1.text(i,j,round(label, 3),ha='center',va='center',color='black',size=14)

ax1.text(2.5,-1.2,'Axonal morphology',size=20,ha='center',va='center')

## B
ax2.imshow(F2,zorder=1, vmin=0, vmax=1 ,cmap=plt.cm.Reds)
lbl = list(F2.index)
ax2.yaxis.set_ticks(range(6))
ax2.xaxis.set_ticks(range(6))
ax2.set_xticklabels(lbl_x,size=14)
ax2.set_yticklabels([],size=14)
ax2.xaxis.tick_top()
for (j,i),label in np.ndenumerate(F2):
    if (label>0.5):
      ax2.text(i,j,round(label, 3),ha='center',va='center',color='white',size=14)
    else:
      ax2.text(i,j,round(label, 3),ha='center',va='center',color='black',size=14)

ax2.text(2.5,-1.2,'Dendritic morphology',size=20,ha='center',va='center')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# C
ax3.imshow(F3,zorder=1, vmin=0, vmax=1 ,cmap=plt.cm.Purples)
lbl = list(F3.index)
ax3.yaxis.set_ticks(range(6))
ax3.xaxis.set_ticks(range(6))
ax3.set_xticklabels(lbl_x,size=14)
ax3.set_yticklabels([],size=14)
ax3.xaxis.tick_top()
for (j,i),label in np.ndenumerate(F3):
    if (label>0.5):
      ax3.text(i,j,round(label, 3),ha='center',va='center',color='white',size=14)
    else:
      ax3.text(i,j,round(label, 3),ha='center',va='center',color='black',size=14)

ax3.text(2.5,-1.2,'Axonal and dendritic morphology',size=20,ha='center',va='center')

plt.subplots_adjust(left=0.03, bottom=0.01, right=1.01, top=0.86, wspace=0.001, hspace=0.01)

from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')

ax1.text(-1.15,-1.2,'a',size=24,ha='center',va='center', fontproperties=font)
ax2.text(-0.6,-1.2,'b',size=24,ha='center',va='center', fontproperties=font)
ax3.text(-0.6,-1.2,'c',size=24,ha='center',va='center', fontproperties=font)

plt.savefig('confusion-matrix-morphology.pdf')

print "score:" ,np.mean(np.diag(F1))
print "score:" ,np.mean(np.diag(F2))
print "score:" ,np.mean(np.diag(F3))






# draw confusion matrixes - Activity (Figure *)
conf_mat = pd.read_csv('conf_mat_act.csv', header=0, index_col=0) # 0.5196016472205472
conf_mat.columns = names
conf_mat.index = names
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
del MATs.index.name
F1 = 2*(MATs*MATp)/(MATs+MATp)
F1[np.isnan(F1)]=0

conf_mat = pd.read_csv('conf_mat_axon_act.csv', header=0, index_col=0) # 0.8555148290794201
conf_mat.columns = names
conf_mat.index = names
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
del MATs.index.name
F2 = 2*(MATs*MATp)/(MATs+MATp)
F2[np.isnan(F2)]=0

conf_mat = pd.read_csv('conf_mat_axon_dend_act.csv', header=0, index_col=0) # 0.86212956644785
conf_mat.columns = names
conf_mat.index = names
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
del MATs.index.name
F3 = 2*(MATs*MATp)/(MATs+MATp)
F3[np.isnan(F3)]=0


fig, (ax1,ax2,ax3 ) = plt.subplots(1, 3, sharey=False, figsize=(17,6))
## A
ax1.imshow(F1,zorder=1, vmin=0, vmax=1 ,cmap=plt.cm.Oranges)
lbl = list(F1.index)
ax1.yaxis.set_ticks(range(6))
ax1.xaxis.set_ticks(range(6))
ax1.set_xticklabels(lbl_x,size=14)
ax1.set_yticklabels(lbl_x,size=14)
ax1.xaxis.tick_top()
for (j,i),label in np.ndenumerate(F1):
    if (label>0.5):
      ax1.text(i,j,round(label, 3),ha='center',va='center',color='white',size=14)
    else:
      ax1.text(i,j,round(label, 3),ha='center',va='center',color='black',size=14)

ax1.text(2.5,-1.2,'Axonal activity',size=20,ha='center',va='center')

## B
ax2.imshow(F2,zorder=1, vmin=0, vmax=1 ,cmap=plt.cm.Greens)
lbl = list(F2.index)
ax2.yaxis.set_ticks(range(6))
ax2.xaxis.set_ticks(range(6))
ax2.set_xticklabels(lbl_x,size=14)
ax2.set_yticklabels([],size=14)
ax2.xaxis.tick_top()
for (j,i),label in np.ndenumerate(F2):
    if (label>0.5):
      ax2.text(i,j,round(label, 3),ha='center',va='center',color='white',size=14)
    else:
      ax2.text(i,j,round(label, 3),ha='center',va='center',color='black',size=14)

ax2.text(2.5,-1.2,'Axonal morphology and activity',size=20,ha='center',va='center')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# C
ax3.imshow(F3,zorder=1, vmin=0, vmax=1 ,cmap=plt.cm.cool)
lbl = list(F3.index)
ax3.yaxis.set_ticks(range(6))
ax3.xaxis.set_ticks(range(6))
ax3.set_xticklabels(lbl_x,size=14)
ax3.set_yticklabels([],size=14)
ax3.xaxis.tick_top()
for (j,i),label in np.ndenumerate(F3):
    if (label>0.5):
      ax3.text(i,j,round(label, 3),ha='center',va='center',color='white',size=14)
    else:
      ax3.text(i,j,round(label, 3),ha='center',va='center',color='black',size=14)

ax3.text(2.5,-1.2,'All',size=20,ha='center',va='center')

plt.subplots_adjust(left=0.03, bottom=0.01, right=1.01, top=0.86, wspace=0.001, hspace=0.01)

from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')

ax1.text(-1.15,-1.2,'a',size=24,ha='center',va='center', fontproperties=font)
ax2.text(-0.6,-1.2,'b',size=24,ha='center',va='center', fontproperties=font)
ax3.text(-0.6,-1.2,'c',size=24,ha='center',va='center', fontproperties=font)

plt.savefig('confusion-matrix-activity.pdf')

print "score:" ,np.mean(np.diag(F1))
print "score:" ,np.mean(np.diag(F2))
print "score:" ,np.mean(np.diag(F3))

