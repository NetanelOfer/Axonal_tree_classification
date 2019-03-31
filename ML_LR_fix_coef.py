import os.path
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import random
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Axon:
df=pd.read_csv('axon16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')

# Dendrite:
df = pd.read_csv('dend16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')

## Logistic Regression
logreg = linear_model.LogisticRegression(penalty='l2', C=1)
df['is_train'] = (np.random.uniform(0, 1, len(df)) <= .75)
yy = pd.factorize(df['name'][df['is_train']==True]) #, sort=True

XX, TEST = df[df['is_train']==True], df[df['is_train']==False]
ttt = TEST['name']
fff = TEST['file']

df1 = df.copy()
df1['name'] = (df1['name']=='Martinotti')
df2=df.copy()
df2['name'] = (df2['name']=='basket')
df3=df.copy()
df3['name'] = (df3['name']=='bitufted')
df4 = df.copy()
df4['name'] = (df4['name']=='chandelier')
df5=df.copy()
df5['name'] = (df5['name']=='double-bouquet')
df6=df.copy()
df6['name'] = (df6['name']=='neurogliaform')

## for Martinotti:
XX, TEST = df1[df1['is_train']==True], df1[df1['is_train']==False]
X = XX.iloc[:,:-3]
TEST = TEST.iloc[:,:-3]
TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
yyy = pd.factorize(df1['name'][df1['is_train']==True], sort=True) #
tt = df1['name'][df1['is_train']==False]
logreg.fit(X,yyy[0])
Mar_prob = logreg.predict_proba(TEST )[:,1]
coef_Mar = logreg.coef_

## for basket:
XX, TEST = df2[df2['is_train']==True], df2[df2['is_train']==False]
X = XX.iloc[:,:-3]
TEST = TEST.iloc[:,:-3]
TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
yyy = pd.factorize(df2['name'][df2['is_train']==True])
tt = df2['name'][df2['is_train']==False]
logreg.fit(X,yyy[0])
bsk_prob = logreg.predict_proba(TEST )[:,1]
coef_bsk = logreg.coef_

## for bitufted:
XX, TEST = df3[df3['is_train']==True], df3[df3['is_train']==False]
X = XX.iloc[:,:-3]
TEST = TEST.iloc[:,:-3]
TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
yyy = pd.factorize(df3['name'][df3['is_train']==True])
tt = df3['name'][df3['is_train']==False]
logreg.fit(X,yyy[0])
bituf_prob = logreg.predict_proba(TEST )[:,1]
coef_bituf = logreg.coef_

## for chandelier:
XX, TEST = df4[df4['is_train']==True], df4[df4['is_train']==False]
X = XX.iloc[:,:-3]
TEST = TEST.iloc[:,:-3]
TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
yyy = pd.factorize(df4['name'][df4['is_train']==True])
tt = df4['name'][df4['is_train']==False]
logreg.fit(X,yyy[0])
chc_prob = logreg.predict_proba(TEST )[:,1]
coef_chc = logreg.coef_

## for double-bouquet:
XX, TEST = df5[df5['is_train']==True], df5[df5['is_train']==False]
X = XX.iloc[:,:-3]
TEST = TEST.iloc[:,:-3]
TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
yyy = pd.factorize(df5['name'][df5['is_train']==True])
tt = df5['name'][df5['is_train']==False]
logreg.fit(X,yyy[0])
dbc_prob = logreg.predict_proba(TEST )[:,1]
coef_dbc = logreg.coef_

## for neurogliaform:
XX, TEST = df6[df6['is_train']==True], df6[df6['is_train']==False]
X = XX.iloc[:,:-3]
TEST = TEST.iloc[:,:-3]
TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
yyy = pd.factorize(df6['name'][df6['is_train']==True])
tt = df6['name'][df6['is_train']==False]
logreg.fit(X,yyy[0])
ngf_prob = logreg.predict_proba(TEST )[:,1]
coef_ngf = logreg.coef_


mul_prob = np.concatenate(([Mar_prob], [bsk_prob], [bituf_prob], [chc_prob], [dbc_prob], [ngf_prob]))
coef_l2 = [np.concatenate((coef_Mar, coef_bsk, coef_bituf, coef_chc, coef_dbc, coef_ngf))]

ppp=[]
for k in range(np.shape(mul_prob)[1]):
  ppp.append(np.argsort(mul_prob.transpose()[k])[-1] )

preds = yy[1][ ppp ]
pp= preds

for i in range(1000):
  logreg = linear_model.LogisticRegression(penalty='l2', C=1)
  df['is_train'] = (np.random.uniform(0, 1, len(df)) <= .75)
  yy = pd.factorize(df['name'][df['is_train']==True])
  XX, TEST = df[df['is_train']==True], df[df['is_train']==False]
  ttt = ttt.append(TEST['name'])
  fff = fff.append(TEST['file'])
  #
  df1 = df.copy()
  df1['name'] = (df1['name']=='Martinotti')
  df2=df.copy()
  df2['name'] = (df2['name']=='basket')
  df3=df.copy()
  df3['name'] = (df3['name']=='bitufted')
  df4 = df.copy()
  df4['name'] = (df4['name']=='chandelier')
  df5=df.copy()
  df5['name'] = (df5['name']=='double-bouquet')
  df6=df.copy()
  df6['name'] = (df6['name']=='neurogliaform')
   
  ## for Martinotti:
  XX, TEST = df1[df1['is_train']==True], df1[df1['is_train']==False]
  X = XX.iloc[:,:-3]
  TEST = TEST.iloc[:,:-3]
  TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
  X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
  yyy = pd.factorize(df1['name'][df1['is_train']==True], sort=True) #
  tt = df1['name'][df1['is_train']==False]
  logreg.fit(X,yyy[0])
  Mar_prob = logreg.predict_proba(TEST )[:,1]
  coef_Mar = logreg.coef_
   
  ## for basket:
  XX, TEST = df2[df2['is_train']==True], df2[df2['is_train']==False]
  X = XX.iloc[:,:-3]
  TEST = TEST.iloc[:,:-3]
  TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
  X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
  yyy = pd.factorize(df2['name'][df2['is_train']==True])
  tt = df2['name'][df2['is_train']==False]
  logreg.fit(X,yyy[0])
  bsk_prob = logreg.predict_proba(TEST )[:,1]
  coef_bsk = logreg.coef_
   
  ## for bitufted:
  XX, TEST = df3[df3['is_train']==True], df3[df3['is_train']==False]
  X = XX.iloc[:,:-3]
  TEST = TEST.iloc[:,:-3]
  TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
  X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
  yyy = pd.factorize(df3['name'][df3['is_train']==True])
  tt = df3['name'][df3['is_train']==False]
  logreg.fit(X,yyy[0])
  bituf_prob = logreg.predict_proba(TEST )[:,1]
  coef_bituf = logreg.coef_
   
  ## for chandelier:
  XX, TEST = df4[df4['is_train']==True], df4[df4['is_train']==False]
  X = XX.iloc[:,:-3]
  TEST = TEST.iloc[:,:-3]
  TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
  X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
  yyy = pd.factorize(df4['name'][df4['is_train']==True])
  tt = df4['name'][df4['is_train']==False]
  logreg.fit(X,yyy[0])
  chc_prob = logreg.predict_proba(TEST )[:,1]
  coef_chc = logreg.coef_
   
  ## for double-bouquet:
  XX, TEST = df5[df5['is_train']==True], df5[df5['is_train']==False]
  X = XX.iloc[:,:-3]
  TEST = TEST.iloc[:,:-3]
  TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
  X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
  yyy = pd.factorize(df5['name'][df5['is_train']==True])
  tt = df5['name'][df5['is_train']==False]
  logreg.fit(X,yyy[0])
  dbc_prob = logreg.predict_proba(TEST )[:,1]
  coef_dbc = logreg.coef_
   
  ## for neurogliaform:
  XX, TEST = df6[df6['is_train']==True], df6[df6['is_train']==False]
  X = XX.iloc[:,:-3]
  TEST = TEST.iloc[:,:-3]
  TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
  X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
  yyy = pd.factorize(df6['name'][df6['is_train']==True])
  tt = df6['name'][df6['is_train']==False]
  logreg.fit(X,yyy[0])
  ngf_prob = logreg.predict_proba(TEST )[:,1]
  coef_ngf = logreg.coef_
   
  mul_prob = np.concatenate(([Mar_prob], [bsk_prob], [bituf_prob], [chc_prob], [dbc_prob], [ngf_prob]))
   
  ppp=[]
  for k in range(np.shape(mul_prob)[1]):
    ppp.append(np.argsort(mul_prob.transpose()[k])[-1] )
   
  coef_all = np.concatenate((coef_Mar, coef_bsk, coef_bituf, coef_chc, coef_dbc, coef_ngf))
  coef_l2 = np.append(coef_l2,[coef_all],axis=0)
  preds = yy[1][ ppp ]
  pp = pp.append(preds)

conf_mat = pd.crosstab(ttt, pp, rownames=['Actual Species'], colnames=['Predicted Species'])
coef_l2_std = coef_l2.std(axis=0)
coef_l2 = coef_l2.mean(axis=0)
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
F1 = 2*(MATs*MATp)/(MATs+MATp)
F1[np.isnan(F1)]=0
F1
print "score:" ,np.mean(np.diag(F1))



pd.set_option('display.max_rows', None)
pd.sort_values('file', ascending=False)
aa=pd.crosstab([ttt,fff] , pp)
# aa.to_csv('neurons_detect_six.csv')
##aa.to_csv('neurons_detect_ax_dn_ac_six.csv')




###### six groups - LR coeff - Axons:
fig, (ax1, ax2 ,ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=False, figsize=(15,8))
ax1.barh(range(len(coef_l2[0])),coef_l2[0], 1/1.5, xerr=coef_l2_std[0], color="blue")
ax1.set_yticks([])
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on',right='on',left='off')
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()
ax5.invert_yaxis()
ax6.invert_yaxis()
ax2.barh(range(len(coef_l2[1])),coef_l2[1], 1/1.5, xerr=coef_l2_std[1], color="blue")
ax2.set_yticks([])
ax3.barh(range(len(coef_l2[2])),coef_l2[2], 1/1.5, xerr=coef_l2_std[2], color="blue")
ax3.set_yticks([])
ax4.barh(range(len(coef_l2[3])),coef_l2[3], 1/1.5, xerr=coef_l2_std[3], color="blue")
ax4.set_yticks([])
ax5.barh(range(len(coef_l2[4])),coef_l2[4], 1/1.5, xerr=coef_l2_std[4], color="blue")
ax5.set_yticks([])
ax6.barh(range(len(coef_l2[5])),coef_l2[5], 1/1.5, xerr=coef_l2_std[5], color="blue")
ax6.set_yticks(range(len(coef_l2[0])))
ll = 1.3 
ax1.set_xlim([-ll,ll])
ax2.set_xlim([-ll,ll])
ax3.set_xlim([-ll,ll])
ax4.set_xlim([-ll,ll])
ax5.set_xlim([-ll,ll])
ax6.set_xlim([-ll,ll])
ax6.set_yticklabels(list(df),fontsize=16)
ax1.set_xlabel('Martinotti',fontsize=18)
ax2.set_xlabel('basket',fontsize=18)
ax3.set_xlabel('bitufted',fontsize=18)
ax4.set_xlabel('chandelier',fontsize=18)
ax5.set_xlabel('double-bouquet',fontsize=18)
ax6.set_xlabel('neurogliaform',fontsize=18)
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')
ax1.text(-1.65,-1.5,'A', fontsize=20, fontproperties=font)
fig.tight_layout()
sns.set_style('darkgrid')
sns.set_style('white')
fig.subplots_adjust(left=0.02,wspace=0.05)
fig.savefig('coeff_axons.pdf')


###### six groups - LR coeff - Dendrites:
fig, (ax1, ax2 ,ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=False, figsize=(15,8))
ax1.barh(range(len(coef_l2[0])),coef_l2[0], 1/1.5, xerr=coef_l2_std[0], color="red")
ax1.set_yticks([])
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on',right='on',left='off')
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()
ax5.invert_yaxis()
ax6.invert_yaxis()
ax2.barh(range(len(coef_l2[1])),coef_l2[1], 1/1.5, xerr=coef_l2_std[1], color="red")
ax2.set_yticks([])
ax3.barh(range(len(coef_l2[2])),coef_l2[2], 1/1.5, xerr=coef_l2_std[2], color="red")
ax3.set_yticks([])
ax4.barh(range(len(coef_l2[3])),coef_l2[3], 1/1.5, xerr=coef_l2_std[3], color="red")
ax4.set_yticks([])
ax5.barh(range(len(coef_l2[4])),coef_l2[4], 1/1.5, xerr=coef_l2_std[4], color="red")
ax5.set_yticks([])
ax6.barh(range(len(coef_l2[5])),coef_l2[5], 1/1.5, xerr=coef_l2_std[5], color="red")
ax6.set_yticks(range(len(coef_l2[0])))
ll = 1.3
ax1.set_xlim([-ll,ll])
ax2.set_xlim([-ll,ll])
ax3.set_xlim([-ll,ll])
ax4.set_xlim([-ll,ll])
ax5.set_xlim([-ll,ll])
ax6.set_xlim([-ll,ll])
ax6.set_yticklabels(list(df),fontsize=16)
ax1.set_xlabel('Martinotti',fontsize=18)
ax2.set_xlabel('basket',fontsize=18)
ax3.set_xlabel('bitufted',fontsize=18)
ax4.set_xlabel('chandelier',fontsize=18)
ax5.set_xlabel('double-bouquet',fontsize=18)
ax6.set_xlabel('neurogliaform',fontsize=18)
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')
ax1.text(-1.65,-1.5,'B', fontsize=20, fontproperties=font)
fig.tight_layout()
sns.set_style('darkgrid')
sns.set_style('white')
fig.subplots_adjust(left=0.02,wspace=0.05)
fig.savefig('coeff_dendrites.pdf')

