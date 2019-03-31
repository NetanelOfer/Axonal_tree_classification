import os.path
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import random
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

## import the data
# Axon:
df = pd.read_csv('axon16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')

# Dendrite:
df = pd.read_csv('dend16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')

# Axon + Dendrite:
df = pd.read_csv('axon16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')
df2 = df
df2 = df2.drop('name', 1)
df = pd.read_csv('dend16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = pd.merge(df2, df,on='file')
df = df.reindex(columns=sorted(df.columns))

# Activity:
df = pd.read_csv('activity_six4.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')

# Activity + axon:
df=pd.read_csv('axon16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')
df = df.drop('name', 1)
abc = pd.read_csv('activity_six4.csv')
abc = abc.drop('Unnamed: 0', 1)
df = pd.merge(df, abc,on='file')
df = df.reindex(columns=sorted(df.columns))

# Activity + axon + Dendrite:
df = pd.read_csv('axon16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = df.sort_values('name')
df2 = df
df2 = df2.drop('name', 1)
df = pd.read_csv('dend16_six.csv')
df = df.drop('Unnamed: 0', 1)
df = pd.merge(df2, df,on='file')
df = df.reindex(columns=sorted(df.columns))
df = df.drop('name', 1)
abc = pd.read_csv('activity_six4.csv')
abc = abc.drop('Unnamed: 0', 1)
df = pd.merge(df, abc,on='file')
df = df.reindex(columns=sorted(df.columns))


## Logistic Regression
def LR(FS_discard):
  logreg = linear_model.LogisticRegression(penalty='l2', C=1)
  df['is_train'] = (np.random.uniform(0, 1, len(df)) <= .75)
  yy = pd.factorize(df['name'][df['is_train']==True])
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
  yyy = pd.factorize(df1['name'][df1['is_train']==True], sort=True)
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
  coef_l2 = np.concatenate((coef_Mar, coef_bsk, coef_bituf, coef_chc, coef_dbc, coef_ngf))
  ppp=[]
  for k in range(np.shape(mul_prob)[1]):
    ppp.append(np.argsort(mul_prob.transpose()[k])[-1] )
  
  preds = yy[1][ ppp ]
  pp= preds
  for fs in range(FS_discard):  # Feature selection
    ind_Mar   = [i for i in np.argsort(abs(coef_l2[0]))[1:]] +[-3]+[-2]+[-1]
    ind_bsk   = [i for i in np.argsort(abs(coef_l2[1]))[1:]] +[-3]+[-2]+[-1]
    ind_bituf = [i for i in np.argsort(abs(coef_l2[2]))[1:]] +[-3]+[-2]+[-1]
    ind_chc   = [i for i in np.argsort(abs(coef_l2[3]))[1:]] +[-3]+[-2]+[-1]
    ind_dbc   = [i for i in np.argsort(abs(coef_l2[4]))[1:]] +[-3]+[-2]+[-1]
    ind_ngf   = [i for i in np.argsort(abs(coef_l2[5]))[1:]] +[-3]+[-2]+[-1]
    
    fts_Mar = list(df1.iloc[:,ind_Mar]) # all the remained features
    fts_bsk = list(df2.iloc[:,ind_bsk])
    fts_bituf = list(df3.iloc[:,ind_bituf])
    fts_chc = list(df4.iloc[:,ind_chc])
    fts_dbc = list(df5.iloc[:,ind_dbc])
    fts_ngf = list(df6.iloc[:,ind_ngf])
    
    ## Logistic Regression - Second round
    logreg = linear_model.LogisticRegression(penalty='l2', C=1)
    df['is_train'] = (np.random.uniform(0, 1, len(df)) <= .75)
    yy = pd.factorize(df['name'][df['is_train']==True])
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
    
    df1 = df1[fts_Mar]
    df2 = df2[fts_bsk]
    df3 = df3[fts_bituf]
    df4 = df4[fts_chc]
    df5 = df5[fts_dbc]
    df6 = df6[fts_ngf]
    
    ## for Martinotti:
    XX, TEST = df1[df1['is_train']==True], df1[df1['is_train']==False]
    X = XX.iloc[:,:-3]
    TEST = TEST.iloc[:,:-3]
    TEST = (TEST - X.mean())/ (X.std() + np.finfo(float).eps)
    X = (X - X.mean())/ ( X.std() + np.finfo(float).eps)
    yyy = pd.factorize(df1['name'][df1['is_train']==True], sort=True)
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
    coef_l2 = np.concatenate((coef_Mar, coef_bsk, coef_bituf, coef_chc, coef_dbc, coef_ngf))
    ppp=[]
    for k in range(np.shape(mul_prob)[1]):
      ppp.append(np.argsort(mul_prob.transpose()[k])[-1] )
    
    preds = yy[1][ ppp ]
    pp= preds
  return(ttt,pp,fff)


FS_discard = 485 
# Axonal morphology - 17, Dendritic morphology - 17 , Axonal + Dendritic morphology - 45, 
# activity - 415 , activity+axons - 445 , activity+axon+dend - 485
CM = LR(FS_discard)
ttt = CM[0]
pp = CM[1]
fff = CM[2]
for i in range(1000):
  CM = LR(FS_discard)
  ttt = ttt.append(CM[0])
  pp = pp.append(CM[1])
  fff = fff.append(CM[2])
  print i


conf_mat = pd.crosstab(ttt, pp, rownames=['Actual Species'], colnames=['Predicted Species'])
MATs = ((conf_mat.T)/conf_mat.sum(1)).T
MATp = ((conf_mat)/conf_mat.sum(0))
F1 = 2*(MATs*MATp)/(MATs+MATp)
F1[np.isnan(F1)]=0
F1
print "score:" ,np.mean(np.diag(F1))

##
pd.set_option('display.max_rows', None)
pd.sort_values('file', ascending=False)
aa=pd.crosstab([ttt,fff] , pp)
aa
# aa.to_csv('neurons_detect_six94.csv')


#conf_mat.to_csv('conf_mat_axon.csv') # 0.7773085980126834
#conf_mat.to_csv('conf_mat_dend.csv') # 0.48819346324670754
#conf_mat.to_csv('conf_mat_axon_dend.csv') # 0.8168883346860051

#conf_mat.to_csv('conf_mat_act.csv') # 0.7668218888591354
#conf_mat.to_csv('conf_mat_axon_act.csv') # 0.8949442781180103
#conf_mat.to_csv('conf_mat_axon_dend_act.csv') # 0.9352124392935816




