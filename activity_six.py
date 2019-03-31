import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns

Double_clls = ['NMO_37699','NMO_37363','NMO_37171','NMO_37469','NMO_37695','NMO_37422','NMO_37148','NMO_37178','NMO_37184','NMO_61613', 'NMO_61618','NMO_37423','NMO_37475','NMO_37694','NMO_37724','NMO_37762']
Martinotti_clls = ['NMO_06140','NMO_79465','NMO_79458','NMO_79459','NMO_79464','NMO_79462','NMO_79457','NMO_79463','NMO_37672','NMO_37787' , 'NMO_37301','NMO_37190','NMO_36965','NMO_37297','NMO_37291','NMO_37290']
Bitufted_clls=['NMO_61580','NMO_37302','NMO_37776','NMO_37385','NMO_37781','NMO_37296','NMO_37704','NMO_37324','NMO_37691','NMO_37316' ,'NMO_37099','NMO_37110','NMO_37096','NMO_37127','NMO_37244','NMO_61570']
Chandelier_clls=['NMO_35831','NMO_04548','NMO_37138','NMO_37763','NMO_37424','NMO_37133','NMO_37821','NMO_37764','NMO_37548','NMO_37818' ,'NMO_07472','NMO_37247','NMO_37114','NMO_37391','NMO_37537','NMO_36983']
Basket_clls = ['NMO_79468','NMO_79469','NMO_37137','NMO_37844','NMO_37310','NMO_37862','NMO_37529','NMO_37188','NMO_37841','NMO_06143' ,'NMO_37504','NMO_37117','NMO_37311','NMO_37015','NMO_37112','NMO_37505']
Neurogliaform_clls = ['NMO_37720','NMO_37623','NMO_37640','NMO_37625','NMO_37721','NMO_37243','NMO_37061','NMO_37075','NMO_06138' ,'NMO_37517','NMO_37062','NMO_06139','NMO_37076','NMO_37059','NMO_37661','NMO_37063']

# creation of dataframe
clls = Martinotti_clls + Basket_clls + Bitufted_clls + Chandelier_clls + Double_clls + Neurogliaform_clls
cll = ['red' ,'green', 'orange' , 'deepskyblue' , 'blue' , 'navy' , 'dimgray' , 'silver']

def curve(q,cl):
  DDD = []
  if (cl==0): # sub-trees:
    for i in range(len(clls)):
      D1=[]
      for En in eval(clls[i])[0]:
        norm = [float(ii)/sum(En) for ii in En]
        if (q==1):
          D1.append(math.exp(1)**-(sum([n*np.log(n) for n in norm])))
        else:
          D1.append( (sum([n**q for n in norm]))**(float(1)/(1-q)) ) 
      DDD.append(D1)
  
  else: #if (cl==1):  # for the number of types of responses (colors):
    for i in range(len(clls)):
      C1 = []
      for j, En in enumerate(eval(clls[i])[1]):
        C = [0]*len(cll)
        for k,cc in enumerate(En):
          C[cll.index(cc)] = C[cll.index(cc)] + eval(clls[i])[0][j][k]
        C = filter(lambda a: a != 0, C)
        norm = [float(ii)/sum(C) for ii in C]
        if (q==1):
          C1.append(math.exp(1)**-(sum([n*np.log(n) for n in norm])))
        else:
          C1.append((sum([n**q for n in norm]))**(float(1)/(1-q)))
      DDD.append(C1)
  
  dd = {'ac' : DDD }
  return(dd)


execfile( "cAC_diversity_full.py")
abc1 = pd.DataFrame(curve(0,0)['ac'])
abd1 = pd.DataFrame(curve(0,1)['ac'])
bc1 = pd.DataFrame(curve(1,0)['ac'])
bd1 = pd.DataFrame(curve(1,1)['ac'])
execfile( "cNAC_diversity_full.py")
abc2 = pd.DataFrame(curve(0,0)['ac'])
abd2 = pd.DataFrame(curve(0,1)['ac'])
bc2 = pd.DataFrame(curve(1,0)['ac'])
bd2 = pd.DataFrame(curve(1,1)['ac'])
execfile( "bAC_diversity_full.py")
abc3 = pd.DataFrame(curve(0,0)['ac'])
abd3 = pd.DataFrame(curve(0,1)['ac'])
bc3 = pd.DataFrame(curve(1,0)['ac'])
bd3 = pd.DataFrame(curve(1,1)['ac'])
execfile( "bNAC_diversity_full.py")
abc4 = pd.DataFrame(curve(0,0)['ac'])
abd4 = pd.DataFrame(curve(0,1)['ac'])
bc4 = pd.DataFrame(curve(1,0)['ac'])
bd4 = pd.DataFrame(curve(1,1)['ac'])

abc = pd.concat([abc1,abc2,abc3,abc4,abd1,abd2,abd3,abd4,bc1,bc2,bc3,bc4,bd1,bd2,bd3,bd4],axis=1, ignore_index=True)

abc['file'] = clls
abc['name'] = ['Martinotti']*len(Martinotti_clls) + ['basket']*len(Basket_clls) + ['bitufted']*len(Bitufted_clls) + ['chandelier']*len(Chandelier_clls) + ['double-bouquet']*len(Double_clls) + ['neurogliaform']*len(Neurogliaform_clls)


# df = abc
# df.to_csv('activity_six4.csv')

## df = pd.read_csv('activity_six.csv')
## df = df.drop('Unnamed: 0', 1)


