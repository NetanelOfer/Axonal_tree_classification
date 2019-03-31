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

clls = Basket_clls + Neurogliaform_clls + Martinotti_clls + Double_clls + Bitufted_clls + Chandelier_clls


fig, axs = plt.subplots(2,2, figsize=(15, 9), facecolor='w', edgecolor='k')
sns.set_style('darkgrid')
sns.set_style('white')
axs = axs.ravel()
for clm in range(4):
  if (clm==0):
    execfile( "cAC_diversity_full.py")
  if (clm==1):
    execfile( "cNAC_diversity_full.py")
  if (clm==2):
    execfile( "bAC_diversity_full.py")
  if (clm==3):
    execfile( "bNAC_diversity_full.py")
  
  DDD = []
  for i in range(len(clls)):
    D1=[]
    for En in eval(clls[i])[0]:
      brn_num = sum(eval(clls[i])[0][0])
      norm = [float(ii)/sum(En) for ii in En]
      q=0
      D1.append( ( (sum([n**q for n in norm]))**(float(1)/(1-q)) )  )
    DDD.append(D1)
  
  dd = {'name' : clls , 'Ent' : DDD , 'type':['Basket']*len(Basket_clls) + ['Neurogliaform']*len(Neurogliaform_clls) + ['Martinotti']*len(Martinotti_clls) + ['DoubleBouquet']*len(Double_clls) + ['Bitufted']*len(Bitufted_clls) + ['Chandelier']*len(Chandelier_clls)}
  df = pd.DataFrame(dd)
  
  meanEntropy_Basket=[]
  stdEntropy_Basket=[]
  for j in range(len(df.Ent[0])):
    freq_En=[]
    for i in range(len(df[df.type=='Basket'])): 
      k = df[df.type=='Basket'].index[i]
      freq_En.append(df.Ent[k][j])
    meanEntropy_Basket.append(np.mean(freq_En))
    stdEntropy_Basket.append(np.std(freq_En))
  
  meanEntropy_Neurogliaform=[]
  stdEntropy_Neurogliaform=[]
  for j in range(len(df.Ent[0])):
    freq_En=[]
    for i in range(len(df[df.type=='Neurogliaform'])): 
      k = df[df.type=='Neurogliaform'].index[i]
      freq_En.append(df.Ent[k][j])
    meanEntropy_Neurogliaform.append(np.mean(freq_En))
    stdEntropy_Neurogliaform.append(np.std(freq_En))
  
  meanEntropy_Martinotti=[]
  stdEntropy_Martinotti=[]
  for j in range(len(df.Ent[0])):
    freq_En=[]
    for i in range(len(df[df.type=='Martinotti'])): 
      k = df[df.type=='Martinotti'].index[i]
      freq_En.append(df.Ent[k][j])
    meanEntropy_Martinotti.append(np.mean(freq_En))
    stdEntropy_Martinotti.append(np.std(freq_En))
  
  meanEntropy_DoubleBouquet=[]
  stdEntropy_DoubleBouquet=[]
  for j in range(len(df.Ent[0])):
    freq_En=[]
    for i in range(len(df[df.type=='DoubleBouquet'])): 
      k = df[df.type=='DoubleBouquet'].index[i]
      freq_En.append(df.Ent[k][j])
    meanEntropy_DoubleBouquet.append(np.mean(freq_En))
    stdEntropy_DoubleBouquet.append(np.std(freq_En))
  
  meanEntropy_Bitufted=[]
  stdEntropy_Bitufted=[]
  for j in range(len(df.Ent[0])):
    freq_En=[]
    for i in range(len(df[df.type=='Bitufted'])): 
      k = df[df.type=='Bitufted'].index[i]
      freq_En.append(df.Ent[k][j])
    meanEntropy_Bitufted.append(np.mean(freq_En))
    stdEntropy_Bitufted.append(np.std(freq_En))
  
  meanEntropy_Chandelier=[]
  stdEntropy_Chandelier=[]
  for j in range(len(df.Ent[0])):
    freq_En=[]
    for i in range(len(df[df.type=='Chandelier'])): 
      k = df[df.type=='Chandelier'].index[i]
      freq_En.append(df.Ent[k][j])
    meanEntropy_Chandelier.append(np.mean(freq_En))
    stdEntropy_Chandelier.append(np.std(freq_En))
  
  ################## Plot all
  axs[clm].plot(range(20,620,20), meanEntropy_Martinotti,'red' , label='Martinotti')
  axs[clm].fill_between(range(20,620,20), [i - j/2 for i, j in zip(meanEntropy_Martinotti, stdEntropy_Martinotti)] , [i + j/2 for i, j in zip(meanEntropy_Martinotti, stdEntropy_Martinotti)] , color='red', alpha='0.1') #
  axs[clm].plot(range(20,620,20), meanEntropy_Basket,'hotpink' , label='basket')
  axs[clm].fill_between(range(20,620,20), [i - j/2 for i, j in zip(meanEntropy_Basket, stdEntropy_Basket)] , [i + j/2 for i, j in zip(meanEntropy_Basket, stdEntropy_Basket)] , color='hotpink', alpha='0.1') #
  axs[clm].plot(range(20,620,20), meanEntropy_Bitufted,'orange' , label='bitufted')
  axs[clm].fill_between(range(20,620,20), [i - j/2 for i, j in zip(meanEntropy_Bitufted, stdEntropy_Bitufted)] , [i + j/2 for i, j in zip(meanEntropy_Bitufted, stdEntropy_Bitufted)] , color='orange', alpha='0.1')
  axs[clm].plot(range(20,620,20), meanEntropy_Chandelier,'deepskyblue' , label='chandelier')
  axs[clm].fill_between(range(20,620,20), [i - j/2 for i, j in zip(meanEntropy_Chandelier, stdEntropy_Chandelier)] , [i + j/2 for i, j in zip(meanEntropy_Chandelier, stdEntropy_Chandelier)] , color='deepskyblue', alpha='0.1')
  axs[clm].plot(range(20,620,20), meanEntropy_DoubleBouquet,'lime' , label='double-bouquet')
  axs[clm].fill_between(range(20,620,20), [i - j/2 for i, j in zip(meanEntropy_DoubleBouquet, stdEntropy_DoubleBouquet)] , [i + j/2 for i, j in zip(meanEntropy_DoubleBouquet, stdEntropy_DoubleBouquet)] , color='lime', alpha='0.1')
  axs[clm].plot(range(20,620,20), meanEntropy_Neurogliaform,'navy' , label='neurogliaform')
  axs[clm].fill_between(range(20,620,20), [i - j/2 for i, j in zip(meanEntropy_Neurogliaform, stdEntropy_Neurogliaform)] , [i + j/2 for i, j in zip(meanEntropy_Neurogliaform, stdEntropy_Neurogliaform)] , color='navy', alpha='0.1')
  axs[clm].tick_params(axis='both', which='major', labelsize=16)
  axs[clm].set_ylim([0,50])
  axs[clm].set_xlim([20,600])

axs[1].set_yticklabels([],size=14)
axs[3].set_yticklabels([],size=14)
axs[0].set_xticklabels([],size=14)
axs[1].set_xticklabels([],size=14)
axs[2].set_xlabel('frequency (Hz)',fontsize=18)
axs[3].set_xlabel('frequency (Hz)',fontsize=18)
axs[0].set_ylabel('number of subtrees',fontsize=18)
axs[2].set_ylabel('number of subtrees',fontsize=18)
axs[2].legend(fontsize=16,loc='upper left')

from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_weight('bold')

axs[0].text(-45,53,'A',size=24,ha='center',va='center', fontproperties=font)
axs[1].text(-10,53,'B',size=24,ha='center',va='center', fontproperties=font)
axs[2].text(-45,53,'C',size=24,ha='center',va='center', fontproperties=font)
axs[3].text(-10,53,'D',size=24,ha='center',va='center', fontproperties=font)

axs[0].text(580,47,'cAC',size=20,ha='right',va='center')
axs[1].text(580,47,'cNAC',size=20,ha='right',va='center')
axs[2].text(580,47,'bAC',size=20,ha='right',va='center')
axs[3].text(580,47,'bNAC',size=20,ha='right',va='center')

fig.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.94, wspace=0.1, hspace=0.15)
fig.savefig('Fig_3D_num_subtrees_freq.pdf')



