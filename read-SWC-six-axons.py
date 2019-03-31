import os.path
import numpy as np
from neuron import h,gui
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import random
import seaborn as sns

def distance(x1,x2,y1,y2,z1,z2): #returns the euclidean distance between two 3D points
  dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
  return dist


def sholl(path):
  f = open(path,'r')
  ll= []
  for line in f.readlines():
    if (line[0]!="#"):
      ll.append(line.strip().split())
  f.close()
  Xs , Ys , Zs = float(ll[0][2]) , float(ll[0][3]) , float(ll[0][4])  # The cordinates of the soma
  for i in range(len(ll)):
    ll[i].append(ll[int(ll[i][6])-1][2]) # add X cordinate of the father
    ll[i].append(ll[int(ll[i][6])-1][3]) # add Y cordinate of the father
    ll[i].append(ll[int(ll[i][6])-1][4]) # add Z cordinate of the father
  
  X = []
  Y = []
  Z = []
  n_intersections100 = 0
  n_intersections200 = 0
  n_intersections300 = 0
  n_intersections400 = 0
  for i in range(len(ll)):
    if (int(ll[i][1])==2): # only axons
      Xa , Ya , Za = float(ll[i][2]) , float(ll[i][3]) , float(ll[i][4])
      Xb , Yb , Zb = float(ll[i][7]) , float(ll[i][8]) , float(ll[i][9])
      dist1 = distance(Xs,Xa,Ys,Ya,Zs,Za)
      dist2 = distance(Xs,Xb,Ys,Yb,Zs,Zb)
      if ((dist1 > 100) and (dist2 < 100)):
        n_intersections100+=1
      if ((dist1 > 200) and (dist2 < 200)):
        n_intersections200+=1
      if ((dist1 > 300) and (dist2 < 300)):
        n_intersections300+=1
      if ((dist1 > 400) and (dist2 < 400)):
        n_intersections400+=1
      X.append(Xa)
      Y.append(Ya)
      Z.append(Za)
  shll = [n_intersections100, n_intersections200, n_intersections300, n_intersections400, abs(min(X))+abs(max(X)), abs(min(Y))+abs(max(Y)), abs(min(Z))+abs(max(Z))]
  return shll


def qual (path):
  rad=[]
  num_seg = 0
  #index , typ , x , y , z , radius
  a = np.loadtxt(path)
  for i in range(len(a)):
    if (a[i][1]==2):
      rad.append(a[i][5])
      num_seg = num_seg + 1
  return [len(np.unique(rad)) , num_seg]

files_list = []
path = os.getcwd()
for root, subFolders , files in os.walk(path):
  for f in files:
    if f.endswith('.swc'):
      files_list.append('%s/%s' % (root , f))

uniq_rad = []
ax_seg_num = []
for f in files_list:
  qq = qual(f)
  uniq_rad.append(qq[0])
  ax_seg_num.append(qq[1])


typ = []
typ2 = []
shl = []
qaul_files_list = []
qual_rad = []
for i in range(len(uniq_rad)):
  if ((uniq_rad[i] > 9)&(uniq_rad[i] < 200)&(ax_seg_num[i] > 1000)):
    qaul_files_list.append(files_list[i])
    typ.append(files_list[i].split("/")[-2])
    typ2.append(files_list[i].split("/")[-2])
    shl.append(sholl(files_list[i]))
    qual_rad.append(uniq_rad[i])

from collections import Counter
Counter(typ2)
################

d = []
lng = []
GRs = []
mean_lng = []
max_lng = []
Total_lng = []
max_BO = []
mean_BO = []
max_PL = []
min_PL = []
mean_PL = []
symm = []

for i in range(len(qual_files_list)):
  print qual_files_list[i]
  h.load_file('import3d.hoc')
  Import3d = h.Import3d_SWC_read()
  
  Import3d.input(qual_files_list[i])
  
  imprt = h.Import3d_GUI(Import3d, 0)
  imprt.instantiate(None)
  print qual_files_list[i]
  num_axon_sec = 0
  for sec in h.allsec():
    if (str(sec)[0:2]=="ax"):
      num_axon_sec = num_axon_sec + 1
    
  tbl = []
  for sec in range(num_axon_sec-1):
    tbl.append([sec , h.axon[sec].L , h.axon[sec].diam ,h.SectionRef(sec=h.axon[sec]).parent])
  
  xy = [-1] # finding the father
  for sec in range(len(tbl)-1):
    xy.append(str(tbl[sec+1][3]).replace("axon[", "").replace("]","").replace("soma[0", "-1"))
  
  l = []
  for sec in range(num_axon_sec-1):
    l.append(tbl[sec][1])
  
  lng.append(l)
  mean_lng.append(np.mean(l))
  Total_lng.append(sum(l))
  if (l==[]):
    max_lng.append(l)
  else:
    max_lng.append(max(l))
  
  diam = []
  for sec in range(num_axon_sec-1):
    diam.append(tbl[sec][2])
  
  d.append(diam)
  
  diam_parent = [[0,0,0]] 
  for sec in range(num_axon_sec-2):
    diam_parent.append([sec+1 , tbl[sec+1][2] , int(xy[sec+1])])
  
  y2 = [h.axon[0].L]  # The total length of all fathers
  for sec in range(len(tbl)-1):
    y2.append(tbl[sec+1][1]+ y2[int(xy[sec+1])])
  
  direct_child = []
  for sec in range(len(tbl)):
    direct_child.append(xy.count(str(sec)))
  
  total_child = []
  for sec in range(len(tbl)-2,-1,-1):
    childrn = [i for i,x in enumerate(xy) if x==str(sec+1)]
    sum_child = direct_child[sec+1]
    for ch in childrn:
      sum_child = sum_child + total_child[len(tbl)-1-ch]
    total_child.append(sum_child)
  
  total_child.append(len(tbl)-1)
  
  sym = []
  for sec in range(len(tbl)):
    childrn = [total_child[len(tbl)-1-i] for i,x in enumerate(xy) if x==str(sec)]
    temp = [-1 , -1]
    kk=0
    for i,ch in enumerate(childrn):
      temp[kk] = ch
      kk=1
    if (sum(temp)>-1):
      sym.append(float(min(temp)+1)/(max(temp)+1) )
  
  symm.append(np.mean(sym))
  
  path_lng = []
  for k in range(len(y2)):
    if (direct_child==[]):
      path_lng.append(h.axon[0].L)
    else:
      if (direct_child[k]==0):
	path_lng.append(y2[k])
  
  max_PL.append(max(path_lng))
  min_PL.append(min(path_lng))
  mean_PL.append(np.mean(path_lng))
  
  BO = [0]
  for k in range(num_axon_sec-2):
    BO.append(BO[int(xy[k+1])]+1)
 
  max_BO.append(max(BO))
  mean_BO.append(np.mean(BO))
  
  sibling = [[]]
  for sec in range(len(tbl)-1,):
    childrn = [i for i,x in enumerate(xy) if x==str(sec+1)]
    sibling.append(childrn)
  
  brother = [0]*len(tbl)
  for G in sibling:
    if (len(G)>1): # only 1
      brother[G[0]] = G[1]
      brother[G[1]] = G[0]
  
  GR = [0]
  for sec in range(num_axon_sec-2):
    if (diam_parent[int(diam_parent[sec+1][2])][1]!=0):
      GR1 = ((diam_parent[sec+1][1]**1.5)+(diam_parent[brother[sec+1]][1]**1.5)) / (diam_parent[int(diam_parent[sec+1][2])][1]**1.5)
    else:
      GR1=0    
    GR.append(GR1)
  
  GRs.append(GR)



###

for i in range(len(GRs)):
  while 0 in GRs[i]:
    GRs[i].remove(0)


mean_GR = []
max_GR = []
mean_D = []
max_D = []
for i in range(len(GRs)):
  mean_GR.append(np.mean(GRs[i]))
  max_GR.append(max(GRs[i]))
  mean_D.append(np.mean(d[i]))
  max_D.append(max(d[i]))


len_above2_GR = []
len_above3_GR = []
len_above4_GR = []
num_branches = []
#fig = plt.figure(figsize=(10,6))
for i in range(len(GRs)):
  num_branches.append(len(GRs[i]))
  above2_GR = []
  above3_GR = []
  above4_GR = []
  for j in range(len(GRs[i])):
    if (GRs[i][j]>2.0):
      above2_GR.append(GRs[i][j])
    if (GRs[i][j]>3.0):
      above3_GR.append(GRs[i][j])
    if (GRs[i][j]>4.0):
      above4_GR.append(GRs[i][j])
  len_above2_GR.append(len(above2_GR))
  len_above3_GR.append(len(above3_GR))
  len_above4_GR.append(len(above4_GR))


len_above200_lng_d = []
len_above300_lng_d = []
len_above400_lng_d = []
mean_lng_d = []
max_lng_d = []
for i in range(len(lng)):
  above200_lng = []
  above300_lng = []
  above400_lng = []
  lng_sqrt_d = []
  for j in range(len(lng[i])):
    lng_sqrt_d.append(((lng[i][j])/(np.sqrt(d[i][j]))))
    if (((lng[i][j])/(np.sqrt(d[i][j])))>200):
      above200_lng.append((lng[i][j])/(np.sqrt(d[i][j])))
    if (((lng[i][j])/(np.sqrt(d[i][j])))>300):
      above300_lng.append((lng[i][j])/(np.sqrt(d[i][j])))
    if ((lng[i][j])/(np.sqrt(d[i][j]))>400):
      above400_lng.append((lng[i][j])/(np.sqrt(d[i][j])))
  mean_lng_d.append(np.mean(lng_sqrt_d))
  max_lng_d.append(max(lng_sqrt_d))
  len_above200_lng_d.append(len(above200_lng))
  len_above300_lng_d.append(len(above300_lng))
  len_above400_lng_d.append(len(above400_lng))


na = np.array(shl)
shl100 = list(na[:,0])
shl200 = na[:,1]
shl300 = na[:,2]
shl400 = na[:,3]
Xdepth = na[:,4]
Ydepth = na[:,5]
Zdepth = na[:,6]

from numpy import *
dd = {'01brn' : np.log10(num_branches) ,'02symm': ma.log10(symm).filled(0) , '03max_BO' : np.log10(max_BO) , '04mean_BO' : np.log10(mean_BO) , 
      '05depthX' : np.log10(Xdepth) , '06depthY' : np.log10(Ydepth) , '07depthZ' : ma.log10(Zdepth).filled(0) ,
      '08sholl100' : ma.log10(shl100).filled(0) ,'09sholl200' : ma.log10(shl200).filled(0) ,'10sholl300' : ma.log10(shl300).filled(0) ,
      '11l200' : ma.log10(len_above200_lng_d).filled(0) , '12l300' : ma.log10(len_above300_lng_d).filled(0) , '13l400' : ma.log10(len_above400_lng_d).filled(0) ,
      '14max_PL' : np.log10(max_PL) , '15min_PL' : np.log10(min_PL) , '16mean_PL' : np.log10(mean_PL) , 
      '17max_lng' : np.log10(max_lng) , '18mean_lng' : np.log10(mean_lng) , '19Total_lng' : np.log10(Total_lng) ,'20max_lng_d' : np.log10(max_lng_d) ,'21mean_lng_d' : np.log10(mean_lng_d) ,
      '22max_D' : np.log10(max_D) , '23mean_D' : np.log10(mean_D) , '24GR2': ma.log10(len_above2_GR).filled(0), '25GR3': ma.log10(len_above3_GR).filled(0),
      '26max GR' : np.log10(max_GR) , '27meanGR' : np.log10(mean_GR) , '28above2' : ma.log10([float(x)/y for x, y in zip(len_above2_GR,num_branches)]).filled(0)}


dd['name'] = typ2
df = pd.DataFrame(dd)

df.describe()

#df.corr()
#df.corr(method='spearman')
df.groupby('name').count()


file_name=[]
for k in range(len(qaul_files_list)):
  file_name.append(qaul_files_list[k].split("/")[-1].split(".")[0])

df['file']=file_name

execfile( "data_dict.py" )
file2=[]
for kk in df['file']:
  file2.append( list(neuro_dict.keys())[list(neuro_dict.values()).index(kk)] )

df['file'] = file2



# df.to_csv('axon16_six.csv')

## df=pd.read_csv('axon16_six.csv')
## df = df.drop('Unnamed: 0', 1)
