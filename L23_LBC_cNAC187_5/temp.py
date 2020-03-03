import os
import neuron
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import *
import neurom as nm
import math
import pylab
from matplotlib import pyplot 

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")
neuron.h.load_file('constants.hoc')

def create_cell(add_synapses=True):
    neuron.h.load_file("morphology.hoc")
    neuron.h.load_file("biophysics.hoc")
    neuron.h.load_file("template.hoc")
    #print("Loading cell cNAC187_L23_LBC_d3f79b893e")
    cell = neuron.h.cNAC187_L23_LBC_d3f79b893e(1 if add_synapses else 0)
    return cell


def create_stimuli(cell, step_number):
    """Create the stimuli"""
    stimuli = []
    dtt = neuron.h.dt #0.025
    #fff = range(100,740,20) #[200, 330 , 400]
    #print "step_number " , step_number
    #print "freq=" , fff[step_number]
    freq = 300 # 340 #fff[step_number] #760 # 2.8 #10 #3 #2.8 #8 # 125 Hz
    interval = round(((1000.0/freq)/dtt))*dtt
    pulses = []
    st = {}
    for i in range(600): # 1 for step
      st["stim" + str(i)] = neuron.h.IClamp(0.5, sec=cell.soma[0])
      st["stim" + str(i)].delay = interval*i + 1 
      st["stim" + str(i)].amp = 20 #100 #20
      st["stim" + str(i)].dur = 1 # 130 for step
      pulses.append((interval*i + 1)/dtt)
    st["stimulus"] = pulses
    stimuli.append(st)
    #stimuli.append(st['stim1'])
    return stimuli


def create_recordings(cell):
    """Create the recordings"""
    num_axon_sec = 0
    for sec in neuron.h.allsec():
      #print str(sec).split(".")[1]
      if (str(sec).split(".")[1][0:2]=="ax"):
        num_axon_sec = num_axon_sec + 1
    
    #my_neuron = nm.load_neuron('morphology/BC143ax2.CNG.swc') #Pvalb-IRES-Cre_Ai14-236447-02-01-01_543103327_m.CNG.swc') #
    #num_axon_sec = len(nm.get('section_lengths',my_neuron, neurite_type=nm.AXON))
    recordings = {}
    recordings['time'] = neuron.h.Vector()
    recordings['soma(0.5)'] = neuron.h.Vector()
    for i in range(num_axon_sec):  # 171
      recordings["v_vect" + str(i)] = neuron.h.Vector()
    for i in range(num_axon_sec):  # 171
      recordings["v_vect" + str(i)].record(cell.axon[i](0.5)._ref_v, 0.025)
    recordings['time'].record(neuron.h._ref_t, 0.025)
    recordings['soma(0.5)'].record(cell.soma[0](0.5)._ref_v, 0.025)
    return recordings

cell = create_cell(add_synapses=False)
stimuli = create_stimuli(cell, 1) #step_number)
recordings = create_recordings(cell)
neuron.h.tstop = 1100 # 3000
neuron.h.cvode_active(0)
neuron.h.run()
time = np.array(recordings['time'])
#

num_axon_sec = 0
for sec in neuron.h.allsec():
  if (str(sec).split(".")[1][0:2]=="ax"):
    num_axon_sec = num_axon_sec + 1

diam = []
for j in range(num_axon_sec):   # 111  # range(i)
  diam.append(cell.axon[j].diam)

tbl = []
for sec in range(num_axon_sec):
  tbl.append([cell.axon[sec] , cell.axon[sec].L , neuron.h.SectionRef(sec=cell.axon[sec]).parent])

xy = ['-1'] #[str(tbl[0][0]).replace("axon[", "").replace("]","")]  # finding the father
for sec in range(len(tbl)-1):
  #print "aaa" , str(tbl[sec+1][2])
  nn = str(tbl[sec+1][2])[-5:-1].replace("[","").replace("n","").replace("o","")
  #print nn
  xy.append(nn) #str(tbl[sec+1][2]).replace("cNAC187_L23_LBC_d3f79b893e[0].axon[", "").replace("]",""))

y2 = [cell.axon[0].L] #tbl[0][1]]  # The total length of all fathers (include the current section)! (correction for only one branch)
for sec in range(len(tbl)-1):
  y2.append(tbl[sec+1][1]+ y2[int(xy[sec+1])])


direct_child = []  # 
for sec in range(len(tbl)):
  direct_child.append(xy.count(str(sec)))

total_child = []
for sec in range(len(tbl)-2,-1,-1):
  childrn = [i for i,x in enumerate(xy) if x==str(sec+1)]
  sum_child = direct_child[sec+1]
  for ch in childrn:
    #print sec+1 , ch , len(tbl)-1-ch , total_child[len(tbl)-1-ch]
    sum_child = sum_child + total_child[len(tbl)-1-ch] #
  total_child.append(sum_child)

total_child.append(len(tbl)-1)

sym = []
two_childrn = []  # the number of children of each child
for sec in range(len(tbl)):
  childrn = [total_child[len(tbl)-1-i] for i,x in enumerate(xy) if x==str(sec)]
  temp = [-1 , -1]
  for i,ch in enumerate(childrn):
    temp[i] = ch
  two_childrn.append(temp)
  if (sum(temp)>0):
    sym.append(float(min(temp)+1)/(max(temp)+1) )

print mean(sym)


dirc = []
zrs = [0]*len(tbl)
for sec in range(len(tbl)):
  if (zrs[int(xy[int(sec)])]==0):
    zrs[int(xy[int(sec)])] = 1
    dirc.append(0)
  else:
    dirc.append(1)

pos = [two_childrn[0][0]+1]
for sec in range(len(tbl)-1):
  pp = pos[int(xy[sec+1])]
  pos.append(pp - (two_childrn[sec+1][1]+2)*(dirc[sec+1]==0) + (two_childrn[sec+1][0]+2)*(dirc[sec+1]!=0) ) #+ (two_childrn[sec+1][1]==0)*(dirc[sec+1]==0) - (two_childrn[sec+1][0]==0)*(dirc[sec+1]!=0) )

new_pos = []
for i in range(len(tbl)):
  k = 0
  for j in pos:
    k = k + 1
    if i == j:
      new_pos.append(k-1)

#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6.4,4.8)) # , figsize=(8,6)) # width , high    6.4,4.8
aa = []
num_spikes = []
aa.append(array(stimuli[0]['stimulus']))
j2 = [i for i in stimuli[0]['stimulus'] if ((i >= (500/neuron.h.dt))&(i<(neuron.h.tstop/neuron.h.dt)))] # 2000 , 5200
for i in new_pos: #range(num_axon_sec2):
  v_value = []
  for j in range(int(neuron.h.tstop/neuron.h.dt + 0)): # 4001   801
    v_value.append(recordings["v_vect"+str(i)][j]) # v_value.append(v_vec[j])
  c = (diff(sign(diff(v_value))) < 0).nonzero()[0] + 1 # local max
  cc = [c[0]]
  for iii in range(len(c)-1):
    if (c[iii+1] > c[iii]+40): # discard close peaks
      cc.append(c[iii+1])
  
  v_max = [] # only local max above zero  OR 30 ??
  for k in range(len(cc)):
    if v_value[cc[k]] > 0: # -60
      v_max.append(cc[k])
  aa.append(array(v_max))
  j3 = [ii for ii in v_max if ((ii >= (500/neuron.h.dt))&(ii<(neuron.h.tstop/neuron.h.dt)))] # 2000 , 5200
  num_spikes.append(len(j3))

const = [float(x) / len(j2) for x in num_spikes]

cll = ['red' ,'green', 'orange' , 'deepskyblue' , 'blue' , 'navy' , 'dimgray' , 'silver'] #'black' , 'gray'] # green instead of yellow
fr = [1 ,.8, 0.75 , 0.66 , 0.5 , 0.375 ,0.2 , 0]

M = np.zeros((len(const), len(fr)))
for i in range(len(const)):
  for k in range(len(fr)):
    M[i][k] = abs(const[i] - fr[k])

clr = []
for i in range(len(const)):
  clr.append(np.argmin(M[i]))   

cl=['z'] # choose color for the raster plot
for i in new_pos:
  cl.append(cll[int(clr[int(pos[i])])])

################################## NEW #########################

#par=[] # the number of parent , the order is the new_pos order 
#for ii in new_pos:
#  if ((cll[int(clr[int(pos[ii])])])!=(cll[int(clr[int(pos[int(xy[ii])])])])):
#    par.append(xy[ii])

for ii in new_pos:
  print ii , "\t" , cll[int(clr[int(pos[ii])])] , "\t" , xy[ii] ,  "\t" ,cll[int(clr[int(pos[int(xy[ii])])])] ,"\t" , ((cll[int(clr[int(pos[ii])])])==(cll[int(clr[int(pos[int(xy[ii])])])])) ,"\t" , total_child[len(tbl)-1-ii] #, "\t" , f 


direct_same_child = [0]*len(new_pos)
son1_color=['zzz']*len(new_pos)
son2_color=['zzz']*len(new_pos)
for ii in new_pos:
  cn=0
  k=-1
  for jj in xy:
    k=k+1
    if ((ii==int(jj))): #&& ((cll[int(clr[int(pos[ii])])])== )):
      cn = cn+1
      if (son1_color[ii]!='zzz'):
	son2_color[ii] = cll[int(clr[int(pos[k])])]
      if (son1_color[ii]=='zzz'):
	son1_color[ii] = cll[int(clr[int(pos[k])])]
  #print ii, cn , direct_child[ii] , cll[int(clr[int(pos[ii])])] , son1_color[ii], son2_color[ii]
  #direct_same_child.append((cll[int(clr[int(pos[ii])])]==son1_color[ii])+(cll[int(clr[int(pos[ii])])]==son2_color[ii]))
  direct_same_child[ii] = ((cll[int(clr[int(pos[ii])])]==son1_color[ii])+(cll[int(clr[int(pos[ii])])]==son2_color[ii]))
  print ii, cn , direct_child[ii] , cll[int(clr[int(pos[ii])])] , son1_color[ii], son2_color[ii] , direct_same_child[ii]


total_same_child = []
#for sec in range(len(tbl)-2,-1,-1):
for sec in range(len(tbl)-2,-2,-1):
  childrn = [i for i,x in enumerate(xy) if x==str(sec+1)]
  #childrn
  #print sec+1 , cll[int(clr[int(pos[sec+1])])] # the father color
  #print "children:" , 
  child_colr = [cll[int(clr[int(pos[i])])] for i,x in enumerate(xy) if x==str(sec+1)] ### color of children
  sum_child = direct_same_child[sec+1]
  for ch in childrn:
    if (cll[int(clr[int(pos[ch])])] ==cll[int(clr[int(pos[sec+1])])]):
      sum_child = sum_child + total_same_child[len(tbl)-1-ch] #
  total_same_child.append(sum_child)

#total_same_child.append(len(tbl)-1)
total_same_child
print "sum(total_same_child):",sum(total_same_child)


#######################



par=[] # the number of parent , the order is the new_pos order 
t_c=[] # total children
stc=[] # sub-tree color
for ii in new_pos:
  if (ii!=0):
    if ((cll[int(clr[int(pos[ii])])])!=(cll[int(clr[int(pos[int(xy[ii])])])])): # if the father color is different
      print xy[ii] , "\t" , total_same_child[len(tbl)-1-ii]
      par.append(xy[ii])
      t_c.append(total_same_child[len(tbl)-1-ii])
      stc.append(cll[int(clr[int(pos[ii])])])
  if (ii==0):
    print "root" , xy[ii] , "\t" , total_same_child[len(tbl)-1-ii]
    par.append(xy[ii])
    t_c.append(total_same_child[len(tbl)-1-ii]-0)
    stc.append(cll[int(clr[int(pos[ii])])])

print "par",par
print "t_c" , t_c
print stc

#kk=-1
#En=[0]*(len(np.unique(par))) #-1)
#for ii in np.unique(par): #[1:]:
#  kk=kk+1
#  ipar = [i for i,x in enumerate(par) if x == ii]
#  print ipar
#  En[kk] = sum([t_c[int(i)]+1 for i in ipar]) #+2


kk=-1
En=[0]*(len(np.unique(par)))
Enc=[0]*(len(np.unique(par)))
for ii in np.unique(par):
  kk=kk+1
  if (par.count(ii)==1):
    ipar = [i for i,x in enumerate(par) if x == ii]
    En[kk] = t_c[ipar[0]]+1
    Enc[kk] = stc[ipar[0]]
  if ((par.count(ii)==2)&(son1_color[int(ii)]==son2_color[int(ii)])):
    ipar = [i for i,x in enumerate(par) if x == ii]
    En[kk] = sum([t_c[int(i)]+1 for i in ipar])
    Enc[kk] = stc[ipar[0]]
  if ((par.count(ii)==2)&(son1_color[int(ii)]!=son2_color[int(ii)])):
    ipar = [i for i,x in enumerate(par) if x == ii]
    #print kk , ipar[0]
    En[kk] = t_c[ipar[0]]+1
    En.append(t_c[ipar[1]]+1)
    Enc[kk] = stc[ipar[0]]
    Enc.append(stc[ipar[1]])


#if (En==[]):
#  En=[len(new_pos)]

print En
print "sum En:" , sum(En)
print "lem En:" , len(En) ,"len Enc:" , len(Enc)
