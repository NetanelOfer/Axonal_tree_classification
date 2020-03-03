#!/usr/bin/env python

import os
import neuron
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import *
#import neurom as nm
import math
import pylab
#from matplotlib import pyplot

def raster(event_times_list, cl, **kwargs):
    ax = plt.subplot2grid((11, 2), (5, 1),rowspan=6) #, sharey=ax1) #plt.gca()
    for ith, trial in enumerate(event_times_list):
      if (ith==0):
        #plt.vlines(trial, ith + .5, ith + 1.5, color='red', **kwargs)
        plt.vlines(trial, ith - 7.5, ith + -0.5, color='black', **kwargs)
      else:
	#plt.vlines(trial, ith + .5, ith + 1.5, **kwargs)
	plt.vlines(trial, ith - .5, ith + 0.5 , color=cl[ith] , **kwargs)
    #plt.ylim(.5, len(event_times_list) + .5)
    plt.ylim(-.5, len(event_times_list) - .5)
    return ax


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
    #fff = [200, 330 , 400]
    #print "step_number " , step_number
    freq = 330 # 2.8 #10 #3 #2.8 #8 # 125 Hz
    interval = round(((1000.0/freq)/dtt))*dtt
    pulses = []
    st = {}
    for i in range(900): # 1 for step
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
    #recordings['axon35'] = neuron.h.Vector()
    #recordings['axon67'] = neuron.h.Vector()
    #recordings['axon124'] = neuron.h.Vector()
    #recordings['axon106'] = neuron.h.Vector()
    #d = {}
    for i in range(num_axon_sec):  # 171
      recordings["v_vect" + str(i)] = neuron.h.Vector()
    for i in range(num_axon_sec):  # 171
      recordings["v_vect" + str(i)].record(cell.axon[i](0.5)._ref_v, 0.025)
    recordings['time'].record(neuron.h._ref_t, 0.025)
    recordings['soma(0.5)'].record(cell.soma[0](0.5)._ref_v, 0.025)
    #recordings['axon35'].record(cell.axon[35](0.5)._ref_v, 0.025)
    #recordings['axon67'].record(cell.axon[67](0.5)._ref_v, 0.025)
    #recordings['axon124'].record(cell.axon[124](0.5)._ref_v, 0.1)
    #recordings['axon106'].record(cell.axon[106](0.5)._ref_v, 0.025)
    return recordings


def run_step(step_number, plot_traces=None):
    """Run step current simulation with index step_number"""
    cell = create_cell(add_synapses=False)
    #neuron.h.topology()
    #for sec in neuron.h.allsec():
    #  neuron.h.psection(sec=sec)
    #dend1 = cell.dend[0]
    #neuron.h.psection(sec=dend1)
    #dend2 = cell.dend[60]
    #neuron.h.psection(sec=dend2)
    stimuli = create_stimuli(cell, step_number)
    recordings = create_recordings(cell)
    neuron.h.tstop = 1100 # 3000
    #print neuron.h.dt
    #print('Disabling variable timestep integration')
    neuron.h.cvode_active(0)
    #print('Running for %f ms' % neuron.h.tstop)
    neuron.h.run()
    time = np.array(recordings['time'])  # numpy
    
    #
    num_axon_sec = 0
    for sec in neuron.h.allsec():
      if (str(sec).split(".")[1][0:2]=="ax"):
        num_axon_sec = num_axon_sec + 1
    
    print "num_axon_sec = " , num_axon_sec
    #my_neuron = nm.load_neuron('morphology/BC143ax2.CNG.swc') # Pvalb-IRES-Cre_Ai14-236447-02-01-01_543103327_m.CNG.swc') #
    #num_axon_sec = len(nm.get('section_lengths',my_neuron, neurite_type=nm.AXON))
    
    diam = []
    for j in range(num_axon_sec):   # 111  # range(i)
      diam.append(cell.axon[j].diam)
    
    tbl = []
    for sec in range(num_axon_sec):
       tbl.append([cell.axon[sec] , cell.axon[sec].L , neuron.h.SectionRef(sec=cell.axon[sec]).parent])
    
    xy = [-1] #[str(tbl[0][0]).replace("axon[", "").replace("]","")]  # finding the father
    for sec in range(len(tbl)-1):
      xy.append(str(tbl[sec+1][2]).replace("cNAC187_L23_LBC_d3f79b893e[0].axon[", "").replace("]",""))

    y2 = [cell.axon[0].L] #tbl[0][1]]  # The total length of all fathers (include the current section)! (correction for only one branch)
    for sec in range(len(tbl)-1):
      y2.append(tbl[sec+1][1]+ y2[int(xy[sec+1])])
    
    #y2 = [tbl[0][1]]  # The total length of all fathers (include the current section)!
    #for sec in range(len(tbl)-1):
    #  y2.append(tbl[sec+1][1]+ y2[int(xy[sec+1])])

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


    two_childrn = []  # the number of children of each child
    for sec in range(len(tbl)):
      childrn = [total_child[len(tbl)-1-i] for i,x in enumerate(xy) if x==str(sec)]
      temp = [-1 , -1]
      for i,ch in enumerate(childrn):
	temp[i] = ch
      two_childrn.append(temp)

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

    #############
    sp = 11
    xlimbegin = 1000 # 50 #75
    xlimend = 1100 # 130
    fig = plt.figure(figsize=(8,9)) # 8,10
    #ax1 = fig.add_subplot(5,1,1)
    #ax1_plot = ax1.plot(recordings['time'], recordings['soma(0.5)'], color='black')
    #ax1.set_xticks([]) # Use ax2's tick labels
    #ax1.set_xlim([xlimbegin,xlimend])
    #ax1.set_ylim([-90,60])
    ax2 = plt.subplot2grid((sp, 2), (5, 1),rowspan=sp-5)#, sharey=ax1)
    ax3 = plt.subplot2grid((sp, 2), (0, 1))
    ax4 = plt.subplot2grid((sp, 2), (1, 1))
    ax5 = plt.subplot2grid((sp, 2), (2, 1))
    ax6 = plt.subplot2grid((sp, 2), (3, 1))
    ax7 = plt.subplot2grid((sp, 2), (0, 0),rowspan=4) # neuronal reconstruction
    #ax_empty1 = plt.subplot2grid((sp, 2), (4, 0))
    #ax_empty2 = plt.subplot2grid((sp, 2), (4, 1))
    #ax2 = fig.add_subplot(4,1,1) #(5,1,2)
    ax7.spines['right'].set_visible(False)
    ax7.spines['left'].set_visible(False)
    ax7.spines['top'].set_visible(False)
    ax7.spines['bottom'].set_visible(False)
    ax7.set_yticks([])
    ax7.set_xticks([])
    ax3.plot(recordings['time'],  recordings['v_vect0'] , color='maroon')
    ax3.set_xticks([]) # Use ax2's tick labels
    ax3.set_xlim([xlimbegin,xlimend])
    ax3.set_ylim([-90,60])
    plt.yticks(fontsize=14)
    #ax3 = fig.add_subplot(4,1,2) #(5,1,3)
    ax4.plot(recordings['time'], recordings['v_vect142'] , color='violet') # 142 instead of 161
    ax4.set_ylabel(' ')
    ax4.set_xticks([]) # Use ax2's tick labels
    ax4.set_xlim([xlimbegin,xlimend])
    ax4.set_ylim([-90,60])
    plt.yticks(fontsize=14)
    #ax4 = fig.add_subplot(4,1,3) #(5,1,4)
    ax5.plot(recordings['time'], recordings['v_vect79'] , color='orange') # 161
    ax5.set_xticks([]) # Use ax2's tick labels
    ax5.set_xlim([xlimbegin,xlimend])
    ax5.set_ylim([-90,60])
    plt.yticks(fontsize=14)
    #ax5 = fig.add_subplot(4,1,4) #(5,1,5)
    ax6_plot = ax6.plot(recordings['time'], recordings['v_vect16'], color='green') # 16 instead of 18
    #ax3.set_ylabel('mV')
    ax6.set_xlabel('time (ms)',size=14)
    ax6.set_xlim([xlimbegin,xlimend])
    ax6.set_ylim([-90,60])
    plt.xticks(fontsize=14) #, rotation=90)
    plt.yticks(fontsize=14)
    #fig.tight_layout()
    #fig.subplots_adjust(hspace=0.1 ,left=0.115)
    #fig.savefig('/home/userlab/neuron/figures/subplots-temp.pdf') #spikes_L23_LBC_cNAC187_5_BC143ax2-b.CNG.swc.pdf')

    #plt.show()
    #print pos[0] # 79
    #print pos[161] # 156
    #print pos[79] # 78
    #print pos[18] # 14
    #print new_pos[134] # 142
    #print new_pos[12] # 16
    
    path = '/home/netanel/Dropbox/neuron/L23_LBC_cNAC187_5/morphology/BC143ax2.CNG.swc'
    f = open(path,'r') # Bipolar Tolias
    
    ll= []
    for line in f.readlines():
      if (line[0]!="#"):
        ll.append(line.strip().split())
    
    f.close()
    
    for i in range(len(ll)):
      ll[i].append(ll[int(ll[i][6])-1][2]) # add X cordinate of the father
      ll[i].append(ll[int(ll[i][6])-1][3]) # add Y cordinate of the father
    
    #fig, ax = plt.subplots(figsize=(6/prop(path),6)) # for Cauli Basket
    clr = ['yellow','green','blue', 'red','orangered']
    for i in range(len(ll)):
      if (int(ll[i][1])>1):
        ax7.plot([float(ll[i][3]), float(ll[i][8])],[-(float(ll[i][2])), -float(ll[i][7])], c=clr[int(ll[i][1])]  , lw = float(ll[i][5])*6)  # For Cauli Basket
    
    
    ax7.annotate('1', xy=(-300, 462), xytext=(-200, 490), arrowprops=dict(facecolor='violet', shrink=0.01) )
    ax7.annotate('2', xy=(-180, -250), xytext=(-280, -220), arrowprops=dict(facecolor='orange', shrink=0.01) )
    ax7.annotate('3', xy=(330, 290), xytext=(230, 350), arrowprops=dict(facecolor='green', shrink=0.01) )
    ax7.annotate('stimulus', xy=(0, 0), xytext=(-400, -100), arrowprops=dict(facecolor='black', shrink=0.01),size=14 )
    ax7.annotate('0', xy=(0, 0), xytext=(100, -80), arrowprops=dict(facecolor='maroon', shrink=0.01) )

    #ax.axis('off')
    #############
    #fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6.4,4.8)) # , figsize=(8,6)) # width , high    6.4,4.8
    #ax1 = plt.subplot2grid((sp, 2), (4, 0),rowspan=8)
    #ax2 = plt.subplot2grid((sp, 2), (4, 1),rowspan=8, sharey=ax1)
    aa = []
    aaa = []
    num_spikes = []
    aa.append(array(stimuli[0]['stimulus']))
    aaa.append(array(stimuli[0]['stimulus']))
    #print "stimuli length = " , len(array(stimuli[0]['stimulus']))
    #print "stimuli= " , array(stimuli[0]['stimulus'])
    #print "dtt " , neuron.h.dt
    j2 = [i for i in stimuli[0]['stimulus'] if ((i >= (500/neuron.h.dt))&(i<(neuron.h.tstop/neuron.h.dt)))] # 2000 , 5200
    #print len(j2)
    #num_spikes.append(len(j2))
    for i in new_pos: #range(num_axon_sec2):
      v_value = []
      for j in range(int(neuron.h.tstop/neuron.h.dt + 0)): # 4001   801
        v_value.append(recordings["v_vect"+str(i)][j]) # v_value.append(v_vec[j])
      c = (diff(sign(diff(v_value))) < 0).nonzero()[0] + 1 # local max
      
      cc = [] # only local max above zero
      for k in range(len(c)):
        if v_value[c[k]] > 0: # -60
          cc.append(c[k])
      
      v_max = [cc[0]]
      for iii in range(len(cc)-1):
	if (cc[iii+1] > cc[iii]+40): # discard close peaks
	  v_max.append(cc[iii+1])
      
      
      #cc = [c[0]]
      #for iii in range(len(c)-1):
	#if (c[iii+1] > c[iii]+40): # discard close peaks
	  #cc.append(c[iii+1])
      
      ##print cc , i
      #v_max = [] # only local max above zero  OR 30 ??
      #for k in range(len(cc)):
        #if v_value[cc[k]] > 0: # -60
          #v_max.append(cc[k])
      aa.append(array(v_max))
      j3 = [ii for ii in v_max if ((ii >= (500/neuron.h.dt))&(ii<(neuron.h.tstop/neuron.h.dt)))] # 2000 , 5200
      aaa.append(np.array(j3) - j3[0] + 500/neuron.h.dt) # alignment
      #print pos[i] , len(j3) , float(len(j3)) / len(j2)
      num_spikes.append(len(j3))
    
    const = [float(x) / len(j2) for x in num_spikes]
    #print "len const " ,len(const)
    #print "num axon sec " , num_axon_sec
    
    cll = ['maroon' ,'green', 'orange' , 'deepskyblue' , 'violet' , 'navy' , 'dimgray' , 'silver'] #'black' , 'gray'] # green instead of yellow
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
    
    
    aa = np.multiply(aa,neuron.h.dt) # for the alignment change to 'aaa'
    P = []
    for pp in cll:
      P.append(float(cl.count(pp))/(len(cl)-1))
    
    print "P=" , P
    P = [y for y in P if y != 0]
    #print "P=" , P
    q = 2
    print "M2=" , sum([n**q for n in P])**(1/(1-q))
    q=0.9999
    print "M0.9999=" , sum([n**q for n in P])**(1/(1-q))
    import math
    print "Entropy2=" , math.exp(1)**-(sum([n*np.log(n) for n in P]))
    print "percentage of branches with interrupted" , 1 - (float(cl.count('maroon'))/(len(cl)-1))
    ax2 = raster(aa,cl)
    ax2.set_xlabel('time (ms)',size=14)
    ax2.set_xlim([1000,neuron.h.tstop]) # 50
    
    ax1 = plt.subplot2grid((sp, 2), (5, 0),rowspan=sp-5, sharey=ax2)
    
    for i in range(num_axon_sec): # plot the dendrogram (axogram)
      ax1.plot([y2[i]-tbl[i][1]-0.3,y2[i]],[pos[i]+1,pos[i]+1],color=cll[int(clr[int(pos[i])])],linewidth=diam[i]*3 , solid_capstyle='butt') # *3
      #ax1.plot([y2[i]-tbl[i][1]-0.3,y2[i]],[pos[i]+1,pos[i]+1],color='b',linewidth=diam[i]*2, solid_capstyle='butt')
      if (i>0):
	ax1.plot([y2[i]-tbl[i][1] , y2[i]-tbl[i][1]] , [pos[i]+1,pos[int(xy[i])]+1],color=cll[int(clr[int(pos[i])])],linewidth= 0.1, solid_capstyle='butt') # diam[i]*5)
	#ax1.plot([y2[i]-tbl[i][1] , y2[i]-tbl[i][1]] , [pos[i]+1,pos[int(xy[i])]+1],color='b',linewidth= 0.1, solid_capstyle='butt')

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax1.set_xlabel('length ($\mu$m)',size=14)
    #ax1.set_ylim([-1,len(tbl)-0])  # ax1.set_ylim([-1,len(tbl)-0.8])  #  shared y axis
    ax1.set_xlim([0, max(y2)]) #math.ceil(max(y2)/100.0)*100])  #  +20
    ax1.set_ylim([-7.5,271.5]) 
    ax2.set_ylim([-7.5,271.5]) 
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    #ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
    
    ax1.text(-15,79,'0', fontsize=10,ha='center',va='center') # print '0' left to the main branch in axogram
    #ax1.text(y2[161]+50,156,'1', fontsize=10,ha='center',va='center')
    ax1.text(y2[142]+15,134,'1', fontsize=10,ha='center',va='center')
    ax1.text(y2[79]+15,78,'2', fontsize=10,ha='center',va='center')
    ax1.text(y2[16]+15,12,'3', fontsize=10,ha='center',va='center')
    #print y2[161] # 156
    #print y2[142]
    #print y2[79] # 78
    #print y2[18] # 14
    
    ax1.text(-40,500,'A', fontsize=15)
    ax1.text(880,500,'B', fontsize=15)
    
    ax1.text(1900,493,'0', fontsize=12) # the number right to train plot (subplot B)
    ax1.text(1900,446,'1', fontsize=12)
    ax1.text(1900,401,'2', fontsize=12)
    ax1.text(1900,355,'3', fontsize=12)
    
    ax1.text(-40,280,'C', fontsize=15)
    ax1.text(880,280,'D', fontsize=15)
    fig.text(0.46, 0.82, 'mV', va='center', rotation='vertical',size=14)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1,wspace=0.15) # ,left=0.115)
    #fig.subplots_adjust(wspace=0.1) #, bottom=0.09)
    fig.savefig('/home/netanel/Dropbox/neuron/cNAC_PIC2.pdf')
    #fig.savefig('/home/userlab/neuron/figures/cNAC_PIC.pdf') 
    #fig.savefig('/home/userlab/neuron/figures/L23_LBC_cNAC187_5-full_BC143ax2.pdf') # -alignment3
    #fig.savefig('/home/userlab/neuron/figures/Pyramidal-cNAC.pdf')
    #fig.show()




def init_simulation():
    """Initialise simulation environment"""

    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    print('Loading constants')
    neuron.h.load_file('constants.hoc')


def main(plot_traces=True):
    """Main"""

    # Import matplotlib to plot the traces
    if plot_traces:
        import matplotlib
        matplotlib.rcParams['path.simplify'] = False

    init_simulation()

    for step_number in range(1, 2):
        run_step(step_number, plot_traces=plot_traces)

    if plot_traces:
        import pylab
        #pylab.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(plot_traces=True)
    elif len(sys.argv) == 2 and sys.argv[1] == '--no-plots':
        main(plot_traces=False)
    else:
        raise Exception(
            "Script only accepts one argument: --no-plots, not %s" %
            str(sys.argv))
