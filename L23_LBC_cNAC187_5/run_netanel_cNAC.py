#!/usr/bin/env python

"""Python script to run cell model"""


"""
/* Copyright (c) 2015 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This work is licensed under a
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode or send a letter to
Creative Commons, 171 Second Street, Suite 300,
San Francisco, California, 94105, USA.
"""

"""
 * @file run.py
 * @brief Run simulation using pyneuron
 * @author Werner Van Geit @ BBP
 * @date 2015
"""

# pylint: disable=C0325, W0212, F0401, W0612, F0401

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

def raster(event_times_list, **kwargs):
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
      if (ith==0):
        #plt.vlines(trial, ith + .5, ith + 1.5, color='red', **kwargs)
        plt.vlines(trial, ith - .5, ith + 0.5, color='red', **kwargs)
      else:
	#plt.vlines(trial, ith + .5, ith + 1.5, **kwargs)
	plt.vlines(trial, ith - .5, ith + 0.5, **kwargs)
    #plt.ylim(.5, len(event_times_list) + .5)
    plt.ylim(-.5, len(event_times_list) - .5)
    return ax


def create_cell(add_synapses=True):
    """Create the cell model"""
    # Load morphology
    neuron.h.load_file("morphology.hoc")
    # Load biophysics
    neuron.h.load_file("biophysics.hoc")
    # Load main cell template
    neuron.h.load_file("template.hoc")

    # Instantiate the cell from the template

    print("Loading cell cNAC187_L23_LBC_d3f79b893e")
    cell = neuron.h.cNAC187_L23_LBC_d3f79b893e(1 if add_synapses else 0)
    return cell


def create_stimuli(cell, step_number):
    """Create the stimuli"""
    stimuli = []
    iclamp = neuron.h.IClamp(0.5, sec=cell.soma[0])
    iclamp.delay = 10 #700
    iclamp.dur = 20 #00
    iclamp.amp = 0.2 #float(step_amp[step_number - 1])
    #stimuli.append(iclamp)
    dtt = neuron.h.dt #0.025
    freq = 2.8 #8 # 125 Hz
    pulses = []
    st = {}
    for i in range(60): # 30
      st["stim" + str(i)] = neuron.h.IClamp(0.5, sec=cell.soma[0])
      st["stim" + str(i)].delay = freq*i + 1 
      st["stim" + str(i)].amp = 20
      st["stim" + str(i)].dur = 1
      pulses.append((freq*i + 1)/dtt)
    st["stimulus"] = pulses
    stimuli.append(st)
    #stimuli.append(st['stim1'])
    return stimuli


def create_recordings(cell):
    """Create the recordings"""
    my_neuron = nm.load_neuron('morphology/C230300D1_-_Clone_5.asc')
    num_axon_sec = len(nm.get('section_lengths',my_neuron, neurite_type=nm.AXON))
    print('Attaching recording electrodes')
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
    neuron.h.tstop = 130 # 3000
    #print neuron.h.dt
    print('Disabling variable timestep integration')
    neuron.h.cvode_active(0)
    print('Running for %f ms' % neuron.h.tstop)
    neuron.h.run()
    time = np.array(recordings['time'])  # numpy
    
    #
    my_neuron = nm.load_neuron('morphology/C230300D1_-_Clone_5.asc')
    num_axon_sec = len(nm.get('section_lengths',my_neuron, neurite_type=nm.AXON))
    
    diam = []
    for j in range(num_axon_sec):   # 111  # range(i)
      diam.append(cell.axon[j].diam)
    
    tbl = []
    for sec in range(num_axon_sec):
       tbl.append([cell.axon[sec] , cell.axon[sec].L , neuron.h.SectionRef(sec=cell.axon[sec]).parent])
    
    xy = [-1] #[str(tbl[0][0]).replace("axon[", "").replace("]","")]  # finding the father
    for sec in range(len(tbl)-1):
      xy.append(str(tbl[sec+1][2]).replace("cNAC187_L23_LBC_d3f79b893e[0].axon[", "").replace("]",""))

    y2 = [tbl[0][1]]  # The total length of all fathers (include the current section)!
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
    xlimbegin = 50 #75
    xlimend = 110
    fig = pyplot.figure(figsize=(8,6)) # 8,6
    #ax1 = fig.add_subplot(5,1,1)
    #ax1_plot = ax1.plot(recordings['time'], recordings['soma(0.5)'], color='black')
    #ax1.set_xticks([]) # Use ax2's tick labels
    #ax1.set_xlim([xlimbegin,xlimend])
    #ax1.set_ylim([-90,60])
    ax2 = fig.add_subplot(4,1,1) #(5,1,2)
    ax2.plot(recordings['time'],  recordings['v_vect0'] , color='red')
    ax2.set_xticks([]) # Use ax2's tick labels
    ax2.set_xlim([xlimbegin,xlimend])
    ax2.set_ylim([-90,60])
    plt.yticks(fontsize=14)
    ax3 = fig.add_subplot(4,1,2) #(5,1,3)
    ax3.plot(recordings['time'], recordings['v_vect18'] , color='blue') # 35
    ax3.set_ylabel(' ')
    ax3.set_xticks([]) # Use ax2's tick labels
    ax3.set_xlim([xlimbegin,xlimend])
    ax3.set_ylim([-90,60])
    plt.yticks(fontsize=14)
    ax4 = fig.add_subplot(4,1,3) #(5,1,4)
    ax4.plot(recordings['time'], recordings['v_vect161'] , color='green')
    ax4.set_xticks([]) # Use ax2's tick labels
    ax4.set_xlim([xlimbegin,xlimend])
    ax4.set_ylim([-90,60])
    plt.yticks(fontsize=14)
    ax5 = fig.add_subplot(4,1,4) #(5,1,5)
    ax5_plot = ax5.plot(recordings['time'], recordings['v_vect67'], color='orange')
    #ax3.set_ylabel('mV')
    ax5.set_xlabel('time (ms)',size=16)
    ax5.set_xlim([xlimbegin,xlimend])
    ax5.set_ylim([-90,60])
    plt.xticks(fontsize=14) #, rotation=90)
    plt.yticks(fontsize=14)
    fig.text(0.01, 0.55, 'mV', va='center', rotation='vertical',size=16)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1 ,left=0.115)
    fig.savefig('/home/userlab/neuron/figures/spikes_L23_LBC_cNAC187_5.pdf')
    plt.show()
    #print new_pos[111]   # plot '106'
    #print new_pos[60]   # plot '67'
    #print new_pos[30]   # plot '35'
    #print new_pos[160] # green '2:1' - 161
    #print new_pos[2] # 18
    
    #############
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6.4,4.8)) # , figsize=(8,6)) # width , high    6.4,4.8
    aa = []
    num_spikes = []
    aa.append(array(stimuli[0]['stimulus']))
    #print "stimuli length = " , len(array(stimuli[0]['stimulus']))
    #print "stimuli= " , array(stimuli[0]['stimulus'])
    #print "dtt " , neuron.h.dt
    j2 = [i for i in stimuli[0]['stimulus'] if ((i >= (50/neuron.h.dt))&(i<(neuron.h.tstop/neuron.h.dt)))] # 2000 , 5200
    #print len(j2)
    #num_spikes.append(len(j2))
    for i in new_pos: #range(num_axon_sec2):
      v_value = []
      for j in range(int(neuron.h.tstop/neuron.h.dt + 1)): # 4001   801
        v_value.append(recordings["v_vect"+str(i)][j]) # v_value.append(v_vec[j])
      c = (diff(sign(diff(v_value))) < 0).nonzero()[0] + 1 # local max
      cc = [c[0]]
      for iii in range(len(c)-1):
	if (c[iii+1] > c[iii]+20): # discard close peaks
	  cc.append(c[iii+1])
      
      #print cc , i
      v_max = [] # only local max above zero  OR 30 ??
      for k in range(len(cc)):
        if v_value[cc[k]] > 0: # 0 c
          v_max.append(cc[k])
      aa.append(array(v_max))
      j3 = [ii for ii in v_max if ((ii >= (50/neuron.h.dt))&(ii<(neuron.h.tstop/neuron.h.dt)))] # 2000 , 5200
      #print pos[i] , len(j3) , float(len(j3)) / len(j2)
      num_spikes.append(len(j3))
    
    const = [float(x) / len(j2) for x in num_spikes]
    #print "len const " ,len(const)
    #print "num axon sec " , num_axon_sec
    aa = np.multiply(aa,neuron.h.dt)
    #fig = plt.figure(figsize=(8,6))
    ax2 = raster(aa)
    ax2.set_xlabel('time (ms)')
    #ax2.plt.ylabel('axonal segment')
    ax2.set_xlim([50,neuron.h.tstop])
    
    cll = ['red' , 'orange' , 'green' , 'blue' , 'yellow' , 'black']
    fr = [1 , 0.75 , 0.66 , 0.5 , 0.2 , 0]
    M = np.zeros((len(const), len(fr)))
    for i in range(len(const)):
      for k in range(len(fr)):
	M[i][k] = abs(const[i] - fr[k])
    
    clr = []
    for i in range(len(const)):
      #print np.argmin(M[i])
      clr.append(np.argmin(M[i]))
      #print divmod(M.argmin(), M.shape[1])[1]
    
    #print "len clr " , len(clr)
    #print "len new pos " ,len(new_pos)
    #for i in reversed(new_pos):
      #print pos[i]+1 , num_spikes[i] , clr[int(pos[i])]
    
    for i in range(num_axon_sec): # plot the dendrogram (axogram)
      #ax1.plot([y2[i]-tbl[i][1]+(diam[i]*7),y2[i]-(diam[i]*7)],[pos[i]+1,pos[i]+1],color=cll[int(clr[int(pos[i])])],linewidth=diam[i]*3, solid_capstyle='butt') # cll[int(clr[i])]
      ax1.plot([y2[i]-tbl[i][1]-0.3,y2[i]],[pos[i]+1,pos[i]+1],color=cll[int(clr[int(pos[i])])],linewidth=diam[i]*2, solid_capstyle='butt')
      #plt.plot([y2[i]-tbl[i][1],y2[i]],[pos[i],pos[i]],color='b',linewidth=0.5)
      if (i>0):
	ax1.plot([y2[i]-tbl[i][1] , y2[i]-tbl[i][1]] , [pos[i]+1,pos[int(xy[i])]+1],color=cll[int(clr[int(pos[i])])],linewidth= 0.1, solid_capstyle='butt') # diam[i]*5)
	#plt.plot([y2[i]-tbl[i][1] , y2[i]-tbl[i][1]] , [pos[i],pos[int(xy[i])]],color=cll[int(clr[int(pos[i])])],linewidth= 0.1)

    ax1.set_yticks([])
    ax1.set_xlabel('length ($\mu$m)')
    #ax1.set_ylim([-1,len(tbl)-0])  # ax1.set_ylim([-1,len(tbl)-0.8])  #  shared y axis
    ax1.set_xlim([0, max(y2)]) #math.ceil(max(y2)/100.0)*100])  #  +20
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)    
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06, bottom=0.09)
    fig.savefig('/home/userlab/neuron/figures/L23_LBC_cNAC187_5-full.pdf')




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
        pylab.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(plot_traces=True)
    elif len(sys.argv) == 2 and sys.argv[1] == '--no-plots':
        main(plot_traces=False)
    else:
        raise Exception(
            "Script only accepts one argument: --no-plots, not %s" %
            str(sys.argv))
