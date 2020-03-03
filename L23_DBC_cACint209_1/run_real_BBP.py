#!/usr/bin/env python

from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
from neuronpy.graphics import spikeplot
from neuronpy.util import spiketrain
from numpy import *
from matplotlib import pyplot
import os
import sys

def raster(event_times_list, **kwargs):
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
      if (ith==0):
        plt.vlines(trial, ith + .5, ith + 1.5, color='red', **kwargs)
      else:
	plt.vlines(trial, ith + .5, ith + 1.5, **kwargs)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax


h.load_file('import3d.hoc')
nl = neuron.h.Import3d_Neurolucida3()
nl.input('morphology/C060400B2_-_Scale_x1.000_y1.050_z1.000_-_Clone_3.asc')
imprt = h.Import3d_GUI(nl,0)
imprt.instantiate(None)

shape_window = h.PlotShape()
neuron.h.topology()

soma = neuron.h.soma[0]
neuron.h.psection(sec=soma)
h.soma[0].insert('hh')
h.soma[0].insert('Im')
neuron.h.psection(sec=soma)

for seg in soma:
   seg.pas.e = -75.300257
   
   

