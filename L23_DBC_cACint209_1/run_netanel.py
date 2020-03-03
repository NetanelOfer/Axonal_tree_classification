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
import numpy
import sys
import matplotlib.pyplot as plt

def create_cell(add_synapses=True):
    """Create the cell model"""
    # Load morphology
    neuron.h.load_file("morphology.hoc")
    # Load biophysics
    neuron.h.load_file("biophysics.hoc")
    # Load main cell template
    neuron.h.load_file("template.hoc")
    # Instantiate the cell from the template
    print("Loading cell cACint209_L23_DBC_4114c4c36c")
    cell = neuron.h.cACint209_L23_DBC_4114c4c36c(1 if add_synapses else 0)
    return cell


def create_stimuli(cell, step_number):
    """Create the stimuli"""
    stimuli = []
    iclamp = neuron.h.IClamp(0.5, sec=cell.soma[0])
    iclamp.delay = 10 #700
    iclamp.dur = 20 #00
    iclamp.amp = 0.2 #float(step_amp[step_number - 1])
    #stimuli.append(iclamp)
    dtt = 0.025
    freq = 3 #8 # 125 Hz
    pulses = []
    st = {}
    for i in range(20):
      st["stim" + str(i)] = neuron.h.IClamp(0.5, sec=cell.soma[0])
      st["stim" + str(i)].delay = freq*i + 1 
      st["stim" + str(i)].amp = 20
      st["stim" + str(i)].dur = 1
      pulses.append((freq*i + 1)/dtt)
    stimuli.append(st)
    #stimuli.append(st['stim1'])
    return stimuli


def create_recordings(cell):
    """Create the recordings"""
    print('Attaching recording electrodes')
    recordings = {}
    recordings['time'] = neuron.h.Vector()
    recordings['soma(0.5)'] = neuron.h.Vector()
    recordings['axon17'] = neuron.h.Vector()
    recordings['axon90'] = neuron.h.Vector()
    recordings['axon124'] = neuron.h.Vector()
    recordings['axon170'] = neuron.h.Vector()
    d = {}
    for i in range(171):
      d["v_vect" + str(i)] = neuron.h.Vector()
    for i in range(171):
      d["v_vect" + str(i)].record(cell.axon[i](0.5)._ref_v, 0.1)
    recordings['time'].record(neuron.h._ref_t, 0.1)
    recordings['soma(0.5)'].record(cell.soma[0](0.5)._ref_v, 0.1)
    recordings['axon17'].record(cell.axon[17](0.5)._ref_v, 0.1)
    recordings['axon90'].record(cell.axon[90](0.5)._ref_v, 0.1)
    recordings['axon124'].record(cell.axon[124](0.5)._ref_v, 0.1)
    recordings['axon170'].record(cell.axon[170](0.5)._ref_v, 0.1)
    
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
    # Overriding default 30s simulation,
    print('Setting simulation time to 80ms')
    neuron.h.tstop = 80 # 3000
    #print neuron.h.dt
    print('Disabling variable timestep integration')
    neuron.h.cvode_active(0)
    print('Running for %f ms' % neuron.h.tstop)
    neuron.h.run()
    time = numpy.array(recordings['time'])
    soma_voltage = numpy.array(recordings['soma(0.5)'])
    recordings_dir = 'python_recordings'
    soma_voltage_filename = os.path.join(
        recordings_dir,
        'soma_voltage_step%d.dat' % step_number)
    numpy.savetxt(
            soma_voltage_filename,
            numpy.transpose(
               numpy.vstack((
                    time,
                    soma_voltage))))
    print('Soma voltage for step %d saved to: %s'
          % (step_number, soma_voltage_filename))
    if plot_traces:
        import pylab
        #pylab.figure()
        f, axarr = plt.subplots(5, sharex=True)
        axarr[0].plot(recordings['time'], recordings['soma(0.5)'])
        axarr[1].plot(recordings['time'], recordings['axon17'])
        axarr[2].plot(recordings['time'], recordings['axon90'])
        axarr[3].plot(recordings['time'], recordings['axon124'])
        axarr[4].plot(recordings['time'], recordings['axon170'])
        #pylab.plot(recordings['time'], recordings['axon170'])  #  recordings['soma(0.5)']
        #pylab
        plt.xlabel('time (ms)')
        #pylab.ylabel('Vm (mV)')
        #pylab.gcf().canvas.set_window_title('Step %d' % step_number)

#ax4 = fig.add_subplot(5,1,3)
#ax4.plot(t_vec, v_vec2) # soma_plot = 
#ax4.set_ylabel('mV')
#ax4.set_xticks([]) # Use ax2's tick labels
#ax4.set_xlim([0,xlimend])
#ax4.set_ylim([-80,55])



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
    for step_number in range(1, 2):   # (1, 4)
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
