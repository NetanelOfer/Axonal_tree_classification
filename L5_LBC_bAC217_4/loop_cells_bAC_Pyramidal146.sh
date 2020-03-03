#!/bin/bash

printf "\n" > bAC_Pyramidal146_branch_f300_new.txt

i=0
file="/home/netanel/Dropbox/neuron/BP_count/Pyramidal146.txt"
while IFS= read f
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_3daa582d70\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_3daa582d70\n" > morphology.hoc
  printf "\n$f\n" >> bAC_Pyramidal146_branch_f300_new.txt
  python run_bAC_BranchingPoint_type.py >> bAC_Pyramidal146_branch_f300_new.txt
done <"$file"

printf "\n"

# chmod u+x loop_cells_bAC_Pyramidal146.sh
# ./loop_cells_bAC_Pyramidal146.sh