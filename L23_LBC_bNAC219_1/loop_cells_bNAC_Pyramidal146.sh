#!/bin/bash

printf "\n" > bNAC_Pyramidal146_branch_f300_new.txt

i=0
file="/home/netanel/Dropbox/neuron/BP_count/Pyramidal146.txt"
while IFS= read f
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_fe2122c75c\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_fe2122c75c\n" > morphology.hoc
  printf "\n$f\n" >> bNAC_Pyramidal146_branch_f300_new.txt
  python run-bNAC_BranchingPoint_type.py >> bNAC_Pyramidal146_branch_f300_new.txt
done <"$file"

printf "\n"

# chmod u+x loop_cells_bNAC_Pyramidal146.sh
# ./loop_cells_bNAC_Pyramidal146.sh