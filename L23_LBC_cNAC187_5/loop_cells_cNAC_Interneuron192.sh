#!/bin/bash

printf "\n" > cNAC_Interneuron192_branch_f300_new.txt

i=0
file="/home/netanel/Dropbox/neuron/BP_count/Interneuron192.txt"
while IFS= read f
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_d3f79b893e\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_d3f79b893e\n" > morphology.hoc
  printf "\n$f\n" >> cNAC_Interneuron192_branch_f300_new.txt
  python run_netanel_cNAC_BranchingPoint_type.py >> cNAC_Interneuron192_branch_f300_new.txt
done <"$file"

printf "\n"

# chmod u+x loop_cells_cNAC_Interneuron192.sh
# ./loop_cells_cNAC_Interneuron192.sh