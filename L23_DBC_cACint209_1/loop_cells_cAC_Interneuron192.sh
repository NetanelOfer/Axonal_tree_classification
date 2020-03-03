#!/bin/bash

printf "\n" > cAC_Interneuron192_branch_f300_new.txt

i=0
file="/home/netanel/Dropbox/neuron/BP_count/Interneuron192.txt"
while IFS= read f
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_4114c4c36c\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_4114c4c36c\n" > morphology.hoc
  printf "\n$f\n" >> cAC_Interneuron192_branch_f300_new.txt
  python run_netanel_cAC_BranchingPoint_type.py >> cAC_Interneuron192_branch_f300_new.txt
done <"$file"

printf "\n"

# chmod u+x loop_cells_cAC_Interneuron192.sh
# ./loop_cells_cAC_Interneuron192.sh