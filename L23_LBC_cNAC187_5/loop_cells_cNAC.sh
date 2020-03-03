#!/bin/bash

printf "\n" > cNAC_branch.txt

i=0
FILES=/home/userlab/neuron/interneuron16_six_groups/**/*.swc # 96 interneurons
#FILES=/home/userlab/neuron/interneuron16_six_groups/a/*.swc # 96 interneurons
for f in $FILES
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_d3f79b893e\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_d3f79b893e\n" > morphology.hoc
  printf "\n$f\n" >> cNAC_branch.txt
  python run_netanel_cNAC_BranchingPoint_type.py >> cNAC_branch.txt
done

printf "\n"

# chmod u+x loop_cells_cNAC.sh
# ./loop_cells_cNAC.sh