#!/bin/bash

printf "\n" > cAC2.txt

i=0
FILES=/home/userlab/neuron/bAC/morphology/**/*.swc
for f in $FILES
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_4114c4c36c\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_4114c4c36c\n" > morphology.hoc
  printf "\n$f\n" >> cAC2.txt
  python run_netanel_cAC_Entropy.py >> cAC2.txt
done

printf "\n"

# chmod u+x loop_cells_cAC.sh
# ./loop_cells_cAC.sh