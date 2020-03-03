#!/bin/bash

printf "\n" > bNAC_fixed3_cont.txt

i=0
FILES=/home/userlab/neuron/L23_LBC_bNAC219_1/morphology/**/*.swc
for f in $FILES
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_fe2122c75c\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_fe2122c75c\n" > morphology.hoc
  printf "\n$f\n" >> bNAC_fixed3_cont.txt
  python run-bNAC_Entropy.py >> bNAC_fixed3_cont.txt
done

printf "\n"

# chmod u+x loop_cells_bNAC.sh
# ./loop_cells_bNAC.sh