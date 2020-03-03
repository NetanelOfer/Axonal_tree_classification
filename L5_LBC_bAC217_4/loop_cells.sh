#!/bin/bash

printf "\n" > bAC2_cont.txt

i=0
FILES=/home/userlab/neuron/bAC/morphology/**/*.swc
for f in $FILES
do
  echo "$i"
  ((i = i + 1))
  printf "begintemplate morphology_3daa582d70\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042$f\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_3daa582d70\n" > morphology.hoc
  printf "\n$f\n" >> bAC2_cont.txt
  python run_bAC_Entropy.py >> bAC2_cont.txt
done

#printf "begintemplate morphology_3daa582d70\npublic morphology\nproc morphology(){localobj nl,import\n\tnl = new Import3d_SWC_read()\n\tnl.quiet = 1\n\tnl.input(\042morphology/BC143ax2.CNG.swc\042)\n\timport = new Import3d_GUI(nl, 0)\n\timport.instantiate(\044o1)\n\t}\nendtemplate morphology_3daa582d70\n" > morphology.hoc

#cat morphology.hoc


printf "\n"
