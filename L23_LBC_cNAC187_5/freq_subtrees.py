import matplotlib.pyplot as plt

# Sub-Trees
cNAC_BC143ax2 = [range(150,370,10) , [0, 0, 1, 1, 1, 1,1,2,2,3,3,4,6,7,7,7,7,11,10,13,13,13]]
cNAC_BC143ax2_pic = [[200, 260, 300, 350] , [1, 4, 7, 13]]
cAC_BC143ax2 = [[30,40,50,60,80,100,130,150,160,180,200,210,230,260,280,330,360],[0,1,1,1,2,2,2,3,6,7,8,10,11,12,14,14,16]]

# percentage of branches
cNAC_BC143ax2 = [range(150,510,10) , [0, 0, .251, .251, .251, .251, .251, .435, .435,.472,.472,.509,.531,.546,.546,.546,.55,.609,.697,.756,.771,.779,.811,.852,.856,.871,.886,.9,.926,.93,.93,.945,.941,.985,.985,.985]]
cNAC_BC143ax2_pic = [[200, 260, 300, 350] , [0.251, .509, .546, .771]]
cAC_BC143ax2 = [range(20,510,10),[0,0,.251,.251,.251,.251,.435,.435,.435,.435,.435,.435,.472,.472,.531,.531,.546,.546,.554,.598,.653,.657,.76,.775,.782,.849,.863,.878,.889,.893,.904,.926,.937,.945,.985,.985,.985,.985,.989,.989,.989,.989,.989,.989,.989,.989,.989,.989,.989      ]]

shf = 0.05 #0.8

fig = plt.figure(figsize=(10,6))
plt.step(cNAC_BC143ax2[0] , cNAC_BC143ax2[1],'b', label='cNAC')
plt.plot(cNAC_BC143ax2_pic[0] , cNAC_BC143ax2_pic[1],'bo')
plt.text(cNAC_BC143ax2_pic[0][0]-0 , cNAC_BC143ax2_pic[1][0]-shf,'A',fontsize=15)
plt.text(cNAC_BC143ax2_pic[0][1]-0 , cNAC_BC143ax2_pic[1][1]-shf,'B',fontsize=15)
plt.text(cNAC_BC143ax2_pic[0][2]-0 , cNAC_BC143ax2_pic[1][2]-shf,'C',fontsize=15)
plt.text(cNAC_BC143ax2_pic[0][3]-0 , cNAC_BC143ax2_pic[1][3]-shf,'D',fontsize=15)
#plt.step(cNAC_BC143ax2[0] , cNAC_BC143ax2[1],'.b')
plt.step(cAC_BC143ax2[0] , cAC_BC143ax2[1],'g', label='cAC')
#plt.step(cAC_BC143ax2[0] , cAC_BC143ax2[1],'.g')
#plt.plot(i,np.mean(GRs[i]),'.', markeredgecolor=col_typ[typ[i]], markerfacecolor=col_typ[typ[i]])
plt.xlabel('frequency (Hz)',fontsize=16)
plt.ylabel('percentage of interrupted branches',fontsize=16)
plt.xticks(fontsize=16) #, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
fig.tight_layout()
fig.savefig('/home/userlab/BBP/NeuroMorpho/cNAC_BC143ax2_freq_per_branches.pdf')



############

L5_STPC_cADpyr232_2 = [range(10,120,10) , [0.134,0.134,0.134,0.134,0.134,0.1546,0.1649,0.2061,0.3711,0.4845,1]]

fig = plt.figure(figsize=(10,6))
plt.step(L5_STPC_cADpyr232_2[0] , L5_STPC_cADpyr232_2[1],'hotpink', label='cAD')

plt.xlabel('frequency (Hz)',fontsize=16)
plt.ylabel('percentage of interrupted branches',fontsize=16)
plt.xticks(fontsize=16) #, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
fig.tight_layout()
fig.savefig('/home/userlab/BBP/NeuroMorpho/step_L5_STPC_cADpyr232_2.pdf')




############ 10 examples

NMO_02720 = [range(10,280,10) , [0,0,0,.0054,.0108,.0108,.0541,.1243,.1513, .2756,.3621,.4108,.481,.5297,.5675,.7513,.7891,.7946,.8054,.8054,.8918,.9189,.9459,.9459,.9567,.9567, .9567]]
NMO_04586 = [range(10,280,10) , [.2512,.2512,.2512,.2512,.2705,.3092,.3092,.3092,.4009,.4396,.541,.5942,.6135,.6280,.6859,.7343,.7536,.8357,.8454,.8454,.8599,.913,.913,.9951,.9951,.9951,.9951 ]]
NMO_35147 = [range(10,280,10) , [0,0,0,0,0,0,0,0,0,0,.1176,.1176,.1176,.1176,.1176,.1294,.4117,.4470,.6588,.9058, .9058 , .9058,.9176,.9411,.9294,.9529,.9529 ]]
NMO_37282 = [range(10,280,10) , [.054,.054,.054,.054,.054,.054,.1261,.2882,.4054,.4504,.7117,.7297,.7927,.8198,.8288,.8288,.8378,.9459,.9459,.9459,.9459,.9459,1  ,1,1,1,1 ]]
NMO_59679 = [range(10,280,10) , [0,0,0,0,0,0, 0 , 0.0945 , .1574 , .1653 , .1653, .1653 , .1889 , .2047 , .2441 , .3465 , .4252 , .4724 , .9685 , .9685, .9685, .9685,.9685,.9685,.9685,.9685,.9685]]
NMO_61457 = [range(10,280,10) , [0.134,0.134,0.134,0.134,0.134,0.1546,0.1649,0.2061,0.3711,0.4845,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]] 
NMO_61655 = [range(10,280,10) , [0.1584,0.1584,0.1584,0.1584,0.1584,.1683,.2475,.2574,.3663,.4356,.4752,.6039,.6138,.6633,.7227,.7326,.7623,.7821,.8415,.8613,.9901,.9901,1,1,1,1,1 ]]

fig = plt.figure(figsize=(10,6))
plt.step(NMO_02720[0] , NMO_02720[1],'yellow', label='NMO_02720') # Yuste , Vn04212005-0-0 , mouse , 
plt.step(NMO_04586[0] , NMO_04586[1],'blue', label='NMO_04586') # Helmstaedter , 2004-02-16-B-L23dendax , rat , L23
plt.step(NMO_35147[0] , NMO_35147[1],'red', label='NMO_35147') # Hay , cell26 , rat , L5
plt.step(NMO_37282[0] , NMO_37282[1],'maroon', label='NMO_37282') # Markram , VD110623-IDA , rat , L5
plt.step(NMO_59679[0] , NMO_59679[1],'green', label='NMO_59679') # Kole , 20140421_c1 , mouse , Thick-tufted , L5
plt.step(NMO_61457[0] , NMO_61457[1],'hotpink', label='NMO_61457') # Staiger , 229_080211AL2-IB_JH , rat , L5b
plt.step(NMO_61655[0] , NMO_61655[1],'turquoise', label='NMO_61655') # Tolias , L5pyr-j140717b , mouse , L5

plt.xlabel('frequency (Hz)',fontsize=16)
plt.ylabel('percentage of interrupted branches',fontsize=16)
plt.xticks(fontsize=16) #, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.xlim([10,500])
fig.tight_layout()
fig.savefig('/home/userlab/BBP/NeuroMorpho/step_PYR2.pdf')


############ Chandelier

NMO_04548 = [range(100,500,20) , [0,0,0,.0192,.0924,.3042,.3761,.4454,.6482,.6585,.6931,.6983,.7188,.7471,.7817,.9987,1,1,1,1 ]]  # Helmstaedter, 2001-11-09-B-L23-dendax
NMO_07472 = [range(100,500,20) , [.0071,.0071,.0071,.0071,.011,.2938,.5426,.5465,.568,.6508,.7003,.8091,.8697,.9498,.9719,.9902,.9908,.9993,.9993 ,1]]  # Kawaguchi - BE62C
NMO_35831 = [range(100,500,20) , [0,0,0,0,0,0,0,0,0,.1546,.1546,.2143,.3907,.7544,.7571,.7951,.8398,.8656,.9483,.9823 ]] # Klausberger - TF35-Axo-axonic-cell
NMO_37138 = [range(100,500,20) , [0,0,0,0,0,0,.0274,.1576,.2563,.4544,.49,.6819,.8293,.8718,.9506,.9623,.965,.9849,.9869,.989 ]] # Markram - MTC050800D-IDD


fig = plt.figure(figsize=(10,6))

plt.step(NMO_04548[0] , NMO_04548[1],'blue' , label='NMO_04548')
plt.step(NMO_07472[0] , NMO_07472[1],'green' , label='NMO_07472')
plt.step(NMO_35831[0] , NMO_35831[1],'navy' , label='NMO_35831')
plt.step(NMO_37138[0] , NMO_37138[1],'maroon', label='NMO_37138')


plt.xlabel('frequency (Hz)',fontsize=16)
plt.ylabel('percentage of interrupted branches',fontsize=16)
plt.xticks(fontsize=16) #, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.xlim([10,500])
fig.tight_layout()
fig.savefig('/home/userlab/BBP/NeuroMorpho/step_ChC.pdf')

############ Neurogliaform

NMO_06138 = [range(180,500,20) , [0,0,.1719,.2217,.2941,.4434,.6923,.8416,.8642,.914,.9411,.9411,.9457,.9502,.9502,.9592 ]] # Cauli , AK160lay
NMO_07504 = [range(160,560,20) , [.0226,.0226,.0226,.0226,.1811,.1811,.3849,.3849,.4452,.6037,.649,.6905,.7283,.7622,.7622,.8415,.9433,.9811,.9811,.9811 ]] # Kawaguchi , PE14A
NMO_37720 = [range(80,500,20) , [0,0, .1864,.1864,.3669,.3849,.5774,.6902,.7699,.809,.8586,.8827,.9127,.9368,.9714,.9759,.9864,.9939,.9969,.9969,.9969  ]] # Markram, SM081023A1-4-IDF
NMO_61561 = [range(140,520,20) , [0,0,.0089,.0134,.0134,.0156,.0178,.0313,.1275,.2438,.2751,.4161,.5458,.966,.9977,.9977,.9977,.9977,.9977 ]] # Tolias , L23NGC-j140908c-cell-3


fig = plt.figure(figsize=(10,6))

plt.step(NMO_06138[0] , NMO_06138[1],'blue' , label='NMO_06138')
plt.step(NMO_07504[0] , NMO_07504[1],'green' , label='NMO_07504')
plt.step(NMO_37720[0] , NMO_37720[1],'navy' , label='NMO_37720')
plt.step(NMO_61561[0] , NMO_61561[1],'maroon', label='NMO_61561')

plt.xlabel('frequency (Hz)',fontsize=16)
plt.ylabel('percentage of interrupted branches',fontsize=16)
plt.xticks(fontsize=16) #, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.xlim([10,500])
fig.tight_layout()
fig.savefig('/home/userlab/BBP/NeuroMorpho/step_NGFC.pdf')


################ Bipolar
NMO_06142 = [range(200,520,20) , [.0282,.0282,.0395,.0395,.0395,.1468,.2259,.3333,.6723,.8305,.8474,.8983,.9209,.9378,.9491,.9887 ]] # Cauli , BC131sdaxlay
NMO_37531 = [range(200,520,20)  , [0,0,.0136,.1095,.8767,.8767,.9178,.9041,.9041,.9178,.9315,.9452,.9452,.9863,.9863,.9863 ]] # Markram , SM110127B1-3-INT-IDC.CNG.swc
NMO_61609 = [range(180,560,20) , [0,0,.0193,.0258,.0258,.0322,.4,.4322,.6451,.6967,.787,.8,.8516,.8645,.9354,.9548,.9612,.9677,.987 ]] # Tolias , L23BPC-j150713g
NMO_79467 = [range(160,520,20) , [0,0,.0497,.5124,.5124,.5124,.5373,.791,.9054,.9203,.9353,.9452,.9502,.9552,.9701,.9751,.9751,.9751 ]] # Stainger , FW20141007-1-1-VIP



fig = plt.figure(figsize=(10,6))

plt.step(NMO_06142[0] , NMO_06142[1],'blue' , label='NMO_06142')
plt.step(NMO_37531[0] , NMO_37531[1],'green' , label='NMO_37531')
plt.step(NMO_61609[0] , NMO_61609[1],'navy' , label='NMO_61609')
plt.step(NMO_79467[0] , NMO_79467[1],'maroon', label='NMO_79467')

plt.xlabel('frequency (Hz)',fontsize=16)
plt.ylabel('percentage of interrupted branches',fontsize=16)
plt.xticks(fontsize=16) #, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.xlim([10,500])
fig.tight_layout()
fig.savefig('/home/userlab/BBP/NeuroMorpho/step_BP.pdf')

