param_hhar_exp2_icarl = [0.750,      0.761, 0.7813634618,      0.812, 0.8222991894]
param_pamap2_exp2_icarl = [0.821,    0.8607621947,      0.863, 0.869, 0.872]
param_skoda_exp2_icarl = [0.796075897,     0.8801510179,      0.8966508394,      0.893, 0.9228500302]
param_ninapro_db2_c10_exp2_icarl = [0.2756410211,      0.3209292622,      0.3065792881,      0.319, 0.3013097693]


param_hhar_exp2_gem = [0.5478674332, 0.609, 0.615, 0.616, 0.61]
param_pamap2_exp2_gem = [0.695,      0.780, 0.782, 0.790, 0.79]
param_skoda_exp2_gem = [0.7082090832,      0.740, 0.773, 0.725, 0.74]
param_ninapro_db2_c10_exp2_gem = [0.2094346671,  0.204, 0.215, 0.216, 0.209]



plot_all_methods_param(param_hhar_exp2_icarl, dataset='hhar', methods='iCaRL', exp_setup='', logaxis='no')
plot_all_methods_param(param_pamap2_exp2_icarl, dataset='pamap2', methods='iCaRL', exp_setup='', logaxis='no')
plot_all_methods_param(param_skoda_exp2_icarl, dataset='skoda', methods='iCaRL', exp_setup='', logaxis='no')
plot_all_methods_param(param_ninapro_db2_c10_exp2_icarl, dataset='ninapro_db2_c10', methods='iCaRL', exp_setup='', logaxis='no')


plot_all_methods_param(param_hhar_exp2_gem, dataset='hhar', methods='GEM', exp_setup='', logaxis='no')
plot_all_methods_param(param_pamap2_exp2_gem, dataset='pamap2', methods='GEM', exp_setup='', logaxis='no')
plot_all_methods_param(param_skoda_exp2_gem, dataset='skoda', methods='GEM', exp_setup='', logaxis='no')
plot_all_methods_param(param_ninapro_db2_c10_exp2_gem, dataset='ninapro_db2_c10', methods='GEM', exp_setup='', logaxis='no')











run_exp12_icarl(rClassD2List=class_D2_list_task1_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=20)


run_exp12_icarl(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=20)





run_exp12_icarl(rClassD2List=class_D2_list_task1_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=20)


run_exp12_icarl(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=1,
            exp_setup='', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=20, cuda_num=0,
            exp_setup='', budget_percent=20)





run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=1,
            exp_setup='', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=0,
            exp_setup='', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=1,
            exp_setup='', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=0,
            exp_setup='', budget_percent=20)


run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=1,
            exp_setup='', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=0,
            exp_setup='', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=1,
            exp_setup='', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=30, cuda_num=0,
            exp_setup='', budget_percent=20)






run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=1,
            subject_idx=11, exp_setup='per-subject', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=0,
            subject_idx=11, exp_setup='per-subject', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=1,
            subject_idx=11, exp_setup='per-subject', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=0,
            subject_idx=11, exp_setup='per-subject', budget_percent=20)


run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=1,
            subject_idx=11, exp_setup='per-subject', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=0,
            subject_idx=11, exp_setup='per-subject', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=1,
            subject_idx=11, exp_setup='per-subject', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=70, cuda_num=0,
            subject_idx=11, exp_setup='per-subject', budget_percent=20)







run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=1,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=0,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=1,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task1_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=0,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=20)


run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=1,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=1)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=0,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=5)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=1,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=10)
run_exp12_icarl(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='ninapro-db2-c10', rEpoch=20, cuda_num=0,
            subject_idx=14,exp_setup='leave-one-user-out', budget_percent=20)

















run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=50, cuda_num=1,
            budget_percent=1, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=50, cuda_num=0,
            budget_percent=5, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=50, cuda_num=1,
            budget_percent=10, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c6_2, 
            rDataset='hhar-noaug', rEpoch=50, cuda_num=0,
            budget_percent=20, exp_setup='',max_num_cases=5)







run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=50, cuda_num=1,
            budget_percent=1, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=50, cuda_num=0,
            budget_percent=5, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=50, cuda_num=1,
            budget_percent=10, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c12_2, 
            rDataset='pamap2', rEpoch=50, cuda_num=0,
            budget_percent=20, exp_setup='',max_num_cases=5)







run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=50, cuda_num=1,
            budget_percent=1, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=50, cuda_num=0,
            budget_percent=5, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=50, cuda_num=1,
            budget_percent=10, exp_setup='',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=50, cuda_num=0,
            budget_percent=20, exp_setup='',max_num_cases=5)

run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=50, cuda_num=0,
            budget_percent=60, exp_setup='',max_num_cases=5)








run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=100, cuda_num=1, subject_idx=11,
      exp_setup='per-subject',budget_percent=1,max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=100, cuda_num=0, subject_idx=11,
      exp_setup='per-subject',budget_percent=5,max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=100, cuda_num=1, subject_idx=11,
      exp_setup='per-subject',budget_percent=10,max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=100, cuda_num=0, subject_idx=11,
      exp_setup='per-subject',budget_percent=20,max_num_cases=5)

run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=100, cuda_num=0, subject_idx=11,
      exp_setup='per-subject',budget_percent=60,max_num_cases=5)








run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=50, cuda_num=1, subject_idx=14,
      exp_setup='leave-one-user-out',budget_percent=1,max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=50, cuda_num=0, subject_idx=14,
      exp_setup='leave-one-user-out',budget_percent=5,max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=50, cuda_num=1, subject_idx=14,
      exp_setup='leave-one-user-out',budget_percent=10,max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
      rDataset='ninapro-db2-c10',rEpoch=50, cuda_num=0, subject_idx=14,
      exp_setup='leave-one-user-out',budget_percent=20,max_num_cases=5)
















y1 = [0.750,0.821,0.796075897,0.2756410211]
y2 = [0.761,0.8607621947,0.8801510179,0.3209292622]
y3 = [0.7813634618,0.863,0.8966508394,0.3065792881]
y4 = [0.812,0.869,0.893,0.319]

init=1.0


ax1.bar(init, y1[0], width=bar_width, align='center', color='C3', hatch='/',label='$\mathcal{B}=1\%$',alpha=1,edgecolor='black')
ax1.text(init, y1[0]+0.03, '%.2f' % y1[0], va='center', ha='center', rotation=90)
for i in range(1,len(y1)):
    ax1.bar(i+init, y1[i], width=bar_width, align='center', color='C3', hatch='/',alpha=1,edgecolor='black')
    ax1.text(i+init, y1[i]+0.03, '%.2f' % y1[i], va='center', ha='center', rotation=90)
    
ax1.bar(init+bar_width, y2[0], width=bar_width, align='center', color='C1', hatch='\\',label='$\mathcal{B}=5\%$',alpha=1,edgecolor='black')
ax1.text(init+bar_width, y2[0]+0.03, '%.2f' % y2[0], va='center', ha='center', rotation=90)
for i in range(1,len(y2)):
    ax1.bar(i+init+bar_width, y2[i], width=bar_width, align='center', color='C1', hatch='\\',alpha=1,edgecolor='black')
    ax1.text(i+init+bar_width, y2[i]+0.03, '%.2f' % y2[i], va='center', ha='center', rotation=90)

ax1.bar(init+2*bar_width, y3[0], width=bar_width, align='center', color='C2', hatch='/',label='$\mathcal{B}=10\%$',alpha=1,edgecolor='black')
ax1.text(init+2*bar_width, y3[0]+0.03, '%.2f' % y3[0], va='center', ha='center', rotation=90)
for i in range(1,len(y3)):
    ax1.bar(i+init+2*bar_width, y3[i], width=bar_width, align='center', color='C2', hatch='/',alpha=1,edgecolor='black')
    ax1.text(i+init+2*bar_width, y3[i]+0.03, '%.2f' % y3[i], va='center', ha='center', rotation=90)

ax1.bar(init+3*bar_width, y4[0], width=bar_width, align='center', color='C0', hatch='\\',label='$\mathcal{B}=20\%$',alpha=1,edgecolor='black')
ax1.text(init+3*bar_width, y4[0]+0.03, '%.2f' % y4[0], va='center', ha='center', rotation=90)
for i in range(1,len(y4)):
    ax1.bar(i+init+3*bar_width, y4[i], width=bar_width, align='center', color='C0', hatch='\\',alpha=1,edgecolor='black')
    ax1.text(i+init+3*bar_width, y4[i]+0.03, '%.2f' % y4[i], va='center', ha='center', rotation=90)




run_exp3_ewc_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2,rDataset='skoda',
                        rEpoch=1, cuda_num=-1,exp_setup='time',max_num_cases=5)
run_exp3_ewc_online_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2,rDataset='skoda',
                        rEpoch=1, cuda_num=-1,exp_setup='time',max_num_cases=5)
run_exp3_si_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2,rDataset='skoda',
                        rEpoch=1, cuda_num=-1,exp_setup='time',max_num_cases=5)
run_exp3_lwf_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2,rDataset='skoda',
                        rEpoch=1, cuda_num=-1,exp_setup='time',max_num_cases=5)


run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=1, exp_setup='time',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=5, exp_setup='time',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=10, exp_setup='time',max_num_cases=5)
run_exp3_icarl_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=20, exp_setup='time',max_num_cases=5)


run_exp3_gem_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=1, exp_setup='time', max_num_cases=5)
run_exp3_gem_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=5, exp_setup='time', max_num_cases=5)
run_exp3_gem_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=10, exp_setup='time', max_num_cases=5)
run_exp3_gem_from_t0_permu(rClassD2List=class_D2_list_task2_c10_2, 
            rDataset='skoda', rEpoch=1, cuda_num=-1,
            budget_percent=20, exp_setup='time', max_num_cases=5)










joint = hhar_joint_precision_dict_list
temp = [exp1_hhar_icarl1_precision_dict_list_list,
      exp1_hhar_icarl5_precision_dict_list_list,
      exp1_hhar_icarl10_precision_dict_list_list,
      exp1_hhar_icarl20_precision_dict_list_list,
      exp1_hhar_icarl40_precision_dict_list_list,
      exp1_hhar_gem1_precision_dict_list_list,
      exp1_hhar_gem5_precision_dict_list_list,
      exp1_hhar_gem10_precision_dict_list_list,
      exp1_hhar_gem20_precision_dict_list_list,
      exp1_hhar_gem40_precision_dict_list_list]
methods=['iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='hhar',methods=methods,epoch=20,exp=2)


joint = pamap2_joint_precision_dict_list
temp = [exp1_pamap2_icarl1_precision_dict_list_list,
      exp1_pamap2_icarl5_precision_dict_list_list,
      exp1_pamap2_icarl10_precision_dict_list_list,
      exp1_pamap2_icarl20_precision_dict_list_list,
      exp1_pamap2_icarl40_precision_dict_list_list,
      exp1_pamap2_gem1_precision_dict_list_list,
      exp1_pamap2_gem5_precision_dict_list_list,
      exp1_pamap2_gem10_precision_dict_list_list,
      exp1_pamap2_gem20_precision_dict_list_list,
      exp1_pamap2_gem40_precision_dict_list_list]
methods=['iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='pamap2',methods=methods,epoch=20,exp=2)


joint = skoda_joint_precision_dict_list
temp = [exp1_skoda_icarl1_precision_dict_list_list,
      exp1_skoda_icarl5_precision_dict_list_list,
      exp1_skoda_icarl10_precision_dict_list_list,
      exp1_skoda_icarl20_precision_dict_list_list,
      exp1_skoda_icarl40_precision_dict_list_list,
      exp1_skoda_gem1_precision_dict_list_list,
      exp1_skoda_gem5_precision_dict_list_list,
      exp1_skoda_gem10_precision_dict_list_list,
      exp1_skoda_gem20_precision_dict_list_list,
      exp1_skoda_gem40_precision_dict_list_list]
methods=['iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='skoda',methods=methods,epoch=30,exp=2)

joint = ninapro_db2_c10_joint_precision_dict_list
temp = [exp1_ninapro_db2_c10_icarl1_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl5_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl10_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl20_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl40_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem1_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem5_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem10_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem20_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem40_precision_dict_list_list]
methods=['iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='ninapro_db2_c10',methods=methods,epoch=20,exp=2)













joint = hhar_joint_precision_dict_list
temp = [exp1_hhar_none_precision_dict_list_list,
        exp1_hhar_ewc_precision_dict_list_list,
        exp1_hhar_si_precision_dict_list_list,
        exp1_hhar_lwf_precision_dict_list_list,
      exp1_hhar_icarl1_precision_dict_list_list,
      exp1_hhar_icarl5_precision_dict_list_list,
      exp1_hhar_icarl10_precision_dict_list_list,
      exp1_hhar_icarl20_precision_dict_list_list,
      exp1_hhar_icarl40_precision_dict_list_list,
      exp1_hhar_gem1_precision_dict_list_list,
      exp1_hhar_gem5_precision_dict_list_list,
      exp1_hhar_gem10_precision_dict_list_list,
      exp1_hhar_gem20_precision_dict_list_list,
      exp1_hhar_gem40_precision_dict_list_list]
methods=['None','EWC','SI','LwF','iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='hhar',methods=methods,epoch=20,exp=1)



joint = pamap2_joint_precision_dict_list
temp = [exp1_pamap2_none_precision_dict_list_list,
        exp1_pamap2_ewc_precision_dict_list_list,
        exp1_pamap2_si_precision_dict_list_list,
        exp1_pamap2_lwf_precision_dict_list_list,
        exp1_pamap2_icarl1_precision_dict_list_list,
      exp1_pamap2_icarl5_precision_dict_list_list,
      exp1_pamap2_icarl10_precision_dict_list_list,
      exp1_pamap2_icarl20_precision_dict_list_list,
      exp1_pamap2_icarl40_precision_dict_list_list,
      exp1_pamap2_gem1_precision_dict_list_list,
      exp1_pamap2_gem5_precision_dict_list_list,
      exp1_pamap2_gem10_precision_dict_list_list,
      exp1_pamap2_gem20_precision_dict_list_list,
      exp1_pamap2_gem40_precision_dict_list_list]
methods=['None','EWC','SI','LwF','iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='pamap2',methods=methods,epoch=20,exp=1)





joint = skoda_joint_precision_dict_list
temp = [exp1_skoda_none_precision_dict_list_list,
        exp1_skoda_ewc_precision_dict_list_list,
        exp1_skoda_si_precision_dict_list_list,
        exp1_skoda_lwf_precision_dict_list_list,
      exp1_skoda_icarl1_precision_dict_list_list,
      exp1_skoda_icarl5_precision_dict_list_list,
      exp1_skoda_icarl10_precision_dict_list_list,
      exp1_skoda_icarl20_precision_dict_list_list,
      exp1_skoda_icarl40_precision_dict_list_list,
      exp1_skoda_gem1_precision_dict_list_list,
      exp1_skoda_gem5_precision_dict_list_list,
      exp1_skoda_gem10_precision_dict_list_list,
      exp1_skoda_gem20_precision_dict_list_list,
      exp1_skoda_gem40_precision_dict_list_list]
methods=['None','EWC','SI','LwF','iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='skoda',methods=methods,epoch=30,exp=1)






joint = ninapro_db2_c10_1subj_joint_precision_dict_list
temp = [exp3_ninapro_db2_c10_1subj_none_precision_dict_list_list,
        exp3_ninapro_db2_c10_1subj_ewc_precision_dict_list_list,
        exp3_ninapro_db2_c10_1subj_ewc_online_precision_dict_list_list,
        exp3_ninapro_db2_c10_1subj_si_precision_dict_list_list,
        exp3_ninapro_db2_c10_1subj_lwf_precision_dict_list_list,
        exp3_ninapro_db2_c10_1subj_icarl20_precision_dict_list_list,
        exp3_ninapro_db2_c10_1subj_gem20_precision_dict_list_list
       ]
methods=['None','EWC','Online EWC','SI','LwF','iCaRL','GEM']
plot_all_methods_wl_exp3(temp, joint, dataset='ninapro_db2_c10_1subj',methods=methods, exp_setup='')






joint = ninapro_db2_c10_joint_precision_dict_list
temp = [exp1_ninapro_db2_c10_none_precision_dict_list_list,
        exp1_ninapro_db2_c10_ewc_precision_dict_list_list,
        exp1_ninapro_db2_c10_si_precision_dict_list_list,
        exp1_ninapro_db2_c10_lwf_precision_dict_list_list,
        exp1_ninapro_db2_c10_icarl1_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl5_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl10_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl20_precision_dict_list_list,
      exp1_ninapro_db2_c10_icarl40_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem1_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem5_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem10_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem20_precision_dict_list_list,
      exp1_ninapro_db2_c10_gem40_precision_dict_list_list]
methods=['None','EWC','SI','LwF','iCaRL1','iCaRL5','iCaRL10','iCaRL20','iCaRL40','GEM1','GEM5','GEM10','GEM20','GEM40',]
all_methods_forget_intransigence(temp, joint, dataset='ninapro_db2_c10',methods=methods,epoch=20,exp=1)





joint = ninapro_db3_c10_joint_precision_dict_list
temp = [exp1_ninapro_db3_c10_none_precision_dict_list_list,
        exp1_ninapro_db3_c10_ewc_precision_dict_list_list,
        exp1_ninapro_db3_c10_si_precision_dict_list_list,
        exp1_ninapro_db3_c10_lwf_precision_dict_list_list
       ]
methods=['None','EWC','SI','LwF']
all_methods_forget_intransigence(temp, joint, dataset='ninapro_db3_c10',methods=methods,epoch=20,exp=1)








joint = ninapro_db2_c10_1subj_joint_precision_dict_list
temp = [exp1_ninapro_db2_c10_1subj_none_precision_dict_list_list,
        exp1_ninapro_db2_c10_1subj_ewc_precision_dict_list_list,
        exp1_ninapro_db2_c10_1subj_si_precision_dict_list_list,
        exp1_ninapro_db2_c10_1subj_lwf_precision_dict_list_list,
        exp1_ninapro_db2_c10_1subj_icarl20_precision_dict_list_list,
        exp1_ninapro_db2_c10_1subj_gem20_precision_dict_list_list
       ]
methods=['None','EWC','SI','LwF','iCaRL','GEM']
plot_all_methods_wl(temp, joint, dataset='ninapro_db2_c10_1subj',methods=methods,epoch=20,exp=1)




joint = ninapro_db2_c10_joint_precision_dict_list
temp = [exp1_ninapro_db2_c10_none_precision_dict_list_list,
        exp1_ninapro_db2_c10_ewc_precision_dict_list_list,
        exp1_ninapro_db2_c10_si_precision_dict_list_list,
        exp1_ninapro_db2_c10_lwf_precision_dict_list_list,
        exp1_ninapro_db2_c10_icarl20_precision_dict_list_list,
        exp1_ninapro_db2_c10_gem20_precision_dict_list_list
       ]
methods=['None','EWC','SI','LwF','iCaRL','GEM']
plot_all_methods_wl(temp, joint, dataset='ninapro_db2_c10',methods=methods,epoch=20,exp=1)







joint = ninapro_db2_c10_1subj_joint_precision_dict_list
temp = [exp2_ninapro_db2_c10_1subj_none_precision_dict_list_list,
        exp2_ninapro_db2_c10_1subj_ewc_precision_dict_list_list,
        exp2_ninapro_db2_c10_1subj_si_precision_dict_list_list,
        exp2_ninapro_db2_c10_1subj_lwf_precision_dict_list_list,
        exp2_ninapro_db2_c10_1subj_icarl20_precision_dict_list_list,
        exp2_ninapro_db2_c10_1subj_gem20_precision_dict_list_list
       ]
methods=['None','EWC','SI','LwF','iCaRL','GEM']
plot_all_methods_wl(temp, joint, dataset='ninapro_db2_c10_1subj',methods=methods,epoch=20,exp=2)




joint = ninapro_db2_c10_joint_precision_dict_list
temp = [exp2_ninapro_db2_c10_none_precision_dict_list_list,
        exp2_ninapro_db2_c10_ewc_precision_dict_list_list,
        exp2_ninapro_db2_c10_si_precision_dict_list_list,
        exp2_ninapro_db2_c10_lwf_precision_dict_list_list,
        exp2_ninapro_db2_c10_icarl20_precision_dict_list_list,
        exp2_ninapro_db2_c10_gem20_precision_dict_list_list
       ]
methods=['None','EWC','SI','LwF','iCaRL','GEM']
plot_all_methods_wl(temp, joint, dataset='ninapro_db2_c10',methods=methods,epoch=20,exp=2)






\multirow{3}{*}{Reg-based} & EWC & $2 \times \mathcal{M} \times \mathcal{T}$ & 0.198MB & 0.198MB & 0.593MB \\
& Online EWC & $2 \times \mathcal{M}$ & - & - & 0.099MB \\
& SI & $3 \times \mathcal{M}$ & 0.148MB & 0.148MB & 0.148MB \\
\cmidrule(l){1-6}
Replay-based & LwF & $\mathcal{M}$ & 0.049MB & 0.049MB & 0.049MB \\
\cmidrule(l){1-6}
\multirow{2}{*}{Replay+Exemplars} & iCaRL (20\%) & $\mathcal{M} + \mathcal{B}$ & 15.841MB & 15.841MB & 15.841MB \\
& GEM (20\%) & $\mathcal{T} \times \mathcal{M} + \mathcal{B}$ & 16.018MB & 16.018MB & 16.184MB \\









##### EWC Logs #####
# exp1 hhar
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 > ../data/results/ewc_exp1_hhar-noaug_layers_1_units_64_trial_6.txt

# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 > ../data/results/ewc_exp1_hhar-noaug_layers_2_units_64_trial_6.txt

# exp1 pamap2
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 > ../data/results/ewc_exp1_pamap2_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 > ../data/results/ewc_exp1_pamap2_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 > ../data/results/ewc_exp1_pamap2_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 > ../data/results/ewc_exp1_pamap2_layers_2_units_64_trial_10.txt

# exp1 skoda
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 > ../data/results/ewc_exp1_skoda_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 > ../data/results/ewc_exp1_skoda_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 > ../data/results/ewc_exp1_skoda_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=1 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 > ../data/results/ewc_exp1_skoda_layers_2_units_64_trial_10.txt

# exp1 ninapro-db2-c10 per-subject
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp1_ninapro-db2-c10-per-subject_layers_2_units_64_trial_10.txt

# exp1 ninapro-db2-c10 leave-one-user-out
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=1 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp1_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_10.txt


##### EWC Logs #####
# exp2 hhar
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 > ../data/results/ewc_exp2_hhar-noaug_layers_1_units_64_trial_6.txt

# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=hhar-noaug --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 > ../data/results/ewc_exp2_hhar-noaug_layers_2_units_64_trial_6.txt

# exp2 pamap2
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 > ../data/results/ewc_exp2_pamap2_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 > ../data/results/ewc_exp2_pamap2_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 > ../data/results/ewc_exp2_pamap2_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=pamap2 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 > ../data/results/ewc_exp2_pamap2_layers_2_units_64_trial_10.txt

# exp2 skoda
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 > ../data/results/ewc_exp2_skoda_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 > ../data/results/ewc_exp2_skoda_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 > ../data/results/ewc_exp2_skoda_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=skoda --rExp=2 --rEpoch=30 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 > ../data/results/ewc_exp2_skoda_layers_2_units_64_trial_10.txt

# exp2 ninapro-db2-c10 per-subject
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=70 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 --exp_setup=per-subject --subject_idx=11 > ../data/results/ewc_exp2_ninapro-db2-c10-per-subject_layers_2_units_64_trial_10.txt

# exp2 ninapro-db2-c10 leave-one-user-out
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=32 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=1 --n_units=64 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_1_units_64_trial_10.txt

# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=32 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_32_trial_10.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=1 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_1.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=2 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_2.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=3 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_3.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=4 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_4.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=5 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_5.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=6 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_6.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=7 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_7.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=8 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_8.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=9 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_9.txt
# python rev_ewc_wrapper.py --rDataset=ninapro-db2-c10 --rExp=2 --rEpoch=20 --cuda_num=0 --n_layers=2 --n_units=64 --trial=10 --exp_setup=leave-one-user-out --subject_idx=14 > ../data/results/ewc_exp2_ninapro-db2-c10-leave-one-user-out_layers_2_units_64_trial_10.txt



##### ref #####
# python ewc_wrapper.py --rDataset=deepbreathe-noaug --rEpoch=25 --cuda_num=1 --exp_setup=th_deepbreathe_units128 --max_num_cases=1 --cb_test_metric=eval_loss --bm_on=1 --th=0.00001 > ../data/results/ewc_deepbreathe_units128_evalLoss_e25_th_0.00001.txt
#
# run_exp3_icarl(class_D2_list_task2_c10_2, 'ninapro-db2-c10', 30, 0, 1, 5)