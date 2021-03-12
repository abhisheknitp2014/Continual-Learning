from rev_main_script import *


def run_exp_joint(rClassD2List,rDataset,rMeasureType,rEpoch,cuda_num,
    exp_type=None,subject_idx=None,exp_setup='',cls_type='lstm',
    n_layers=1,n_units=32,lr2=0.001,batch_size=32,trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names #####
    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu'] 
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    ##### n_tasks #####
    num_tasks = 1
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks, 
                           experiment='sensor', scenario='class', cls_type=cls_type, seed=0)
    epoch = rEpoch
    args.cl_alg = 'joint' if rEpoch!=1 or rMeasureType != 'time' else 'measureTime_'+'joint'
    args.rMeasureType = rMeasureType

    precision_dict_list_list = []
    args_D1_l = []

    # load dataset
    # Prepare data for chosen experiment
    args.class_D2_l = rClassD2List
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_singletask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, class_D2_l=args.class_D2_l, subject_idx=subject_idx, exp_setup=exp_setup,
        args=args)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args_D1 = {}
    acc_D1_max = -0.1
    args.hp = 0
    args.clsSeq = 1

    fc_lay_l,fc_units_l,lr_l,lr2_l,batch_l,ewc_lambda_l,budget_size_l=get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 200
    fc_nl = fc_nl_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr = lr2

    clsSeq = trial-1
    precision_dict_list = [{} for i in range(num_tasks)]
    ##### Tasl 1 #####
    acc_D1_temp_l = []
    args.hp += 1
    # setup args for experiment D1
    print("args fc_lay: %d fc_units: %d lr: %f fc_nl: %s batch: %d" % (fc_lay, fc_units, lr, fc_nl, batch))
    iter_per_epoch = int(len(train_datasets[0])/batch+1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    args.seed = clsSeq
    model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, None)
    precision_dict_list[0] = precision_dict

    best_hp = args.hp
    path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'
    print(path_best_model)
    model = torch.load(path_best_model)

    acc_D1_max = np.max(precision_dict['per_task_weighted_f1'][0])
    acc_D1_temp = precision_dict['per_task_weighted_f1'][0][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    # compare with the best accuracy, store the arguments to args_D1 and store the best model
    # if acc_D1_temp > acc_D1_max:
    #     acc_D1_max = acc_D1_temp

    ##### result reporting
    args_D1['cl_alg'] = args.cl_alg
    args_D1['budget_size'] = 0
    args_D1['scenario'] = getExpType(args)
    args_D1['dataset'] = args.dataset
    args_D1['fc_lay'] = fc_lay
    args_D1['fc_units'] = fc_units
    args_D1['lr'] = lr
    args_D1['lr2'] = lr2
    args_D1['batch'] = batch
    args_D1['saved_model_epoch_l'] = saved_model_epoch_l
    args_D1['total_epochs'] = np.sum(saved_model_epoch_l)
    args_D1['weighted_f1'] = acc_D1_temp
    args_D1['hp'] = args.hp
    args_D1['clsSeq'] = clsSeq
    args_D1_l.append(copy.copy(args_D1))

    print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,0,args.hp,np.sum(saved_model_epoch_l),acc_D1_temp))
    print('saved_model_epoch_l: ', saved_model_epoch_l)
    print("acc_D1_temp: ", acc_D1_temp)
    print("acc_D1_max: ", acc_D1_max)
    acc_D1_temp_l.append(acc_D1_temp)
    precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    if rMeasureType == 'time':
        putTime2List(args)
        pass
    else:
        pass
        # plot_all_metrics(precision_dict_list, epoch, args)
        # precision_dict_list_list[clsSeq] = precision_dict_list[:]
        # args_D1_l_l[clsSeq] = copy.copy(args_D1_l)
    print('precision_dict_list_list: ', precision_dict_list_list)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    print('Best precision_dict_list', precision_dict_list_list[best_hp - 1])
    # if rMeasureType != 'time':
        # plot_all_metrics_summed(precision_dict_list_list, epoch, args)
        # rev_plot_all_metrics_summed(precision_dict_list_list, epoch, args, args_D1_l)
    reportTimeMem(model, args, args.cl_alg)
    print('args_l: ', args_D1_l)
    print('acc_D1_temp_l: ', acc_D1_temp_l)
    print('Best args', args_D1_l[best_hp - 1])
    print('Best HP idx: {}, acc_D1_max: {} '.format(best_hp, acc_D1_max))


def run_exp12_none(rClassD2List,rDataset,rMeasureType,rEpoch,cuda_num,exp_type=None,subject_idx=None,exp_setup='',
                  n_layers=1,n_units=32,lr2=0.001,batch_size=32,trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names #####
    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu']
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    class_D2_list = rClassD2List
    ##### n_tasks #####
    num_tasks = 2
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks,
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'none' if rEpoch != 1 or rMeasureType != 'time' else 'measureTime_' + 'none'
    args.rMeasureType = rMeasureType
    ## rev
    precision_dict_list_list = []
    args_Dn_l = []

    # for clsSeq, class_D2_l in enumerate(class_D2_list):
    clsSeq = trial-1
    class_D2_l = class_D2_list[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(class_D2_list)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.class_D2_l = class_D2_l
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, class_D2_l=class_D2_l, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args_D1 = {}
    acc_D1_max = -0.1
    args.hp = 0
    args.clsSeq = clsSeq+1


    fc_lay_l,fc_units_l,lr_l,lr2_l,batch_l,ewc_lambda_l,budget_size_l=get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 200
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    batch = batch_l[0]
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]

    ##### Task 1 #####
    args.D1orD2 = 1
    args.hp = 1
    # setup args for experiment D1
    print("args fc_lay: %d fc_units: %d lr: %f fc_nl: %s batch: %d" % (fc_lay, fc_units, lr, fc_nl, batch))
    iter_per_epoch = int(len(train_datasets[0])/batch+1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                        + '_hp'+str(args.hp)+'.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['per_task_weighted_f1'][0][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    saved_model_epoch_l.append(0)
    ##### task 2 #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.D1orD2 = 2
    args.hp = 0
    best_hp = 0
    for lr2 in lr2_l:
        args.hp += 1
        print("args lr2: %f batch: %d" %(lr2, batch))
        iter_per_epoch = int(len(train_datasets[1])/batch+1)
        total_iters = iter_per_epoch * epoch
        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        # train on D1 with args_D1 or load model_D1
        model2 = copy.deepcopy(model)
        # train on D2
        model2,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model2)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                    + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                    + '_hp'+str(args.hp)+'.pth'
        model2 = torch.load(path_best_model)
        precision_dict_list[1] = precision_dict
        saved_model_epoch_l[1] = model2.epoch

        acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model2.epoch-1]
        # compare the best accuracy on D1 U D2, D2, D1 // store the arguments to args_D2 and store the best model
        if acc_Dn_temp > acc_Dn_max:
            acc_Dn_max = acc_Dn_temp
            best_hp = args.hp

        ##### result reporting
        args_Dn['cl_alg'] = args.cl_alg
        args_Dn['budget_size'] = 0
        args_Dn['scenario'] = getExpType(args)
        args_Dn['dataset'] = args.dataset
        args_Dn['fc_lay'] = fc_lay
        args_Dn['fc_units'] = fc_units
        args_Dn['lr'] = lr
        args_Dn['lr2'] = lr2
        args_Dn['batch'] = batch
        args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
        args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
        args_Dn['weighted_f1'] = acc_Dn_temp
        args_Dn['hp'] = args.hp
        args_Dn['clsSeq'] = clsSeq
        args_Dn_l.append(copy.copy(args_Dn))

        print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
        print('saved_model_epoch_l: ', saved_model_epoch_l)
        print("acc_Dn_temp: ", acc_Dn_temp)
        print("acc_Dn_max: ", acc_Dn_max)
        acc_Dn_temp_l.append(acc_Dn_temp)
        precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass
    # else:
    #     pass
        # plot_all_metrics(precision_dict_list, epoch, args)
        # precision_dict_list_list[clsSeq] = precision_dict_list[:]
        # args_Dn_l_l[clsSeq] = copy.copy(args_Dn_l)
    print('precision_dict_list_list: ', precision_dict_list_list)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    print('Best precision_dict_list', precision_dict_list_list[best_hp - 1])
    # if rMeasureType != 'time':
        # plot_all_metrics_summed(precision_dict_list_list, epoch, args)
        # rev_plot_all_metrics_summed(precision_dict_list_list, epoch, args, args_Dn_l)
    reportTimeMem(model, args, args.cl_alg)
    print('args_l: ', args_Dn_l)
    print('acc_Dn_temp_l: ', acc_Dn_temp_l)
    print('Best args', args_Dn_l[best_hp - 1])
    print('Best HP idx: {}, acc_D1_max: {}, acc_Dn_max: {}'.format(best_hp, acc_D1_max, acc_Dn_max))





def run_exp3_none_from_t0_permu(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='', max_num_cases=1,
                               n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names ##### 
    input_class_seq_l=makeClassSeqList(class_D2_list=rClassD2List,rDataset=rDataset,max_num_cases=max_num_cases)

    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu'] 
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    ##### n_tasks #####
    num_tasks = len(input_class_seq_l[0])
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks, 
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'none' if rEpoch!=1 or rMeasureType != 'time' else 'measureTime_'+'none'
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(1)]
    args_Dn_l = []

    # for clsSeq, input_class_seq in enumerate(input_class_seq_l):
    clsSeq = trial - 1
    input_class_seq = input_class_seq_l[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(input_class_seq_l)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.input_class_seq = input_class_seq
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment_multi_tasks(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, input_class_seq=input_class_seq, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args.clsSeq = clsSeq + 1 # clsSeq is the saem as trial

    fc_lay_l, fc_units_l, lr_l, lr2_l, batch_l, ewc_lambda_l, budget_size_l = get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 20 if exp_setup == 'per-subject' or rDataset[:7] == 'emotion' else 5
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    # budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    # initialize variables to start again from task 0
    args.D1orD2 = 1
    args.hp = get_ewc_hp_num(n_layers, n_units)

    # setup args for experiment D1
    iter_per_epoch = int(len(train_datasets[0]) / batch + 1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                 test_total_dataset, None)
    if rMeasureType == 'time':
            putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                           + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                           + '_hp' + str(args.hp) + '.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    acc_l = [acc_D1_max]


    ##### task 2 to n #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.hp = 0
    best_hp = 0
    lr2 = lr2_l[0]


    args.hp += 1
    if args.hp > 1:
        model = torch.load(path_best_model_init)
    print("\n===== initialize from t1: args lr2: %f batch: %d" %(lr2,batch))
    
    for task_idx in range(1, num_tasks):
        args.D1orD2 = task_idx + 1  # start from 2
        iter_per_epoch = int(len(train_datasets[task_idx]) / batch + 1)
        total_iters = iter_per_epoch * epoch
        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        # train on D1 with args_D1 or load model_D1 / train on D2
        model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
                args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                              + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                              + '_hp' + str(args.hp) + '.pth'
        model = torch.load(path_best_model)
        precision_dict_list[task_idx] = precision_dict
        acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
        acc_l.append(acc_Dn_temp)
        print('len(precision_dict[1][all_tasks_acc]): ', len(precision_dict['all_tasks_acc']))
        print('saved model.epoch: ', model.epoch)
        saved_model_epoch_l.append(model.epoch)

    # compare with the best accuracy, store the arguments to args_Dn and store the best model
    if acc_Dn_temp > acc_Dn_max:
        acc_Dn_max = acc_Dn_temp
        best_hp = args.hp

    ##### result reporting
    args_Dn['cl_alg'] = args.cl_alg
    args_Dn['budget_size'] = 0
    args_Dn['scenario'] = getExpType(args)
    args_Dn['dataset'] = args.dataset
    args_Dn['fc_lay'] = fc_lay
    args_Dn['fc_units'] = fc_units
    args_Dn['lr'] = lr
    args_Dn['lr2'] = lr2
    args_Dn['batch'] = batch
    args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
    args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
    args_Dn['weighted_f1'] = acc_Dn_temp
    args_Dn['hp'] = args.hp
    args_Dn['clsSeq'] = clsSeq
    args_Dn_l.append(copy.copy(args_Dn))

    print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
    print('saved_model_epoch_l: ', saved_model_epoch_l)
    print("acc_Dn_temp: ", acc_Dn_temp)
    print("acc_Dn_max: ", acc_Dn_max)
    acc_Dn_temp_l.append(acc_Dn_temp)
    precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'

    print('precision_dict_list_list: ', precision_dict_list_list)
    model = torch.load(path_best_model)
    # print('precision_dict_list_list_best_hp: ', precision_dict_list_list[best_hp - 1])

    # compute (1) average performance and (2) average forgetting w.r.t. many metrics (acc, f1, prec, rec)
    # compute (3) Intransigence=a* - a_{k,k} = but we compute a_{k,k} for now.
    avgAccForgetAkk_l[0] = computeAvgAccForgetAkk(model, test_datasets, precision_dict_list_list[best_hp - 1], args)
    # print('\nsaved_model epoch: {} / best HP idx: {} / a_kk: {}'.format(model.epoch, best_hp, acc_Dn_max))
    # print("args_D2: ", args_D2)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass

    reportTimeMem(model, args, args.cl_alg)
    print(path_best_model)
    print('best_saved_model epoch sum: {} and best HP idx: {}'.format(np.sum(args_Dn['saved_model_epoch_l']), best_hp))
    print("best_args_Dn: ", args_Dn)
    print("acc_list: ", acc_l)
    print("best_acc_Dn: ", acc_Dn_max)


def run_exp12_gem(rClassD2List, rDataset,rMeasureType,rEpoch,cuda_num,
    exp_type=None,subject_idx=None,exp_setup='',cls_type='lstm',
    budget_percent=5,n_layers=1,n_units=32,lr2=0.001,batch_size=32,trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names #####
    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu']
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    class_D2_list = rClassD2List
    ##### n_tasks #####
    num_tasks = 2
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks,
                           experiment='sensor', scenario='class', cls_type=cls_type, seed=0)
    epoch = rEpoch
    args.cl_alg = 'gem' + str(budget_percent) if rEpoch != 1 or rMeasureType != 'time' else 'measureTime_' + 'gem' + str(budget_percent)
    ## rev
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    args_Dn_l = []

    # for clsSeq, class_D2_l in enumerate(class_D2_list):
    clsSeq = trial-1
    class_D2_l = class_D2_list[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(class_D2_list)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.class_D2_l = class_D2_l
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, class_D2_l=class_D2_l, subject_idx=subject_idx,exp_setup=exp_setup,
        args=args)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args_D1 = {}
    acc_D1_max = -0.1
    args.hp = 0
    args.clsSeq = clsSeq+1


    fc_lay_l,fc_units_l,lr_l,lr2_l,batch_l,ewc_lambda_l,budget_size_l=get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 200
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    args.D1orD2 = 1
    args.hp = 1
    # setup args for experiment D1
    print("args fc_lay: %d fc_units: %d lr: %f fc_nl: %s batch: %d" % (fc_lay, fc_units, lr, fc_nl, batch))
    iter_per_epoch = int(len(train_datasets[0])/batch+1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    args.set_params_D12_gem(lr, batch, int(budget/num_tasks), 0.5) # batchsize=10 specified in gem paper but we change differently. n_memories=2000 same as iCARL, strength=0.5 same as in git repository
    model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                        + '_hp'+str(args.hp)+'.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['per_task_weighted_f1'][0][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    saved_model_epoch_l.append(0)
    ##### task 2 #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.D1orD2 = 2
    args.hp = 0
    best_hp = 0
    for lr2 in lr2_l:
        args.hp += 1
        print("args lr2: %f budget: %d, batch: %d" %(lr2, budget, batch))
        iter_per_epoch = int(len(train_datasets[1])/batch+1)
        total_iters = iter_per_epoch * epoch
        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        args.set_params_D12_gem(lr2, batch, int(budget/num_tasks), 0.5) # batchsize=10 specified in gem paper but we change differently. n_memories=2000 same as iCARL, strength=0.5 same as in git repository
        # train on D1 with args_D1 or load model_D1
        model2 = copy.deepcopy(model)
        # train on D2
        model2,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model2)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                    + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                    + '_hp'+str(args.hp)+'.pth'
        model2 = torch.load(path_best_model)
        precision_dict_list[1] = precision_dict
        saved_model_epoch_l[1] = model2.epoch

        acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model2.epoch-1]
        # compare the best accuracy on D1 U D2, D2, D1 // store the arguments to args_D2 and store the best model
        if acc_Dn_temp > acc_Dn_max:
            acc_Dn_max = acc_Dn_temp
            best_hp = args.hp

        ##### result reporting
        args_Dn['cl_alg'] = args.cl_alg
        args_Dn['budget_percent'] = budget_percent
        args_Dn['budget'] = budget
        args_Dn['scenario'] = getExpType(args)
        args_Dn['dataset'] = args.dataset
        args_Dn['fc_lay'] = fc_lay
        args_Dn['fc_units'] = fc_units
        args_Dn['lr'] = lr
        args_Dn['lr2'] = lr2
        args_Dn['batch'] = batch
        args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
        args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
        args_Dn['weighted_f1'] = acc_Dn_temp
        args_Dn['hp'] = args.hp
        args_Dn['clsSeq'] = clsSeq
        args_Dn_l.append(copy.copy(args_Dn))

        print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_percent: %d, budget: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,budget_percent,budget,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
        print('saved_model_epoch_l: ', saved_model_epoch_l)
        print("acc_Dn_temp: ", acc_Dn_temp)
        print("acc_Dn_max: ", acc_Dn_max)
        acc_Dn_temp_l.append(acc_Dn_temp)
        precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass
    # else:
    #     pass
        # plot_all_metrics(precision_dict_list, epoch, args)
        # precision_dict_list_list[clsSeq] = precision_dict_list[:]
        # args_Dn_l_l[clsSeq] = copy.copy(args_Dn_l)
    print('precision_dict_list_list: ', precision_dict_list_list)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    print('Best precision_dict_list', precision_dict_list_list[best_hp - 1])
    # if rMeasureType != 'time':
        # plot_all_metrics_summed(precision_dict_list_list, epoch, args)
        # rev_plot_all_metrics_summed(precision_dict_list_list, epoch, args, args_Dn_l)
    reportTimeMem(model, args, args.cl_alg)
    print('args_l: ', args_Dn_l)
    print('acc_Dn_temp_l: ', acc_Dn_temp_l)
    print('Best args', args_Dn_l[best_hp - 1])
    print('Best HP idx: {}, acc_D1_max: {}, acc_Dn_max: {}'.format(best_hp, acc_D1_max, acc_Dn_max))


def run_exp3_gem_from_t0_permu(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,
        exp_type=None, subject_idx=None,exp_setup='',cls_type='lstm',
        budget_percent=5, max_num_cases=1,
        n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names ##### 
    input_class_seq_l = makeClassSeqList(class_D2_list=rClassD2List, rDataset=rDataset, max_num_cases=max_num_cases)

    experiment = 'sensor'
    scenario = 'class'
    optimizer = 'adam'
    fc_nl_l = ['relu']
    datasets_l = ['hhar-noaug', 'pamap2', 'skoda', 'opp_thomas']

    ##### n_tasks #####
    num_tasks = len(input_class_seq_l[0])
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks,
                            experiment='sensor', scenario='class', cls_type=cls_type, seed=0)
    epoch = rEpoch
    args.cl_alg = 'gem' + str(budget_percent) if rEpoch != 1 or rMeasureType != 'time' else 'measureTime_' + 'gem' + str(budget_percent)

    # precision_dict_list_list = [[{} for i in range(num_tasks)] for j in range(len(input_class_seq_l))]
    # rev
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    # avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(len(input_class_seq_l))]
    avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(1)]
    args_Dn_l = []

    # for clsSeq, input_class_seq in enumerate(input_class_seq_l):
    clsSeq = trial - 1
    input_class_seq = input_class_seq_l[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(input_class_seq_l)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.input_class_seq = input_class_seq
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment_multi_tasks(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, input_class_seq=input_class_seq, subject_idx=subject_idx,exp_setup=exp_setup,
        args=args)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args.clsSeq = clsSeq + 1 # clsSeq is the saem as trial

    fc_lay_l, fc_units_l, lr_l, lr2_l, batch_l, ewc_lambda_l, budget_size_l = get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 20 if exp_setup == 'per-subject' or rDataset[:7] == 'emotion' else 5
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    # initialize variables to start again from task 0
    args.D1orD2 = 1
    args.hp = get_icarl_hp_num(n_layers, n_units, budget_percent)

    # setup args for experiment D1
    iter_per_epoch = int(len(train_datasets[0]) / batch + 1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    args.set_params_D12_gem(lr, batch, 94 if int(budget/num_tasks) > 94 else int(budget/num_tasks), 0.5) # batchsize=10 specified in gem paper but we change differently. n_memories=2000 same as iCARL, strength=0.5 same as in git repository
    model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                 test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                           + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                           + '_hp' + str(args.hp) + '.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    acc_l = [acc_D1_max]
    # saved_model_epoch_l.append(0)

    ##### task 2 to n #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    best_hp = 0
    lr2 = lr2_l[0]
    for task_idx in range(1, num_tasks):
        args.D1orD2 = task_idx + 1  # start from 2
        iter_per_epoch = int(len(train_datasets[task_idx]) / batch + 1)
        total_iters = iter_per_epoch * epoch
        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        args.set_params_D12_gem(lr2, batch, 94 if int(budget/num_tasks) > 94 else int(budget/num_tasks), 0.5) # batchsize=10 specified in gem paper but we change differently. n_memories=2000 same as iCARL, strength=0.5 same as in git repository
        model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                      test_total_dataset, model)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
            args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                          + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                          + '_hp' + str(args.hp) + '.pth'
        model = torch.load(path_best_model)
        precision_dict_list[task_idx] = precision_dict
        acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
        acc_l.append(acc_Dn_temp)
        print('len(precision_dict[1][all_tasks_acc]): ', len(precision_dict['all_tasks_acc']))
        print('saved model.epoch: ', model.epoch)
        saved_model_epoch_l.append(model.epoch)


    # compare with the best accuracy, store the arguments to args_Dn and store the best model
    if acc_Dn_temp > acc_Dn_max:
        acc_Dn_max = acc_Dn_temp
        best_hp = args.hp

    ##### result reporting
    args_Dn['cl_alg'] = args.cl_alg
    args_Dn['budget_percent'] = budget_percent
    args_Dn['budget'] = budget
    args_Dn['scenario'] = getExpType(args)
    args_Dn['dataset'] = args.dataset
    args_Dn['fc_lay'] = fc_lay
    args_Dn['fc_units'] = fc_units
    args_Dn['lr'] = lr
    args_Dn['lr2'] = lr2
    args_Dn['batch'] = batch
    args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
    args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
    args_Dn['weighted_f1'] = acc_Dn_temp
    args_Dn['hp'] = args.hp
    args_Dn['clsSeq'] = clsSeq
    args_Dn_l.append(copy.copy(args_Dn))

    print(
        "args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_percent: %d, budget: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (
        fc_lay, fc_units, lr, lr2, batch, budget_percent, budget, args.hp, np.sum(saved_model_epoch_l), acc_Dn_temp))
    print('saved_model_epoch_l: ', saved_model_epoch_l)
    print("acc_Dn_temp: ", acc_Dn_temp)
    print("acc_Dn_max: ", acc_Dn_max)
    acc_Dn_temp_l.append(acc_Dn_temp)
    precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'
    print('precision_dict_list_list: ', precision_dict_list_list)
    model = torch.load(path_best_model)
    # print('precision_dict_list_list_best_hp: ', precision_dict_list_list[best_hp - 1])

    # compute (1) average performance and (2) average forgetting w.r.t. many metrics (acc, f1, prec, rec)
    # compute (3) Intransigence=a* - a_{k,k} = but we compute a_{k,k} for now.
    avgAccForgetAkk_l[0] = computeAvgAccForgetAkk(model, test_datasets, precision_dict_list_list[0], args)
    # print('\nsaved_model epoch: {} / best HP idx: {} / a_kk: {}'.format(model.epoch, best_hp, acc_Dn_max))
    # print("args_D2: ", args_D2)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass
    reportTimeMem(model, args, args.cl_alg)
    print(path_best_model)
    print('best_saved_model epoch sum: {} and best HP idx: {}'.format(np.sum(args_Dn['saved_model_epoch_l']), best_hp))
    print("best_args_Dn: ", args_Dn)
    print("acc_list: ", acc_l)
    print("best_acc_Dn: ", acc_Dn_max)






def run_exp12_icarl(rClassD2List, rDataset,rMeasureType,rEpoch,cuda_num,exp_type=None,subject_idx=None,exp_setup='',
    herding=1,parallel=1, quantize=32, budget_percent=5, augment=-1, scaling=1, sd=1.0,
    n_layers=1,n_units=32,lr2=0.001,batch_size=32,trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names #####
    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu']
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    class_D2_list = rClassD2List
    ##### n_tasks #####
    num_tasks = 2
    args = MyArgs(cuda=True, cuda_num=cuda_num,parallel=parallel,quantize=quantize) if cuda_num >= 0 else MyArgs(cuda=False,parallel=parallel,quantize=quantize)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks,
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'icarl' + str(budget_percent) if rEpoch != 1 or rMeasureType != 'time' else 'measureTime_' + 'icarl' + str(budget_percent)
    ## rev
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    args_Dn_l = []

    # for clsSeq, class_D2_l in enumerate(class_D2_list):
    clsSeq = trial-1
    class_D2_l = class_D2_list[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(class_D2_list)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.class_D2_l = class_D2_l
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, class_D2_l=class_D2_l, subject_idx=subject_idx,exp_setup=exp_setup,
        args=args)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    cuda = torch.cuda.is_available() and args.cuda

    if (quantize < 16):
        args.quantizer = Quantizer()
        args.quantizer.gatherStats(train_datasets[0], cuda_num, quantize)

    args.D1orD2 = 1

    args_D1 = {}
    acc_D1_max = -0.1
    args.hp = 0
    args.clsSeq = clsSeq+1


    fc_lay_l,fc_units_l,lr_l,lr2_l,batch_l,ewc_lambda_l,budget_size_l=get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 200
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    args.D1orD2 = 1
    args.hp = 1
    # setup args for experiment D1
    print("args fc_lay: %d fc_units: %d lr: %f fc_nl: %s batch: %d" % (fc_lay, fc_units, lr, fc_nl, batch))
    iter_per_epoch = int(len(train_datasets[0])/batch+1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    args.set_params_D12_icarl(lr, batch, budget, herding, parallel)
    model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
    precision_dict_list[0] = precision_dict_exemplars

    path_best_model_init = '../data/saved_model/'+exp_setup+'icarl_weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                        + '_hp'+str(args.hp)+'.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict_exemplars['per_task_weighted_f1'][0][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    saved_model_epoch_l.append(0)
    ##### task 2 #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.D1orD2 = 2
    args.hp = 0
    best_hp = 0
    for lr2 in lr2_l:
        args.hp += 1
        print("args lr2: %f budget: %d, batch: %d" %(lr2, budget, batch))
        # iter_per_epoch = int(len(train_datasets[1])/batch+1)
        # total_iters = iter_per_epoch * epoch
        if augment < 0:
            iter_per_epoch = int(len(train_datasets[1]) / batch + 1)
            total_iters = iter_per_epoch * epoch
        else:
            # compute dynamic batch size
            args.batch = batch
            args.batch_new_task = int(len(train_datasets[1])/(budget*scaling+len(train_datasets[1]))*batch)
            args.batch_exemplars = int((batch - args.batch_new_task)/scaling + 1)

            iter_per_epoch = int(len(train_datasets[1]) / args.batch_new_task + 1)
            total_iters = iter_per_epoch * epoch
            args.sd = sd

        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        args.set_params_D12_icarl(lr2, batch, budget, herding, parallel, augment, scaling)
        # train on D1 with args_D1 or load model_D1
        model2 = copy.deepcopy(model)
        # train on D2
        model2,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model2)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/'+exp_setup+'icarl_weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                    + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                    + '_hp'+str(args.hp)+'.pth'
        model2 = torch.load(path_best_model)
        precision_dict_list[1] = precision_dict_exemplars
        saved_model_epoch_l[1] = model2.epoch

        acc_Dn_temp = precision_dict_exemplars['all_tasks_weighted_f1'][model2.epoch-1]
        # compare the best accuracy on D1 U D2, D2, D1 // store the arguments to args_D2 and store the best model
        if acc_Dn_temp > acc_Dn_max:
            acc_Dn_max = acc_Dn_temp
            best_hp = args.hp

        ##### result reporting
        args_Dn['cl_alg'] = args.cl_alg
        args_Dn['budget_percent'] = budget_percent
        args_Dn['budget'] = budget
        args_Dn['scenario'] = getExpType(args)
        args_Dn['dataset'] = args.dataset
        args_Dn['fc_lay'] = fc_lay
        args_Dn['fc_units'] = fc_units
        args_Dn['lr'] = lr
        args_Dn['lr2'] = lr2
        args_Dn['batch'] = batch
        args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
        args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
        args_Dn['weighted_f1'] = acc_Dn_temp
        args_Dn['hp'] = args.hp
        args_Dn['clsSeq'] = clsSeq
        args_Dn_l.append(copy.copy(args_Dn))

        print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_percent: %d, budget: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,budget_percent,budget,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
        print('saved_model_epoch_l: ', saved_model_epoch_l)
        print("acc_Dn_temp: ", acc_Dn_temp)
        print("acc_Dn_max: ", acc_Dn_max)
        acc_Dn_temp_l.append(acc_Dn_temp)
        precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass
    # else:
    #     pass
        # plot_all_metrics(precision_dict_list, epoch, args)
        # precision_dict_list_list[clsSeq] = precision_dict_list[:]
        # args_Dn_l_l[clsSeq] = copy.copy(args_Dn_l)
    print('precision_dict_list_list: ', precision_dict_list_list)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    print('Best precision_dict_list', precision_dict_list_list[best_hp - 1])
    # if rMeasureType != 'time':
        # plot_all_metrics_summed(precision_dict_list_list, epoch, args)
        # rev_plot_all_metrics_summed(precision_dict_list_list, epoch, args, args_Dn_l)
    reportTimeMem(model, args, args.cl_alg)
    print('args_l: ', args_Dn_l)
    print('acc_Dn_temp_l: ', acc_Dn_temp_l)
    print('Best args', args_Dn_l[best_hp - 1])
    print('Best HP idx: {}, acc_D1_max: {}, acc_Dn_max: {}'.format(best_hp, acc_D1_max, acc_Dn_max))



def run_exp3_icarl_from_t0_permu(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='', 
    herding=1,parallel=1,quantize=32, budget_percent=5, augment=-1,scaling=1,sd=1.0, max_num_cases=1,
    n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names ##### 
    input_class_seq_l = makeClassSeqList(class_D2_list=rClassD2List, rDataset=rDataset, max_num_cases=max_num_cases)

    experiment = 'sensor'
    scenario = 'class'
    optimizer = 'adam'
    fc_nl_l = ['relu']
    datasets_l = ['hhar-noaug', 'pamap2', 'skoda', 'opp_thomas']

    ##### n_tasks #####
    num_tasks = len(input_class_seq_l[0])
    args = MyArgs(cuda=True, cuda_num=cuda_num,parallel=parallel,quantize=quantize) if cuda_num >= 0 else MyArgs(cuda=False,parallel=parallel,quantize=quantize)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks,
                            experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'icarl' + str(budget_percent) if rEpoch != 1 or rMeasureType != 'time' else 'measureTime_' + 'icarl' + str(budget_percent)

    # precision_dict_list_list = [[{} for i in range(num_tasks)] for j in range(len(input_class_seq_l))]
    # rev
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    # avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(len(input_class_seq_l))]
    avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(1)]
    args_Dn_l = []

    # for clsSeq, input_class_seq in enumerate(input_class_seq_l):
    clsSeq = trial - 1
    input_class_seq = input_class_seq_l[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(input_class_seq_l)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.input_class_seq = input_class_seq
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment_multi_tasks(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, input_class_seq=input_class_seq, subject_idx=subject_idx,exp_setup=exp_setup,
        args=args)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    # stats = Quantizer.gatherStats(train_datasets, args.device)
    # print(stats)
    cuda = torch.cuda.is_available() and args.cuda

    if (quantize < 16):
        args.quantizer = Quantizer()
        args.quantizer.gatherStats(train_datasets[0], cuda_num, quantize)

    args.D1orD2 = 1


    args.clsSeq = clsSeq + 1 # clsSeq is the saem as trial

    fc_lay_l, fc_units_l, lr_l, lr2_l, batch_l, ewc_lambda_l, budget_size_l = get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 20 if exp_setup == 'per-subject' or rDataset[:7] == 'emotion' else 5
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    # initialize variables to start again from task 0
    args.D1orD2 = 1
    args.hp = get_icarl_hp_num(n_layers, n_units, budget_percent)

    # setup args for experiment D1
    iter_per_epoch = int(len(train_datasets[0]) / batch + 1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    args.set_params_D12_icarl(lr, batch, budget, herding, parallel)
    model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                 test_total_dataset, None)
    if rMeasureType == 'time':
            putTime2List(args)
    precision_dict_list[0] = precision_dict_exemplars

    path_best_model_init = '../data/saved_model/' + exp_setup + 'icarl_weighted_best_' + getExpType(
        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                           + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                           + '_hp' + str(args.hp) + '.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict_exemplars['all_tasks_weighted_f1'][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    acc_l = [acc_D1_max]
    # saved_model_epoch_l.append(0)

    ##### task 2 to n #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    best_hp = 0
    lr2 = lr2_l[0]

    for task_idx in range(1, num_tasks):
        args.D1orD2 = task_idx + 1  # start from 2
        if augment < 0:
            iter_per_epoch = int(len(train_datasets[task_idx]) / batch + 1)
            total_iters = iter_per_epoch * epoch
        else:
            # compute dynamic batch size
            args.batch = batch
            args.batch_new_task = int(len(train_datasets[task_idx])/(budget*scaling+len(train_datasets[task_idx]))*batch)
            args.batch_exemplars = int((batch - args.batch_new_task)/scaling + 1)

            iter_per_epoch = int(len(train_datasets[task_idx]) / args.batch_new_task + 1)
            total_iters = iter_per_epoch * epoch
            args.sd = sd

        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        args.set_params_D12_icarl(lr2, batch, budget, herding, parallel, augment, scaling)
        model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                      test_total_dataset, model)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/' + exp_setup + 'icarl_weighted_best_' + getExpType(
            args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                          + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                          + '_hp' + str(args.hp) + '.pth'
        model = torch.load(path_best_model)
        precision_dict_list[task_idx] = precision_dict_exemplars
        acc_Dn_temp = precision_dict_exemplars['all_tasks_weighted_f1'][model.epoch - 1]
        acc_l.append(acc_Dn_temp)
        print('len(precision_dict_exemplars[1][all_tasks_acc]): ', len(precision_dict_exemplars['all_tasks_acc']))
        print('saved model.epoch: ', model.epoch)
        saved_model_epoch_l.append(model.epoch)


    # compare with the best accuracy, store the arguments to args_Dn and store the best model
    if acc_Dn_temp > acc_Dn_max:
        acc_Dn_max = acc_Dn_temp
        best_hp = args.hp

    ##### result reporting
    args_Dn['cl_alg'] = args.cl_alg
    args_Dn['budget_percent'] = budget_percent
    args_Dn['budget'] = budget
    args_Dn['scenario'] = getExpType(args)
    args_Dn['dataset'] = args.dataset
    args_Dn['fc_lay'] = fc_lay
    args_Dn['fc_units'] = fc_units
    args_Dn['lr'] = lr
    args_Dn['lr2'] = lr2
    args_Dn['batch'] = batch
    args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
    args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
    args_Dn['weighted_f1'] = acc_Dn_temp
    args_Dn['hp'] = args.hp
    args_Dn['clsSeq'] = clsSeq
    args_Dn_l.append(copy.copy(args_Dn))

    print(
        "args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_percent: %d, budget: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (
        fc_lay, fc_units, lr, lr2, batch, budget_percent, budget, args.hp, np.sum(saved_model_epoch_l), acc_Dn_temp))
    print('saved_model_epoch_l: ', saved_model_epoch_l)
    print("acc_Dn_temp: ", acc_Dn_temp)
    print("acc_Dn_max: ", acc_Dn_max)
    acc_Dn_temp_l.append(acc_Dn_temp)
    precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    path_best_model = '../data/saved_model/'+exp_setup+'icarl_weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'
    print('precision_dict_list_list: ', precision_dict_list_list)
    model = torch.load(path_best_model)
    # print('precision_dict_list_list_best_hp: ', precision_dict_list_list[best_hp - 1])

    # compute (1) average performance and (2) average forgetting w.r.t. many metrics (acc, f1, prec, rec)
    # compute (3) Intransigence=a* - a_{k,k} = but we compute a_{k,k} for now.
    avgAccForgetAkk_l[0] = computeAvgAccForgetAkk(model, test_datasets, precision_dict_list_list[0], args)
    # print('\nsaved_model epoch: {} / best HP idx: {} / a_kk: {}'.format(model.epoch, best_hp, acc_Dn_max))
    # print("args_D2: ", args_D2)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass
    reportTimeMem(model, args, args.cl_alg)
    print(path_best_model)
    print('best_saved_model epoch sum: {} and best HP idx: {}'.format(np.sum(args_Dn['saved_model_epoch_l']), best_hp))
    print("best_args_Dn: ", args_Dn)
    print("acc_list: ", acc_l)
    print("best_acc_Dn: ", acc_Dn_max)






def run_exp12_lwf(rClassD2List,rDataset,rMeasureType,rEpoch,cuda_num,exp_type=None,subject_idx=None,exp_setup='',
                  n_layers=1,n_units=32,lr2=0.001,batch_size=32,trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names #####
    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu']
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    class_D2_list = rClassD2List
    ##### n_tasks #####
    num_tasks = 2
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks,
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'lwf' if rEpoch != 1 or rMeasureType != 'time' else 'measureTime_' + 'lwf'
    args.rMeasureType = rMeasureType
    ## rev
    precision_dict_list_list = []
    args_Dn_l = []

    # for clsSeq, class_D2_l in enumerate(class_D2_list):
    clsSeq = trial-1
    class_D2_l = class_D2_list[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(class_D2_list)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.class_D2_l = class_D2_l
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, class_D2_l=class_D2_l, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args_D1 = {}
    acc_D1_max = -0.1
    args.hp = 0
    args.clsSeq = clsSeq+1


    fc_lay_l,fc_units_l,lr_l,lr2_l,batch_l,ewc_lambda_l,budget_size_l=get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 200
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    batch = batch_l[0]
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]

    ##### Task 1 #####
    args.D1orD2 = 1
    args.hp = 1
    # setup args for experiment D1
    print("args fc_lay: %d fc_units: %d lr: %f fc_nl: %s batch: %d" % (fc_lay, fc_units, lr, fc_nl, batch))
    iter_per_epoch = int(len(train_datasets[0])/batch+1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                        + '_hp'+str(args.hp)+'.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['per_task_weighted_f1'][0][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    saved_model_epoch_l.append(0)
    ##### task 2 #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.D1orD2 = 2
    args.hp = 0
    best_hp = 0
    for lr2 in lr2_l:
        args.hp += 1
        print("args lr2: %f batch: %d" %(lr2, batch))
        iter_per_epoch = int(len(train_datasets[1])/batch+1)
        total_iters = iter_per_epoch * epoch
        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        args.set_params_D2_lwf(lr2, batch)
        # train on D1 with args_D1 or load model_D1
        model2 = copy.deepcopy(model)
        # train on D2
        model2,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model2)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                    + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                    + '_hp'+str(args.hp)+'.pth'
        model2 = torch.load(path_best_model)
        precision_dict_list[1] = precision_dict
        saved_model_epoch_l[1] = model2.epoch

        acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model2.epoch-1]
        # compare the best accuracy on D1 U D2, D2, D1 // store the arguments to args_D2 and store the best model
        if acc_Dn_temp > acc_Dn_max:
            acc_Dn_max = acc_Dn_temp
            best_hp = args.hp

        ##### result reporting
        args_Dn['cl_alg'] = args.cl_alg
        args_Dn['budget_size'] = 0
        args_Dn['scenario'] = getExpType(args)
        args_Dn['dataset'] = args.dataset
        args_Dn['fc_lay'] = fc_lay
        args_Dn['fc_units'] = fc_units
        args_Dn['lr'] = lr
        args_Dn['lr2'] = lr2
        args_Dn['batch'] = batch
        args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
        args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
        args_Dn['weighted_f1'] = acc_Dn_temp
        args_Dn['hp'] = args.hp
        args_Dn['clsSeq'] = clsSeq
        args_Dn_l.append(copy.copy(args_Dn))

        print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
        print('saved_model_epoch_l: ', saved_model_epoch_l)
        print("acc_Dn_temp: ", acc_Dn_temp)
        print("acc_Dn_max: ", acc_Dn_max)
        acc_Dn_temp_l.append(acc_Dn_temp)
        precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass
    # else:
    #     pass
        # plot_all_metrics(precision_dict_list, epoch, args)
        # precision_dict_list_list[clsSeq] = precision_dict_list[:]
        # args_Dn_l_l[clsSeq] = copy.copy(args_Dn_l)
    print('precision_dict_list_list: ', precision_dict_list_list)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    print('Best precision_dict_list', precision_dict_list_list[best_hp - 1])
    # if rMeasureType != 'time':
        # plot_all_metrics_summed(precision_dict_list_list, epoch, args)
        # rev_plot_all_metrics_summed(precision_dict_list_list, epoch, args, args_Dn_l)
    reportTimeMem(model, args, args.cl_alg)
    print('args_l: ', args_Dn_l)
    print('acc_Dn_temp_l: ', acc_Dn_temp_l)
    print('Best args', args_Dn_l[best_hp - 1])
    print('Best HP idx: {}, acc_D1_max: {}, acc_Dn_max: {}'.format(best_hp, acc_D1_max, acc_Dn_max))





def run_exp3_lwf_from_t0_permu(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='', max_num_cases=1,
                               n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names ##### 
    input_class_seq_l=makeClassSeqList(class_D2_list=rClassD2List,rDataset=rDataset,max_num_cases=max_num_cases)

    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu'] 
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    ##### n_tasks #####
    num_tasks = len(input_class_seq_l[0])
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks, 
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'lwf' if rEpoch!=1 or rMeasureType != 'time' else 'measureTime_'+'lwf'
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(1)]
    args_Dn_l = []

    # for clsSeq, input_class_seq in enumerate(input_class_seq_l):
    clsSeq = trial - 1
    input_class_seq = input_class_seq_l[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(input_class_seq_l)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.input_class_seq = input_class_seq
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment_multi_tasks(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, input_class_seq=input_class_seq, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args.clsSeq = clsSeq + 1 # clsSeq is the saem as trial

    fc_lay_l, fc_units_l, lr_l, lr2_l, batch_l, ewc_lambda_l, budget_size_l = get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 20 if exp_setup == 'per-subject' or rDataset[:7] == 'emotion' else 5
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    # budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    # initialize variables to start again from task 0
    args.D1orD2 = 1
    args.hp = get_ewc_hp_num(n_layers, n_units)

    # setup args for experiment D1
    iter_per_epoch = int(len(train_datasets[0]) / batch + 1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                 test_total_dataset, None)
    if rMeasureType == 'time':
            putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                           + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                           + '_hp' + str(args.hp) + '.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    acc_l = [acc_D1_max]


    ##### task 2 to n #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.hp = 0
    best_hp = 0
    lr2 = lr2_l[0]


    args.hp += 1
    if args.hp > 1:
        model = torch.load(path_best_model_init)
    print("\n===== initialize from t1: args lr2: %f batch: %d" %(lr2,batch))
    
    for task_idx in range(1, num_tasks):
        args.D1orD2 = task_idx + 1  # start from 2
        iter_per_epoch = int(len(train_datasets[task_idx]) / batch + 1)
        total_iters = iter_per_epoch * epoch
        args.set_params_train(iters=total_iters)
        args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
        args.set_params_D2_lwf(lr2, batch)
        # train on D1 with args_D1 or load model_D1 / train on D2
        model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model)
        if rMeasureType == 'time':
            putTime2List(args)
        path_best_model = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
                args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                              + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                              + '_hp' + str(args.hp) + '.pth'
        model = torch.load(path_best_model)
        precision_dict_list[task_idx] = precision_dict
        acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
        acc_l.append(acc_Dn_temp)
        print('len(precision_dict[1][all_tasks_acc]): ', len(precision_dict['all_tasks_acc']))
        print('saved model.epoch: ', model.epoch)
        saved_model_epoch_l.append(model.epoch)

    # compare with the best accuracy, store the arguments to args_Dn and store the best model
    if acc_Dn_temp > acc_Dn_max:
        acc_Dn_max = acc_Dn_temp
        best_hp = args.hp

    ##### result reporting
    args_Dn['cl_alg'] = args.cl_alg
    args_Dn['budget_size'] = 0
    args_Dn['scenario'] = getExpType(args)
    args_Dn['dataset'] = args.dataset
    args_Dn['fc_lay'] = fc_lay
    args_Dn['fc_units'] = fc_units
    args_Dn['lr'] = lr
    args_Dn['lr2'] = lr2
    args_Dn['batch'] = batch
    args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
    args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
    args_Dn['weighted_f1'] = acc_Dn_temp
    args_Dn['hp'] = args.hp
    args_Dn['clsSeq'] = clsSeq
    args_Dn_l.append(copy.copy(args_Dn))

    print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
    print('saved_model_epoch_l: ', saved_model_epoch_l)
    print("acc_Dn_temp: ", acc_Dn_temp)
    print("acc_Dn_max: ", acc_Dn_max)
    acc_Dn_temp_l.append(acc_Dn_temp)
    precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'

    print('precision_dict_list_list: ', precision_dict_list_list)
    model = torch.load(path_best_model)
    # print('precision_dict_list_list_best_hp: ', precision_dict_list_list[best_hp - 1])

    # compute (1) average performance and (2) average forgetting w.r.t. many metrics (acc, f1, prec, rec)
    # compute (3) Intransigence=a* - a_{k,k} = but we compute a_{k,k} for now.
    avgAccForgetAkk_l[0] = computeAvgAccForgetAkk(model, test_datasets, precision_dict_list_list[best_hp - 1], args)
    # print('\nsaved_model epoch: {} / best HP idx: {} / a_kk: {}'.format(model.epoch, best_hp, acc_Dn_max))
    # print("args_D2: ", args_D2)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass

    reportTimeMem(model, args, args.cl_alg)
    print(path_best_model)
    print('best_saved_model epoch sum: {} and best HP idx: {}'.format(np.sum(args_Dn['saved_model_epoch_l']), best_hp))
    print("best_args_Dn: ", args_Dn)
    print("acc_list: ", acc_l)
    print("best_acc_Dn: ", acc_Dn_max)








def run_exp12_ewc(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='',fisher_n=None,
                  n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names #####
    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu'] 
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    class_D2_list = rClassD2List
    ##### n_tasks #####
    num_tasks = 2
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks, 
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'ewc' if rEpoch!=1 or rMeasureType != 'time' else 'measureTime_'+'ewc'
    args.rMeasureType = rMeasureType
    ## rev
    precision_dict_list_list = []
    args_Dn_l = []

    # for clsSeq, class_D2_l in enumerate(class_D2_list):
    clsSeq = trial-1
    class_D2_l = class_D2_list[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(class_D2_list)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.class_D2_l = class_D2_l
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, class_D2_l=class_D2_l, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args_D1 = {}
    acc_D1_max = -0.1
    args.hp = 0
    args.clsSeq = clsSeq+1



    fc_lay_l,fc_units_l,lr_l,lr2_l,batch_l,ewc_lambda_l,budget_size_l=get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 200
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    batch = batch_l[0]
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]

    ##### Task 1 #####
    args.D1orD2 = 1
    args.hp = 1
    # setup args for experiment D1
    print("args fc_lay: %d fc_units: %d lr: %f fc_nl: %s batch: %d" % (fc_lay, fc_units, lr, fc_nl, batch))
    iter_per_epoch = int(len(train_datasets[0])/batch+1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                        + '_hp'+str(args.hp)+'.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['per_task_weighted_f1'][0][model.epoch-1]
    saved_model_epoch_l = [model.epoch]
    saved_model_epoch_l.append(0)
    ##### task 2 #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.D1orD2 = 2
    args.hp = 0
    best_hp = 0
    for lr2 in lr2_l:
        for ewc_lambda in ewc_lambda_l:
            args.hp += 1
            print("args lr2: %f ewc_lambda: %d batch: %d" %(lr2, ewc_lambda, batch))
            iter_per_epoch = int(len(train_datasets[1])/batch+1)
            total_iters = iter_per_epoch * epoch
            args.set_params_train(iters=total_iters)
            args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
            args.set_params_D2_ewc(lr2, ewc_lambda, batch)
            # train on D1 with args_D1 or load model_D1
            model2 = copy.deepcopy(model)
            # train on D2
            model2,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model2)
            if rMeasureType == 'time':
                putTime2List(args)
            path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                        + '_hp'+str(args.hp)+'.pth'
            model2 = torch.load(path_best_model)
            precision_dict_list[1] = precision_dict
            saved_model_epoch_l[1] = model2.epoch

            acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model2.epoch-1]
            # compare the best accuracy on D1 U D2, D2, D1 // store the arguments to args_D2 and store the best model
            if acc_Dn_temp > acc_Dn_max:
                acc_Dn_max = acc_Dn_temp
                best_hp = args.hp

            ##### result reporting
            args_Dn['cl_alg'] = args.cl_alg
            args_Dn['budget_size'] = 0
            args_Dn['scenario'] = getExpType(args)
            args_Dn['dataset'] = args.dataset
            args_Dn['fc_lay'] = fc_lay
            args_Dn['fc_units'] = fc_units
            args_Dn['lr'] = lr
            args_Dn['lr2'] = lr2
            args_Dn['batch'] = batch
            args_Dn['ewc_lambda'] = ewc_lambda
            args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
            args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
            args_Dn['weighted_f1'] = acc_Dn_temp
            args_Dn['hp'] = args.hp
            args_Dn['clsSeq'] = clsSeq
            args_Dn_l.append(copy.copy(args_Dn))

            print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, ewc_lambda: %f, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,ewc_lambda,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
            print('saved_model_epoch_l: ', saved_model_epoch_l)
            print("acc_Dn_temp: ", acc_Dn_temp)
            print("acc_Dn_max: ", acc_Dn_max)
            acc_Dn_temp_l.append(acc_Dn_temp)
            precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    # if rMeasureType == 'time':
    #     if fisher_n is not None:
    #         plot_all_metrics(precision_dict_list, epoch, args)
    #     putTime2List(args)
    #     pass
    # else:
    #     pass
        # plot_all_metrics(precision_dict_list, epoch, args)
        # precision_dict_list_list[clsSeq] = precision_dict_list[:]
        # args_Dn_l_l[clsSeq] = copy.copy(args_Dn_l)
    print('precision_dict_list_list: ', precision_dict_list_list)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    print('Best precision_dict_list', precision_dict_list_list[best_hp - 1])
    # if rMeasureType != 'time':
        # plot_all_metrics_summed(precision_dict_list_list, epoch, args)
        # rev_plot_all_metrics_summed(precision_dict_list_list, epoch, args, args_Dn_l)
    reportTimeMem(model, args, args.cl_alg)
    print('args_l: ', args_Dn_l)
    print('acc_Dn_temp_l: ', acc_Dn_temp_l)
    print('Best args', args_Dn_l[best_hp - 1])
    print('Best HP idx: {}, acc_D1_max: {}, acc_Dn_max: {}'.format(best_hp, acc_D1_max, acc_Dn_max))



def run_exp3_ewc_from_t0_permu(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='', max_num_cases=1, fisher_n=None,
                               n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names ##### 
    input_class_seq_l=makeClassSeqList(class_D2_list=rClassD2List,rDataset=rDataset,max_num_cases=max_num_cases)

    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu'] 
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    ##### n_tasks #####
    num_tasks = len(input_class_seq_l[0])
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks, 
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'ewc' if rEpoch!=1 or rMeasureType != 'time' else 'measureTime_'+'ewc'
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(1)]
    args_Dn_l = []

    # for clsSeq, input_class_seq in enumerate(input_class_seq_l):
    clsSeq = trial - 1
    input_class_seq = input_class_seq_l[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(input_class_seq_l)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.input_class_seq = input_class_seq
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment_multi_tasks(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, input_class_seq=input_class_seq, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args.clsSeq = clsSeq + 1 # clsSeq is the saem as trial

    fc_lay_l, fc_units_l, lr_l, lr2_l, batch_l, ewc_lambda_l, budget_size_l = get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 20 if exp_setup == 'per-subject' or rDataset[:7] == 'emotion' else 5
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    # budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    # initialize variables to start again from task 0
    args.D1orD2 = 1
    args.hp = get_ewc_hp_num(n_layers, n_units)

    # setup args for experiment D1
    iter_per_epoch = int(len(train_datasets[0]) / batch + 1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                 test_total_dataset, None)
    if rMeasureType == 'time':
            putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                           + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                           + '_hp' + str(args.hp) + '.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    acc_l = [acc_D1_max]


    ##### task 2 to n #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.hp = 0
    best_hp = 0
    lr2 = lr2_l[0]
    for ewc_lambda in ewc_lambda_l:
        args.hp += 1
        if args.hp > 1:
            model = torch.load(path_best_model_init)
        print("\n===== initialize from t1: args lr2: %f batch: %d ewc_lambda: %f " %(lr2,batch,ewc_lambda))
        
        for task_idx in range(1, num_tasks):
            args.D1orD2 = task_idx + 1  # start from 2
            iter_per_epoch = int(len(train_datasets[task_idx]) / batch + 1)
            total_iters = iter_per_epoch * epoch
            args.set_params_train(iters=total_iters)
            args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
            args.set_params_D2_ewc(lr2, ewc_lambda, batch)
            # train on D1 with args_D1 or load model_D1 / train on D2
            model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model)
            if rMeasureType == 'time':
                putTime2List(args)
            path_best_model = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
                    args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                                  + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                  + '_hp' + str(args.hp) + '.pth'
            model = torch.load(path_best_model)
            precision_dict_list[task_idx] = precision_dict
            acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
            acc_l.append(acc_Dn_temp)
            print('len(precision_dict[1][all_tasks_acc]): ', len(precision_dict['all_tasks_acc']))
            print('saved model.epoch: ', model.epoch)
            saved_model_epoch_l.append(model.epoch)

        # compare with the best accuracy, store the arguments to args_Dn and store the best model
        if acc_Dn_temp > acc_Dn_max:
            acc_Dn_max = acc_Dn_temp
            best_hp = args.hp

        ##### result reporting
        args_Dn['cl_alg'] = args.cl_alg
        args_Dn['budget_size'] = 0
        args_Dn['scenario'] = getExpType(args)
        args_Dn['dataset'] = args.dataset
        args_Dn['fc_lay'] = fc_lay
        args_Dn['fc_units'] = fc_units
        args_Dn['lr'] = lr
        args_Dn['lr2'] = lr2
        args_Dn['batch'] = batch
        args_Dn['ewc_lambda'] = ewc_lambda
        args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
        args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
        args_Dn['weighted_f1'] = acc_Dn_temp
        args_Dn['hp'] = args.hp
        args_Dn['clsSeq'] = clsSeq
        args_Dn_l.append(copy.copy(args_Dn))

        print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, ewc_lambda: %f, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,ewc_lambda,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
        print('saved_model_epoch_l: ', saved_model_epoch_l)
        print("acc_Dn_temp: ", acc_Dn_temp)
        print("acc_Dn_max: ", acc_Dn_max)
        acc_Dn_temp_l.append(acc_Dn_temp)
        precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'

    print('precision_dict_list_list: ', precision_dict_list_list)
    model = torch.load(path_best_model)
    # print('precision_dict_list_list_best_hp: ', precision_dict_list_list[best_hp - 1])

    # compute (1) average performance and (2) average forgetting w.r.t. many metrics (acc, f1, prec, rec)
    # compute (3) Intransigence=a* - a_{k,k} = but we compute a_{k,k} for now.
    avgAccForgetAkk_l[0] = computeAvgAccForgetAkk(model, test_datasets, precision_dict_list_list[best_hp - 1], args)
    # print('\nsaved_model epoch: {} / best HP idx: {} / a_kk: {}'.format(model.epoch, best_hp, acc_Dn_max))
    # print("args_D2: ", args_D2)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass

    reportTimeMem(model, args, args.cl_alg)
    print(path_best_model)
    print('best_saved_model epoch sum: {} and best HP idx: {}'.format(np.sum(args_Dn['saved_model_epoch_l']), best_hp))
    print("best_args_Dn: ", args_Dn)
    print("acc_list: ", acc_l)
    print("best_acc_Dn: ", acc_Dn_max)







def run_exp3_ewc_online_from_t0_permu(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='', max_num_cases=1, fisher_n=None,
                                      n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names ##### 
    input_class_seq_l=makeClassSeqList(class_D2_list=rClassD2List,rDataset=rDataset,max_num_cases=max_num_cases)

    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu'] 
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    ##### n_tasks #####
    num_tasks = len(input_class_seq_l[0])
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks, 
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'ewc_online' if rEpoch!=1 or rMeasureType != 'time' else 'measureTime_'+'ewc_online'
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(1)]
    args_Dn_l = []

    # for clsSeq, input_class_seq in enumerate(input_class_seq_l):
    clsSeq = trial - 1
    input_class_seq = input_class_seq_l[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(input_class_seq_l)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.input_class_seq = input_class_seq
    args.exp_setup = exp_setup
    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment_multi_tasks(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, input_class_seq=input_class_seq, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args.clsSeq = clsSeq + 1 # clsSeq is the saem as trial

    fc_lay_l, fc_units_l, lr_l, lr2_l, batch_l, ewc_lambda_l, budget_size_l = get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 20 if exp_setup == 'per-subject' or rDataset[:7] == 'emotion' else 5
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    # budget = int(config['train_size'] / 100 * budget_percent)
    

    ##### Task 1 #####
    # initialize variables to start again from task 0
    args.D1orD2 = 1
    args.hp = get_ewc_hp_num(n_layers, n_units)

    # setup args for experiment D1
    iter_per_epoch = int(len(train_datasets[0]) / batch + 1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                 test_total_dataset, None)
    if rMeasureType == 'time':
            putTime2List(args)
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                           + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                           + '_hp' + str(args.hp) + '.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    acc_l = [acc_D1_max]


    ##### task 2 to n #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.hp = 0
    best_hp = 0
    lr2 = lr2_l[0]
    for ewc_lambda in ewc_online_lambda_l:
        for ewc_gamma in ewc_online_gamma_l:
            args.hp += 1
            if args.hp > 1:
                model = torch.load(path_best_model_init)
            print("\n===== initialize from t1: args lr2: %f batch: %d ewc_lambda: %f ewc_gamma: %f" %(lr2,batch,ewc_lambda, ewc_gamma))
            
            for task_idx in range(1, num_tasks):
                args.D1orD2 = task_idx + 1  # start from 2
                iter_per_epoch = int(len(train_datasets[task_idx]) / batch + 1)
                total_iters = iter_per_epoch * epoch
                args.set_params_train(iters=total_iters)
                args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
                args.set_params_D2_ewc_online(lr2, batch, ewc_lambda, ewc_gamma)
                # train on D1 with args_D1 or load model_D1 / train on D2
                model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model)
                if rMeasureType == 'time':
                    putTime2List(args)
                path_best_model = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
                        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                                      + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                      + '_hp' + str(args.hp) + '.pth'
                model = torch.load(path_best_model)
                precision_dict_list[task_idx] = precision_dict
                acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
                acc_l.append(acc_Dn_temp)
                print('len(precision_dict[1][all_tasks_acc]): ', len(precision_dict['all_tasks_acc']))
                print('saved model.epoch: ', model.epoch)
                saved_model_epoch_l.append(model.epoch)

            # compare with the best accuracy, store the arguments to args_Dn and store the best model
            if acc_Dn_temp > acc_Dn_max:
                acc_Dn_max = acc_Dn_temp
                best_hp = args.hp

            ##### result reporting
            args_Dn['cl_alg'] = args.cl_alg
            args_Dn['budget_size'] = 0
            args_Dn['scenario'] = getExpType(args)
            args_Dn['dataset'] = args.dataset
            args_Dn['fc_lay'] = fc_lay
            args_Dn['fc_units'] = fc_units
            args_Dn['lr'] = lr
            args_Dn['lr2'] = lr2
            args_Dn['batch'] = batch
            args_Dn['ewc_lambda'] = ewc_lambda
            args_Dn['ewc_gamma'] = ewc_gamma
            args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
            args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
            args_Dn['weighted_f1'] = acc_Dn_temp
            args_Dn['hp'] = args.hp
            args_Dn['clsSeq'] = clsSeq
            args_Dn_l.append(copy.copy(args_Dn))

            print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, ewc_lambda: %f, ewc_gamma: %f, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,ewc_lambda,ewc_gamma,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
            print('saved_model_epoch_l: ', saved_model_epoch_l)
            print("acc_Dn_temp: ", acc_Dn_temp)
            print("acc_Dn_max: ", acc_Dn_max)
            acc_Dn_temp_l.append(acc_Dn_temp)
            precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'

    print('precision_dict_list_list: ', precision_dict_list_list)
    model = torch.load(path_best_model)
    # print('precision_dict_list_list_best_hp: ', precision_dict_list_list[best_hp - 1])

    # compute (1) average performance and (2) average forgetting w.r.t. many metrics (acc, f1, prec, rec)
    # compute (3) Intransigence=a* - a_{k,k} = but we compute a_{k,k} for now.
    avgAccForgetAkk_l[0] = computeAvgAccForgetAkk(model, test_datasets, precision_dict_list_list[best_hp - 1], args)
    # print('\nsaved_model epoch: {} / best HP idx: {} / a_kk: {}'.format(model.epoch, best_hp, acc_Dn_max))
    # print("args_D2: ", args_D2)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass

    reportTimeMem(model, args, args.cl_alg)
    print(path_best_model)
    print('best_saved_model epoch sum: {} and best HP idx: {}'.format(np.sum(args_Dn['saved_model_epoch_l']), best_hp))
    print("best_args_Dn: ", args_Dn)
    print("acc_list: ", acc_l)
    print("best_acc_Dn: ", acc_Dn_max)







def run_exp12_si(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='',
                  n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names #####
    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu']
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    class_D2_list = rClassD2List
    ##### n_tasks #####
    num_tasks = 2
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks,
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'si' if rEpoch != 1 or rMeasureType != 'time' else 'measureTime_' + 'si'
    args.rMeasureType = rMeasureType
    ## rev
    precision_dict_list_list = []
    args_Dn_l = []

    # for clsSeq, class_D2_l in enumerate(class_D2_list):
    clsSeq = trial-1
    class_D2_l = class_D2_list[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(class_D2_list)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.class_D2_l = class_D2_l
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, class_D2_l=class_D2_l, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args_D1 = {}
    acc_D1_max = -0.1
    args.hp = 0
    args.clsSeq = clsSeq+1



    fc_lay_l,fc_units_l,lr_l,lr2_l,batch_l,ewc_lambda_l,budget_size_l=get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 200
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    batch = batch_l[0]
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]

    ##### Task 1 #####
    args.D1orD2 = 1
    args.hp = 1
    # setup args for experiment D1
    print("args fc_lay: %d fc_units: %d lr: %f fc_nl: %s batch: %d" % (fc_lay, fc_units, lr, fc_nl, batch))
    iter_per_epoch = int(len(train_datasets[0])/batch+1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    args.set_params_D1_si()
    model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
        si_c_l = [0.5]

    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                        + '_hp'+str(args.hp)+'.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['per_task_weighted_f1'][0][model.epoch-1]
    saved_model_epoch_l = [model.epoch]
    saved_model_epoch_l.append(0)
    ##### task 2 #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.D1orD2 = 2
    args.hp = 0
    best_hp = 0
    for lr2 in lr2_l:
        for si_c in si_c_l:
            for si_epsilon in si_epsilon_l:
                args.hp += 1
                print("args lr2: %f batch: %d" %(lr2, batch))
                iter_per_epoch = int(len(train_datasets[1])/batch+1)
                total_iters = iter_per_epoch * epoch
                args.set_params_train(iters=total_iters)
                args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
                args.set_params_D2_si(lr2, batch, si_c, si_epsilon)
                # train on D1 with args_D1 or load model_D1
                model2 = copy.deepcopy(model)
                # train on D2
                model2,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model2)
                if rMeasureType == 'time':
                    putTime2List(args)
                path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                            + '_hp'+str(args.hp)+'.pth'
                model2 = torch.load(path_best_model)
                precision_dict_list[1] = precision_dict
                saved_model_epoch_l[1] = model2.epoch

                acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model2.epoch-1]
                # compare the best accuracy on D1 U D2, D2, D1 // store the arguments to args_D2 and store the best model
                if acc_Dn_temp > acc_Dn_max:
                    acc_Dn_max = acc_Dn_temp
                    best_hp = args.hp

                ##### result reporting
                args_Dn['cl_alg'] = args.cl_alg
                args_Dn['budget_size'] = 0
                args_Dn['scenario'] = getExpType(args)
                args_Dn['dataset'] = args.dataset
                args_Dn['fc_lay'] = fc_lay
                args_Dn['fc_units'] = fc_units
                args_Dn['lr'] = lr
                args_Dn['lr2'] = lr2
                args_Dn['batch'] = batch
                args_Dn['si_c'] = si_c
                args_Dn['si_epsilon'] = si_epsilon
                args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
                args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
                args_Dn['weighted_f1'] = acc_Dn_temp
                args_Dn['hp'] = args.hp
                args_Dn['clsSeq'] = clsSeq
                args_Dn_l.append(copy.copy(args_Dn))

                print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, si_c: %f, si_epsilon: %f, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,si_c,si_epsilon,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
                print('saved_model_epoch_l: ', saved_model_epoch_l)
                print("acc_Dn_temp: ", acc_Dn_temp)
                print("acc_Dn_max: ", acc_Dn_max)
                acc_Dn_temp_l.append(acc_Dn_temp)
                precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass
    # else:
    #     pass
        # plot_all_metrics(precision_dict_list, epoch, args)
        # precision_dict_list_list[clsSeq] = precision_dict_list[:]
        # args_Dn_l_l[clsSeq] = copy.copy(args_Dn_l)
    print('precision_dict_list_list: ', precision_dict_list_list)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    print('Best precision_dict_list', precision_dict_list_list[best_hp - 1])
    # if rMeasureType != 'time':
        # plot_all_metrics_summed(precision_dict_list_list, epoch, args)
        # rev_plot_all_metrics_summed(precision_dict_list_list, epoch, args, args_Dn_l)
    reportTimeMem(model, args, args.cl_alg)
    print('args_l: ', args_Dn_l)
    print('acc_Dn_temp_l: ', acc_Dn_temp_l)
    print('Best args', args_Dn_l[best_hp - 1])
    print('Best HP idx: {}, acc_D1_max: {}, acc_Dn_max: {}'.format(best_hp, acc_D1_max, acc_Dn_max))



def run_exp3_si_from_t0_permu(rClassD2List, rDataset, rMeasureType,rEpoch, cuda_num,exp_type=None, subject_idx=None,exp_setup='', max_num_cases=1,
                              n_layers=1, n_units=32, lr2=0.001, batch_size=32, trial=1):
    ##### permutation arguments #####
    ##### fixed params used in the code with its names ##### 
    input_class_seq_l=makeClassSeqList(class_D2_list=rClassD2List,rDataset=rDataset,max_num_cases=max_num_cases)

    experiment='sensor'
    scenario='class'
    optimizer = 'adam'
    fc_nl_l = ['relu'] 
    datasets_l=['hhar-noaug', 'pamap2','skoda','opp_thomas']

    ##### n_tasks #####
    num_tasks = len(input_class_seq_l[0])
    args = MyArgs(cuda=True, cuda_num=cuda_num) if cuda_num >= 0 else MyArgs(cuda=False)
    args.set_params_dataset(dataset=rDataset, tasks=num_tasks, 
                           experiment='sensor', scenario='class', cls_type='lstm', seed=0)
    epoch = rEpoch
    args.cl_alg = 'si' if rEpoch!=1 or rMeasureType != 'time' else 'measureTime_'+'si'
    args.rMeasureType = rMeasureType
    precision_dict_list_list = []
    avgAccForgetAkk_l = [[{} for i in range(num_tasks)] for j in range(1)]
    args_Dn_l = []

    # for clsSeq, input_class_seq in enumerate(input_class_seq_l):
    clsSeq = trial - 1
    input_class_seq = input_class_seq_l[clsSeq]
    print('\n\n===== Class Sequence %d / %d =====\n\n' %(clsSeq+1, len(input_class_seq_l)))
    precision_dict_list = [{} for i in range(num_tasks)]
    # load dataset
    # Prepare data for chosen experiment
    args.input_class_seq = input_class_seq
    args.exp_setup = exp_setup

    (train_datasets,test_datasets,test_total_dataset),config,classes_per_task,num_classes_per_task_l,weights_per_class=get_multitask_experiment_multi_tasks(
        name=args.experiment, scenario=args.scenario, tasks=num_tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed == 0 else False, dataset=args.dataset, input_class_seq=input_class_seq, subject_idx=subject_idx,exp_setup=exp_setup)

    test_datasets.append(test_total_dataset)
    args.config = config
    args.exemplars_sets_indexes = [[] for i in range(config['classes'])]
    args.classes_per_task = classes_per_task
    args.num_classes_per_task_l = num_classes_per_task_l
    args.weights_per_class = weights_per_class
    args.D1orD2 = 1

    args.clsSeq = clsSeq + 1 # clsSeq is the saem as trial

    fc_lay_l, fc_units_l, lr_l, lr2_l, batch_l, ewc_lambda_l, budget_size_l = get_ewc_hparams(exp_setup,rMeasureType)

    saved_model_epoch_l = []
    patience = 20 if exp_setup == 'per-subject' or rDataset[:7] == 'emotion' else 5
    lr = lr_l[0]
    fc_nl = fc_nl_l[0]
    # batch = batch_l[0]
    batch = batch_size
    fc_lay = n_layers
    fc_units = n_units
    lr2_l = [lr2]
    # budget = int(config['train_size'] / 100 * budget_percent)

    ##### Task 1 #####
    # initialize variables to start again from task 0
    args.D1orD2 = 1
    args.hp = get_ewc_hp_num(n_layers, n_units)

    # setup args for experiment D1
    iter_per_epoch = int(len(train_datasets[0]) / batch + 1)
    total_iters = iter_per_epoch * epoch
    args.set_params_train(iters=total_iters)
    args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
    args.set_params_D1(fc_lay, fc_units, fc_nl, lr, batch)
    args.set_params_D1_si()
    model, precision_dict, precision_dict_exemplars = run_D1orD2(args, train_datasets, test_datasets,
                                                                 test_total_dataset, None)
    if rMeasureType == 'time':
        putTime2List(args)
        si_c_l = [0.5]
            
    precision_dict_list[0] = precision_dict

    path_best_model_init = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                           + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                           + '_hp' + str(args.hp) + '.pth'
    model = torch.load(path_best_model_init)
    acc_D1_max = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
    saved_model_epoch_l = [model.epoch]
    acc_l = [acc_D1_max]


    ##### task 2 to n #####
    args_Dn = {}
    acc_Dn_max = -0.1
    acc_Dn_temp_l = []
    args.hp = 0
    best_hp = 0
    lr2 = lr2_l[0]
    for si_epsilon in si_epsilon_l:
        for si_c in si_c_l:
            args.hp += 1
            if args.hp > 1:
                model = torch.load(path_best_model_init)
            print("\n===== initialize from t1: args lr2: %f si_c: %f si_epsilon: %f" %(lr2, si_c, si_epsilon))
            
            for task_idx in range(1, num_tasks):
                args.D1orD2 = task_idx + 1  # start from 2
                iter_per_epoch = int(len(train_datasets[task_idx]) / batch + 1)
                total_iters = iter_per_epoch * epoch
                args.set_params_train(iters=total_iters)
                args.set_params_eval(prec_log=iter_per_epoch, patience=patience)
                args.set_params_D2_si(lr2, batch, si_c, si_epsilon)            
                # train on D1 with args_D1 or load model_D1 / train on D2
                model,precision_dict,precision_dict_exemplars=run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model)
                if rMeasureType == 'time':
                    putTime2List(args)
                path_best_model = '../data/saved_model/' + exp_setup + 'weighted_best_' + getExpType(
                        args) + '_' + args.cls_type + '_' + args.cl_alg + '_' \
                                      + args.dataset + '_' + 'clsSeq' + str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                      + '_hp' + str(args.hp) + '.pth'
                model = torch.load(path_best_model)
                precision_dict_list[task_idx] = precision_dict
                acc_Dn_temp = precision_dict['all_tasks_weighted_f1'][model.epoch - 1]
                acc_l.append(acc_Dn_temp)
                print('len(precision_dict[1][all_tasks_acc]): ', len(precision_dict['all_tasks_acc']))
                print('saved model.epoch: ', model.epoch)
                saved_model_epoch_l.append(model.epoch)

            # compare with the best accuracy, store the arguments to args_Dn and store the best model
            if acc_Dn_temp > acc_Dn_max:
                acc_Dn_max = acc_Dn_temp
                best_hp = args.hp

            ##### result reporting
            args_Dn['cl_alg'] = args.cl_alg
            args_Dn['budget_size'] = 0
            args_Dn['scenario'] = getExpType(args)
            args_Dn['dataset'] = args.dataset
            args_Dn['fc_lay'] = fc_lay
            args_Dn['fc_units'] = fc_units
            args_Dn['lr'] = lr
            args_Dn['lr2'] = lr2
            args_Dn['batch'] = batch
            args_Dn['si_c'] = si_c
            args_Dn['si_epsilon'] = si_epsilon
            args_Dn['saved_model_epoch_l'] = saved_model_epoch_l
            args_Dn['total_epochs'] = np.sum(saved_model_epoch_l)
            args_Dn['weighted_f1'] = acc_Dn_temp
            args_Dn['hp'] = args.hp
            args_Dn['clsSeq'] = clsSeq
            args_Dn_l.append(copy.copy(args_Dn))

            print("args fc_lay: %d, fc_units: %d, lr: %f, lr2: %f, batch: %d, si_c: %f, si_epsilon: %f, budget_size: %d, args.hp: %d, total epochs: %d, weighted_f1: %f" % (fc_lay,fc_units,lr,lr2,batch,si_c,si_epsilon,0,args.hp,np.sum(saved_model_epoch_l),acc_Dn_temp))
            print('saved_model_epoch_l: ', saved_model_epoch_l)
            print("acc_Dn_temp: ", acc_Dn_temp)
            print("acc_Dn_max: ", acc_Dn_max)
            acc_Dn_temp_l.append(acc_Dn_temp)
            precision_dict_list_list.append(copy.copy(precision_dict_list))


    print("")
    print("=======================================")
    print("============ All Finished =============")
    print("=======================================")
    path_best_model = '../data/saved_model/'+exp_setup+'weighted_best_'+getExpType(args)+'_'+args.cls_type+'_'+args.cl_alg+'_' \
                                        + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                        + '_hp'+str(best_hp)+'.pth'

    print('precision_dict_list_list: ', precision_dict_list_list)
    model = torch.load(path_best_model)
    # print('precision_dict_list_list_best_hp: ', precision_dict_list_list[best_hp - 1])

    # compute (1) average performance and (2) average forgetting w.r.t. many metrics (acc, f1, prec, rec)
    # compute (3) Intransigence=a* - a_{k,k} = but we compute a_{k,k} for now.
    avgAccForgetAkk_l[0] = computeAvgAccForgetAkk(model, test_datasets, precision_dict_list_list[best_hp - 1], args)
    # print('\nsaved_model epoch: {} / best HP idx: {} / a_kk: {}'.format(model.epoch, best_hp, acc_Dn_max))
    # print("args_D2: ", args_D2)
    print("exemplars_sets_indexes: ", args.exemplars_sets_indexes)
    print("\n===== final summary of average metrics and forgetting =====\n")
    # if rMeasureType == 'time':
    #     putTime2List(args)
    #     pass

    reportTimeMem(model, args, args.cl_alg)
    print(path_best_model)
    print('best_saved_model epoch sum: {} and best HP idx: {}'.format(np.sum(args_Dn['saved_model_epoch_l']), best_hp))
    print("best_args_Dn: ", args_Dn)
    print("acc_list: ", acc_l)
    print("best_acc_Dn: ", acc_Dn_max)


