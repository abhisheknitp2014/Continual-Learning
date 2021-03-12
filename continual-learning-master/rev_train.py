import torch
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
import utils
import time
from data import SubDataset, ExemplarDataset
from continual_learner import ContinualLearner
from data import *
# from parallel_func import *
# import ray

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#####
##### Let's use this function only for 2 task SLP experiment
#####
def train_cl(model, train_datasets, replay_mode="none", scenario="class",classes_per_task=None,iters=2000,batch_size=32,
             generator=None, gen_iters=0, gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, eval_cbs_exemplars=list(),
             num_classes_per_task_l=None, experiment='sensor', config=None, args=None):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    ##### if n_tasks >= 3, then we are doing multi tasks
    n_tasks = len(train_datasets)

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (args.si == True):
        start_si = time.time()
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
        args.train_time_si_update += time.time() - start_si

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        if args.D1orD2 == 1:
            if task > 1:
                continue
        elif args.D1orD2 >= 2:
            if task != args.D1orD2:
                continue
            else:
                # +++++ we should do this in the beginning of each task +++++
                # +++++ since we change 'ewc_lambda' for each task +++++
                # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
                if isinstance(model, ContinualLearner) and (model.ewc_lambda > 0):
                    prev_task = task - 1
                    # change task to prev_task below
                    # -find allowed classes
                    if scenario == 'task':
                        allowed_classes = list(range(classes_per_task * (prev_task - 1), classes_per_task * prev_task))
                    else:
                        if scenario == 'class':
                            if experiment == 'sensor':
                                allowed_classes = list(range(sum(num_classes_per_task_l[:(prev_task)])))
                            else:
                                allowed_classes = list(range(classes_per_task * prev_task))
                        else:
                            allowed_classes = None

                    if model.mask_dict is not None:
                        model.apply_XdGmask(task=prev_task)
                    # -estimate FI-matrix
                    # train_datasets[prev_task-1] # cuz task's offset value is 1
                    time_ewc = time.time()
                    model.estimate_fisher(train_datasets[prev_task - 1], allowed_classes=allowed_classes)
                    args.train_time_ewc_update += time.time() - time_ewc


                # +++++ SI: calculate and update the normalized path integral +++++
                if isinstance(model, ContinualLearner) and (model.si_c > 0):
                    start_si = time.time()
                    model.update_omega(model.prev_W, model.epsilon)
                    args.train_time_si_update += time.time() - start_si


                # ++++ No 'class' scenario but still do this before training ++++
                # REPLAY: update source for replay
                # previous_model = copy.deepcopy(model).eval()
                if replay_mode == 'generative':
                    previous_model = copy.deepcopy(model).eval()
                    Generative = True
                    previous_generator = copy.deepcopy(
                        generator).eval() if generator is not None else previous_model
                elif replay_mode == 'current':
                    start_lwf = time.time()
                    previous_model = copy.deepcopy(model).eval()
                    Current = True
                    args.train_time_lwf_update += time.time() - start_lwf
                elif replay_mode in ('exemplars', 'exact'):
                    previous_model = copy.deepcopy(model).eval()
                    Exact = True
                    if replay_mode == "exact":
                        previous_datasets = train_datasets[:task]
                    else:
                        if scenario == "task":
                            previous_datasets = []
                            for task_id in range(task):
                                previous_datasets.append(
                                    ExemplarDataset(
                                        model.exemplar_sets[
                                        (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                        target_transform=lambda y, x=classes_per_task * task_id: y + x)
                                )
                        else:
                            target_transform = (
                                lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                            previous_datasets = [
                                ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]


        # ----- In our experiments, no offline replay setting -----
        # If offline replay-setting, create large database of all tasks so far
        if replay_mode=="offline" and (not scenario=="task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets


        # +++++ iCaRL has add_exemplars=True and use_exemplars=True +++++
        # Add exemplars (if available) to current dataset (if requested)
        if add_exemplars and use_exemplars and task>1:
            start_icarl = time.time()
            # ---------- ADHOC SOLUTION: permMNIST needs transform to tensor, while splitMNIST does not ---------- #
            if len(train_datasets)>10:
                target_transform = (lambda y, x=classes_per_task: torch.tensor(y%x)) if (
                        scenario=="domain"
                ) else (lambda y: torch.tensor(y))
            else:
                target_transform = (lambda y, x=classes_per_task: y%x) if scenario=="domain" else None
            # ---------------------------------------------------------------------------------------------------- #
            # print('train.py: target_transform: ', target_transform)

            exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=target_transform, quantize=args.quantize, args=args)
            if args.augment < 0:
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            args.train_time_icarl_construct += time.time() - start_icarl
        elif add_exemplars and use_exemplars and args.D1orD2==1:
        # else:
            training_dataset = train_dataset
            # print('task: ', task)
            model.init_exemplar_set(sum(num_classes_per_task_l), args.quantize)
            if args.parallel > 1:
                # print('init exemplars sets: num_classes_per_task_l: ', sum(num_classes_per_task_l))
                
                model.init_exemplar_dict()
        else:
            training_dataset = train_dataset

        # +++++ SI +++++
        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (args.si == True):
            start_si = time.time()
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()
            args.train_time_si_running_weights += time.time() - start_si




        ##### Find [active_classes] #####
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            if experiment == 'sensor': # for sensor datasets, # classes per task is different
                active_classes = list(range(sum(num_classes_per_task_l[:task])))
            else: # for MNIST dataset, the # classes per task is equal
                active_classes = list(range(classes_per_task * task))




        # ----- Reset state of optimizer(s) for every task (if requested) -----
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # ----- Initialize # iters left on current data-loader(s) -----
        iters_left = iters_left_previous = 1
        if scenario=="task":
            up_to_task = task if replay_mode=="offline" else task-1
            iters_left_previous = [1]*up_to_task
            data_loader_previous = [None]*up_to_task

        # ----- Define tqdm progress bar(s) -----
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))


        ###################################################
        # +++++ ENTERING POINT OF TRAINING ITERATIONS +++++
        # Loop over all iterations
        ###################################################
        best_f1 = -0.1
        best_acc = -0.1
        trials = 0
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)


        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                if args.augment < 0:
                    data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                else:
                    data_loader = iter(utils.get_data_loader(training_dataset, args.batch_new_task, cuda=cuda, drop_last=True))
                    data_loader_exemplars = iter(cycle(utils.get_data_loader(exemplar_dataset, args.batch_exemplars, cuda=cuda, drop_last=True)))

                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)
            if Exact: # ICARL doesn't go into this if statement
                if scenario=="task":
                    up_to_task = task if replay_mode=="offline" else task-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task>1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                train_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = scores = None
            else: # ICARL goes into here
                x, y = next(data_loader) #--> sample training data of current task                    
                y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                x_ex = y_ex = None
                if args.augment >= 0:
                    x_ex, y_ex = next(data_loader_exemplars)
                    x_ex, y_ex = x_ex.to(device), y_ex.to(device)

                # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and scenario=="class" and (previous_model is not None): # ICARL doesn't go into this since previous model is None
                    with torch.no_grad():
                        if experiment == 'sensor':
                            print('icarl')
                            scores = previous_model(x)[:, :(sum(num_classes_per_task_l[:(task-1)]))]
                        else:
                            scores = previous_model(x)[:, :(classes_per_task * (task - 1))]
                else:
                    scores = None


            #####-----REPLAYED BATCH-----#####
            if not Exact and not Generative and not Current:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Exact Replay <<--##
            if Exact:
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets=="hard") else None
                    # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                    if (model.replay_targets=="soft"):
                        with torch.no_grad():
                            scores_ = previous_model(x_)

                        # scores_ = scores_[:, :(classes_per_task*(task-1))] if scenario=="class" else scores_
                        # -> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                        if scenario == 'class':
                            if experiment == 'sensor':
                                scores_ = scores_[:, :(sum(num_classes_per_task_l[:(task-1)]))]
                            else:
                                scores_ = scores_[:, :(classes_per_task * (task - 1))]
                        else:
                            scores_ = scores_

                elif scenario=="task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets=="hard":
                            y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    # If required, get target scores (i.e, [scores_]         -- using previous model
                    if (model.replay_targets=="soft") and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            with torch.no_grad():
                                scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(scores_temp)


            ##-->> Generative / Current Replay <<--##
            if Generative or Current:
                start_lwf = time.time()
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                x_ = x if Current else previous_generator.sample(batch_size)

                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                    with torch.no_grad():
                        all_scores_ = previous_model(x_)
                # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                if scenario in ("domain", "class") and (
                        (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                ):

                    if scenario == 'class':
                        if experiment == 'sensor':
                            scores_ = all_scores_[:, :(sum(num_classes_per_task_l[:(task-1)]))]
                        else:
                            scores_ = all_scores_[:, :(classes_per_task * (task - 1))]
                    else:
                        scores_ = all_scores_
                    # scores_ = all_scores_[:,:(classes_per_task * (task - 1))] if scenario == "class" else all_scores_

                    _, y_ = torch.max(scores_, dim=1)
                else:
                    # NOTE: it's possible to have scenario=domain with task-mask (so actually it's the Task-IL scenario)
                    # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                    scores_ = list()
                    y_ = list()
                    for task_id in range(task - 1):
                        # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                        if hasattr(previous_model, "mask_dict") and previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(task=task_id + 1)
                            with torch.no_grad():
                                all_scores_ = previous_model(x_)
                        if scenario=="domain":
                            temp_scores_ = all_scores_
                        else:
                            temp_scores_ = all_scores_[:,
                                           (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        scores_.append(temp_scores_)
                        y_.append(temp_y_)

                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None
                args.train_time_lwf_loss += time.time() - start_lwf


            ################################
            #####---> Train MAIN MODEL #####
            if batch_index <= iters:

                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, x_ex=x_ex, y_ex=y_ex,
                                scores=scores, scores_=scores_,
                                active_classes=active_classes, num_classes_per_task_l=num_classes_per_task_l,
                                task=task, rnt = 1./task,args=args)
                # print('batch_index: ', batch_index, loss_dict)
                # Update running parameter importance estimates in W
                if isinstance(model, ContinualLearner) and (args.si == True):
                    start_si = time.time()
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()
                    args.train_time_si_running_weights += time.time() - start_si

                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                start_eval = time.time()
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                if not (add_exemplars or use_exemplars):
                    for eval_cb in eval_cbs:
                        if eval_cb is not None:
                            eval_cb(model, batch_index, task=task)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, task=task)
                args.eval_time += time.time() - start_eval

                # +++++ EXEMPLARS (iCaRL): update exemplar sets +++++
                # +++++ we can run this at the beginning of each task to update exemplars +++++
                # +++++ but then, we cannot see
                if (add_exemplars or use_exemplars) or replay_mode == "exemplars":
                    if batch_index % args.prec_log == 0:
                        # model = copy.deepcopy(model)
                        start_icarl = time.time()
                        if experiment == 'sensor':
                            exemplars_per_class = int(
                                np.floor(model.memory_budget / sum(num_classes_per_task_l[:(task)])))
                            model.reduce_exemplar_sets(exemplars_per_class)
                            if scenario == 'domain':
                                new_classes = list(range(classes_per_task))
                            else:
                                new_classes = list(range(sum(num_classes_per_task_l[:(task - 1)]),
                                                         sum(num_classes_per_task_l[:(task)])))
                                # print(num_classes_per_task_l)
                                # print(task)
                                # print('new_classes: ', new_classes)
                        else:
                            exemplars_per_class = int(np.floor(model.memory_budget / (classes_per_task * task)))
                            # reduce examplar-sets
                            model.reduce_exemplar_sets(exemplars_per_class)
                            # for each new class trained on, construct examplar-set
                            new_classes = list(range(classes_per_task)) if scenario == "domain" else list(
                                range(classes_per_task * (task - 1),
                                      classes_per_task * task))
                        args.train_time_icarl_reduce += time.time() - start_icarl
                        start_icarl = time.time()
                        if args.parallel <= 1 or task > 1:
                            for class_id in new_classes:
                                # start = time.time()
                                # create new dataset containing only all examples of this class
                                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                                # based on this dataset, construct new exemplar-set for this class
                                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class, class_id=class_id, args=args)
                                # print("Constructed exemplar-set for class {}: {} seconds".format(class_id, round(
                                #     time.time() - start)))
                        # elif args.parallel >= 2:
                        #     print('task: ', task)
                        #     print('rev_train: model.exemplar_sets: ', model.exemplar_sets)
                        #     model.eval()
                        #     if len(new_classes) >= args.parallel: # class parallel only
                        #         model_id = ray.put(model)
                        #         ray_results = ray.get([class_parallel.remote(model_id,i,new_classes,train_dataset,exemplars_per_class,args) for i in range(args.parallel)])
                        #         for exemplar_dict in ray_results:
                        #             for idx, k in exemplar_dict.items():
                        #                 print('rev_train: idx: ', idx)
                        #                 model.exemplar_sets.append(k)
                        #     elif len(new_classes) < args.parallel and args.herding == 3: # class parallel + construct parallel
                        #         model_id = ray.put(model)
                        #         ray_results = ray.get([construct_parallel.remote(model_id,i,new_classes,train_dataset,exemplars_per_class,args) for i in range(len(new_classes))])
                        #
                        #     # set mode of model back (dropout / Batch normalization layer is activated in train mode)
                        #     model.train()
                        #     print('rev_train: args.parallel: ', args.parallel)
                        #     args.parallel = 1
                        #     print('rev_train: args.parallel: ', args.parallel)


                        # for i in new_classes:
                        #     print('rev_train: model.exemplar_sets: length', len(model.exemplar_sets[i]))
                        model.compute_means = True
                        args.train_time_icarl_construct += time.time() - start_icarl

                        # evaluate this way of classifying on test set
                        start_eval = time.time()
                        for eval_cb in eval_cbs_exemplars:
                            if eval_cb is not None:
                                eval_cb(model, batch_index, task=task, args=args)
                        args.eval_time += time.time() - start_eval
                        # model.precision_dict_exemplars = model.precision_dict_exemplars.copy()


            #---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, y, x_=x_, y_=y_, scores_=scores_,
                                                    active_classes=active_classes,num_classes_per_task_l=num_classes_per_task_l,
                                                    task=task, rnt=1./task,args=args)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, task=task)

            # +++++ early-stopping code +++++
            start_eval = time.time()
            if hasattr(model, "precision_dict") or hasattr(model, "precision_dict_exemplars"):
                if batch_index % args.prec_log == 0:
                    if add_exemplars or use_exemplars:
                        if n_tasks <= 2:
                            if args.D1orD2 == 2 and task == 2:
                                f1 = model.precision_dict_exemplars['all_tasks_weighted_f1'][-1]
                                acc = model.precision_dict_exemplars['all_tasks_acc'][-1]
                            else:
                                f1 = model.precision_dict_exemplars['per_task_weighted_f1'][task-1][-1]
                                acc = model.precision_dict_exemplars['per_task_acc'][task-1][-1]
                        else:
                            f1 = model.precision_dict_exemplars['all_tasks_weighted_f1'][-1]
                            acc = model.precision_dict_exemplars['all_tasks_acc'][-1]
                    else:
                        if n_tasks <= 2:
                            if args.D1orD2 == 2 and task == 2:
                                f1 = model.precision_dict['all_tasks_weighted_f1'][-1]
                                acc = model.precision_dict['all_tasks_acc'][-1]
                            else:
                                f1 = model.precision_dict['per_task_weighted_f1'][task-1][-1]
                                acc = model.precision_dict['per_task_acc'][task-1][-1]
                        else:
                            f1 = model.precision_dict['all_tasks_weighted_f1'][-1]
                            acc = model.precision_dict['all_tasks_acc'][-1]
                    # print('mem_cnt: ', model.mem_cnt)
                    if f1 > best_f1:
                        trials = 0
                        best_f1 = f1
                        best_acc = acc if acc > best_acc else best_acc
                        model.epoch = int(batch_index / args.prec_log)
                        if (add_exemplars and use_exemplars):
                            model.epoch = int(batch_index / args.prec_log)
                        # +++++ SI +++++
                        # store W values into model so that we can use this model.prev_W in the beginning of next task
                        # to do 'update_omega' operation
                        if args.si == True:
                            start_si = time.time()
                            # print('Store W values to model.prev_W for later update_omega')
                            model.prev_W = W
                            args.train_time_si_update += time.time() - start_si
                        if len(num_classes_per_task_l) <= 2:
                            if num_classes_per_task_l[-1] == 1: # EXP 1
                                if (add_exemplars and use_exemplars):
                                    torch.save(model,
                                           '../data/saved_model/'+args.exp_setup+'icarl_weighted_best_exp1_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp)+'.pth')
                                else:
                                    torch.save(model,
                                           '../data/saved_model/'+args.exp_setup+'weighted_best_exp1_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp)+'.pth')
                                # torch.save(model.state_dict(),
                                #            '../data/saved_model/'+args.exp_setup+'weighted_best_exp1_'+args.cls_type+'_'+args.cl_alg+'_'
                                #             + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                #             + '_hp'+str(args.hp)+'.pth')
                            else: # EXP 2
                                if (add_exemplars and use_exemplars):
                                    torch.save(model,
                                           '../data/saved_model/'+args.exp_setup+'icarl_weighted_best_exp2_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp)+'.pth')
                                else:
                                    torch.save(model,
                                           '../data/saved_model/'+args.exp_setup+'weighted_best_exp2_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp)+'.pth')
                                # torch.save(model.state_dict(),
                                #            '../data/saved_model/'+args.exp_setup+'weighted_best_exp2_'+args.cls_type+'_'+args.cl_alg+'_'
                                #             + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                #             + '_hp'+str(args.hp)+'.pth')
                        else: # EXP 3
                            if (add_exemplars and use_exemplars):
                                torch.save(model,
                                           '../data/saved_model/'+args.exp_setup+'icarl_weighted_best_exp3_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp)+'.pth')
                            else:
                                torch.save(model,
                                           '../data/saved_model/'+args.exp_setup+'weighted_best_exp3_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp)+'.pth')
                            # torch.save(model.state_dict(),
                            #                '../data/saved_model/'+args.exp_setup+'weighted_best_exp3_'+args.cls_type+'_'+args.cl_alg+'_'
                            #                 + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2)
                            #                 + '_hp'+str(args.hp)+'.pth')
                        print('Epoch {} best model saved with weighted f1: {}, accuracy: {}, loss {}'.format(
                            int(batch_index / args.prec_log), best_f1, best_acc, loss_dict['loss_total']))
                    else:
                        trials += 1
                        if trials >= args.patience:
                            print('Early stopping on epoch {}, trials: {}, patience: {}'.format(int(batch_index / args.prec_log), trials, args.patience))
                            break
            args.eval_time += time.time() - start_eval


        #########################################
        ##----------> UPON FINISHING EACH TASK...
        ##----------> NOW WE DO THIS STEP BEFORE EACH TASK USING PREVIOUSLY STORED MODEL & PREVIOUS DATASET
        #########################################

        # ----- Close progres-bar(s) -----
        progress.close()
        if generator is not None:
            progress_gen.close()
        
        # happens only at the last task when calculating time
        if (args.rMeasureType == 'time') and (args.D1orD2 == (len(train_datasets))):
            # +++++ we should do this in the beginning of each task +++++
            # +++++ since we change 'ewc_lambda' for each task +++++
            # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
            if isinstance(model, ContinualLearner) and (model.ewc_lambda > 0):
                task = args.D1orD2
                # change task to task below
                # -find allowed classes
                if scenario == 'task':
                    allowed_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
                else:
                    if scenario == 'class':
                        if experiment == 'sensor':
                            allowed_classes = list(range(sum(num_classes_per_task_l[:(task)])))
                        else:
                            allowed_classes = list(range(classes_per_task * task))
                    else:
                        allowed_classes = None

                if model.mask_dict is not None:
                    model.apply_XdGmask(task=task)
                # -estimate FI-matrix
                # train_datasets[task-1] # cuz task's offset value is 1
                time_ewc = time.time()
                model.estimate_fisher(train_datasets[task - 1], allowed_classes=allowed_classes)
                args.train_time_ewc_update += time.time() - time_ewc


            # +++++ SI: calculate and update the normalized path integral +++++
            if isinstance(model, ContinualLearner) and (model.si_c > 0):
                start_si = time.time()
                model.update_omega(model.prev_W, model.epsilon)
                args.train_time_si_update += time.time() - start_si


            # ++++ No 'class' scenario but still do this before training ++++
            # REPLAY: update source for replay
            # previous_model = copy.deepcopy(model).eval()
            if replay_mode == 'generative':
                previous_model = copy.deepcopy(model).eval()
                Generative = True
                previous_generator = copy.deepcopy(
                    generator).eval() if generator is not None else previous_model
            elif replay_mode == 'current':
                start_lwf = time.time()
                previous_model = copy.deepcopy(model).eval()
                Current = True
                args.train_time_lwf_update += time.time() - start_lwf
            elif replay_mode in ('exemplars', 'exact'):
                previous_model = copy.deepcopy(model).eval()
                Exact = True
                if replay_mode == "exact":
                    previous_datasets = train_datasets[:task]
                else:
                    if scenario == "task":
                        previous_datasets = []
                        for task_id in range(task):
                            previous_datasets.append(
                                ExemplarDataset(
                                    model.exemplar_sets[
                                    (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                    target_transform=lambda y, x=classes_per_task * task_id: y + x)
                            )
                    else:
                        target_transform = (
                            lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                        previous_datasets = [
                            ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]

