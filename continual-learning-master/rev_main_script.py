#!/usr/bin/env python3
import sys,os,json,pickle
import collections,math,random
import time,datetime,pytz
import numpy as np
import pandas as pd
from collections import Counter
# import ray

from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
############################################
########## Plot Style Declaration ##########
# Alternatives include bmh, fivethirtyeight,ggplot,dark_background,seaborn-deep,etc
plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'times new roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



import argparse
import copy
import torch
from torch import optim
import visual_plt
import utils
from param_stamp import get_param_stamp, get_param_stamp_from_args
import evaluate
from data import get_singletask_experiment,get_multitask_experiment,get_multitask_experiment_multi_tasks,DATASET_CONFIGS
from encoder import Classifier
from vae_models import AutoEncoder
import callbacks as cb
from rev_train import train_cl
from continual_learner import ContinualLearner
from exemplars import ExemplarHandler
from replayer import Replayer
from Gem import Gem
from rev_quantizer import Quantizer


def run_D1orD2(args, train_datasets, test_datasets, test_total_dataset, model=None):
    config = args.config
    classes_per_task = args.classes_per_task

    # Set default arguments
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -if [log_per_task], reset all logs
    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
    # -if [iCaRL] is selected, select all accompanying options
    if hasattr(args, "icarl") and args.icarl:
        args.use_exemplars = True
        args.add_exemplars = True
        args.bce = True
        args.bce_distill = True
    # -if XdG is selected but not the Task-IL scenario, give error
    if (not args.scenario == "task") and args.gating_prop > 0:
        raise ValueError("'XdG' is only compatible with the Task-IL scenario.")
    # -if EWC, SI or XdG is selected together with 'feedback', give error
    if args.feedback and (args.ewc or args.si or args.gating_prop > 0 or args.icarl):
        raise NotImplementedError("EWC, SI, XdG and iCaRL are not supported with feedback connections.")
    # -if binary classification loss is selected together with 'feedback', give error
    if args.feedback and args.bce:
        raise NotImplementedError("Binary classification loss not supported with feedback connections.")
    # -if XdG is selected together with both replay and EWC, give error (either one of them alone with XdG is fine)
    if args.gating_prop > 0 and (not args.replay == "none") and (args.ewc or args.si):
        raise NotImplementedError("XdG is not supported with both '{}' replay and EWC / SI.".format(args.replay))
        # --> problem is that applying different task-masks interferes with gradient calculation
        #    (should be possible to overcome by calculating backward step on EWC/SI-loss also for each mask separately)
    # -if 'BCEdistill' is selected for other than scenario=="class", give error
    if args.bce_distill and not args.scenario == "class":
        raise ValueError("BCE-distill can only be used for class-incremental learning.")
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    scenario = args.scenario
    # If Task-IL scenario is chosen with single-headed output layer, set args.scenario to "domain"
    # (but note that when XdG is used, task-identity information is being used so the actual scenario is still Task-IL)
    if args.singlehead and args.scenario == "task":
        scenario = "domain"

    # If only want param-stamp, get it printed to screen and exit
    if hasattr(args, "get_stamp") and args.get_stamp:
        _ = get_param_stamp_from_args(args=args)
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda:"+str(args.cuda_num) if cuda else "cpu")
    args.device = device
    print('args.device: ', args.device)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------------------------------------------------------#

    # ----------------#
    # ----- DATA -----#
    # ----------------#

    # Prepare data for chosen experiment
    # (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
    #     name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir,
    #     verbose=True, exception=True if args.seed == 0 else False,
    # )

    # -------------------------------------------------------------------------------------------------#

    # ------------------------------#
    # ----- MODEL (CLASSIFIER) -----#
    # ------------------------------#

    # Define main model (i.e., classifier, if requested with feedback connections)
    if model==None:
        if args.experiment == 'sensor':
            # if args.gem == True:
            #     model = Gem(n_inputs=config['features'], n_outputs=config['classes'],
            #                 n_tasks=len(args.num_classes_per_task_l), args=args).to(device)
            # else:
            model = Classifier(
                num_features=config['features'], num_seq=config['seq'], classes=config['classes'],
                fc_layers=args.fc_lay+1, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
                fc_bn=True if args.fc_bn == "yes" else False, excit_buffer=True if args.gating_prop > 0 else False,
                binaryCE=args.bce, binaryCE_distill=args.bce_distill, experiment=args.experiment, cls_type=args.cls_type,
                args=args
            ).to(device)
        else:
            if args.feedback:
                model = AutoEncoder(
                    image_size=config['features'], image_channels=config['seq'], classes=config['classes'],
                    fc_layers=args.fc_lay, fc_units=args.fc_units, z_dim=args.z_dim,
                    fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
                    args=args
                ).to(device)
                model.lamda_pl = 1.  # --> to make that this VAE is also trained to classify
            else:
                model = Classifier(
                    num_features=config['features'], num_seq=config['seq'], classes=config['classes'],
                    fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
                    fc_bn=True if args.fc_bn == "yes" else False, excit_buffer=True if args.gating_prop > 0 else False,
                    binaryCE=args.bce, binaryCE_distill=args.bce_distill, experiment=args.experiment,
                    args=args
                ).to(device)

    # Define optimizer (only include parameters that "requires_grad")
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type == "sgd":
        model.optimizer = optim.SGD(model.optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))

    # -------------------------------------------------------------------------------------------------#

    # ----------------------------------#
    # ----- CL-STRATEGY: EXEMPLARS -----#
    # ----------------------------------#

    # Store in model whether, how many and in what way to store exemplars
    if isinstance(model, ExemplarHandler) and (args.use_exemplars or args.add_exemplars or args.replay == "exemplars"):
        model.memory_budget = args.budget
        model.norm_exemplars = args.norm_exemplars
        model.herding = args.herding

    # -------------------------------------------------------------------------------------------------#

    # -----------------------------------#
    # ----- CL-STRATEGY: ALLOCATION -----#
    # -----------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        if args.ewc:
            model.fisher_n = args.fisher_n
            model.gamma = args.gamma
            model.online = args.online
            model.emp_FI = args.emp_fi

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner):
        model.si_c = args.si_c if args.si else 0
        if args.si:
            model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and args.gating_prop > 0:
        mask_dict = {}
        excit_buffer_list = []
        for task_id in range(args.tasks):
            mask_dict[task_id + 1] = {}
            for i in range(model.fcE.layers):
                layer = getattr(model.fcE, "fcLayer{}".format(i + 1)).linear
                if task_id == 0:
                    excit_buffer_list.append(layer.excit_buffer)
                n_units = len(layer.excit_buffer)
                gated_units = np.random.choice(n_units, size=int(args.gating_prop * n_units), replace=False)
                mask_dict[task_id + 1][i] = gated_units
        model.mask_dict = mask_dict
        model.excit_buffer_list = excit_buffer_list

    # -------------------------------------------------------------------------------------------------#

    # -------------------------------#
    # ----- CL-STRATEGY: REPLAY -----#
    # -------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, Replayer):
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = True if (args.replay == "generative" and not args.feedback) else False
    if train_gen:
        # -specify architecture
        generator = AutoEncoder(
            image_size=config['features'], image_channels=config['seq'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.g_z_dim, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
        ).to(device)
        # -set optimizer(s)
        generator.optim_list = [
            {'params': filter(lambda p: p.requires_grad, generator.parameters()), 'lr': args.lr_gen}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
    else:
        generator = None

    # -------------------------------------------------------------------------------------------------#

    # ---------------------#
    # ----- REPORTING -----#
    # ---------------------#

    # Get parameter-stamp (and print on screen)
#     if args.cls_type != 'mlp':
#         model.name_lstm = "LSTM([{} X {} X {}])".format(config['features'], config['seq'], config['classes'])
#         param_stamp = get_param_stamp(
#             args, model.name_lstm, verbose=True, replay=True if (not args.replay == "none") else False,
#             replay_model_name=generator.name if (args.replay == "generative" and not args.feedback) else None,
#         )
#     else:
    param_stamp = get_param_stamp(
        args, model.name, verbose=True, replay=True if (not args.replay == "none") else False,
        replay_model_name=generator.name if (args.replay == "generative" and not args.feedback) else None,
    )

    # Print some model-characteristics on the screen
    # -main model
    print("\n")
    utils.print_model_info(model, title="MAIN MODEL")
    # -generator
    if generator is not None:
        utils.print_model_info(generator, title="GENERATOR")

    # Prepare for plotting in visdom
    # -define [precision_dict] to keep track of performance during training for storing and for later plotting in pdf
    precision_dict = evaluate.initiate_precision_dict(args.tasks)
    precision_dict_exemplars = evaluate.initiate_precision_dict(args.tasks) if args.use_exemplars else None
    # -visdom-settings
    if args.visdom:
        env_name = "{exp}{tasks}-{scenario}".format(exp=args.experiment, tasks=args.tasks, scenario=args.scenario)
        graph_name = "{fb}{replay}{syn}{ewc}{xdg}{icarl}{bud}".format(
            fb="1M-" if args.feedback else "", replay="{}{}".format(args.replay, "D" if args.distill else ""),
            syn="-si{}".format(args.si_c) if args.si else "",
            ewc="-ewc{}{}".format(args.ewc_lambda,
                                  "-O{}".format(args.gamma) if args.online else "") if args.ewc else "",
            xdg="" if args.gating_prop == 0 else "-XdG{}".format(args.gating_prop),
            icarl="-iCaRL" if (args.use_exemplars and args.add_exemplars and args.bce and args.bce_distill) else "",
            bud="-bud{}".format(args.budget) if (
                    args.use_exemplars or args.add_exemplars or args.replay == "exemplars"
            ) else "",
        )
        visdom = {'env': env_name, 'graph': graph_name}
        if args.use_exemplars:
            visdom_exemplars = {'env': env_name, 'graph': "{}-EX".format(graph_name)}
    else:
        visdom = visdom_exemplars = None

    # -------------------------------------------------------------------------------------------------#

    # ---------------------#
    # ----- CALLBACKS -----#
    # ---------------------#

    # Callbacks for reporting on and visualizing loss
    generator_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, model=model if args.feedback else generator, tasks=args.tasks,
                        iters_per_task=args.iters if args.feedback else args.g_iters,
                        replay=False if args.replay == "none" else True)
    ] if (train_gen or args.feedback) else [None]
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, tasks=args.tasks,
                           iters_per_task=args.iters, replay=False if args.replay == "none" else True)
    ] if (not args.feedback) else [None]

    # Callbacks for evaluating and plotting generated / reconstructed samples
    sample_cbs = [
        cb._sample_cb(log=args.sample_log, visdom=visdom, config=config, test_datasets=test_datasets,
                      sample_size=args.sample_n, iters_per_task=args.iters if args.feedback else args.g_iters)
    ] if (train_gen or args.feedback) else [None]

    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [prec_log]
#     eval_cb = cb._eval_cb(
#         log=args.prec_log, test_datasets=test_datasets, visdom=visdom, precision_dict=None, iters_per_task=args.iters,
#         test_size=args.prec_n, classes_per_task=classes_per_task, scenario=scenario,
#     )
    # -pdf / reporting: summary plots (i.e, only after each task)
    eval_cb_full = cb._eval_cb(
        log=args.prec_log, test_datasets=test_datasets, precision_dict=precision_dict,
        iters_per_task=args.iters, classes_per_task=classes_per_task, scenario=scenario,
        num_classes_per_task_l=args.num_classes_per_task_l, experiment=args.experiment
    )
    # -with exemplars (both for visdom & reporting / pdf)
    eval_cb_exemplars = cb._eval_cb(
        log=args.prec_log, test_datasets=test_datasets, visdom=visdom_exemplars, classes_per_task=classes_per_task,
        precision_dict=precision_dict_exemplars, scenario=scenario, iters_per_task=args.iters,
        with_exemplars=True,
        num_classes_per_task_l=args.num_classes_per_task_l, experiment=args.experiment
    ) if args.use_exemplars else None
    # -collect them in <lists>
#     eval_cbs = [eval_cb, eval_cb_full]
    eval_cbs = [eval_cb_full]
    eval_cbs_exemplars = [eval_cb_exemplars]

    # -------------------------------------------------------------------------------------------------#

    # --------------------#
    # ----- TRAINING -----#
    # --------------------#

    print("--> Training:")
    # Keep track of training-time
    start = time.time()
    # Train model
    train_cl(
            model, train_datasets, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
            iters=args.iters, batch_size=args.batch,
            generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
            sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
            eval_cbs_exemplars=eval_cbs_exemplars, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
            num_classes_per_task_l=args.num_classes_per_task_l, experiment=args.experiment, config=config, args=args
        )
    # if len(train_datasets) <= 3:
    #     train_cl(
    #         model, train_datasets, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
    #         iters=args.iters, batch_size=args.batch,
    #         generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
    #         sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
    #         eval_cbs_exemplars=eval_cbs_exemplars, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
    #         num_classes_per_task_l=args.num_classes_per_task_l, experiment=args.experiment, config=config, args=args
    #     )
    # else:
    #     train_cl_multi_tasks(
    #         model, train_datasets, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
    #         iters=args.iters, batch_size=args.batch,
    #         generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
    #         sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
    #         eval_cbs_exemplars=eval_cbs_exemplars, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
    #         num_classes_per_task_l=args.num_classes_per_task_l, experiment=args.experiment, config=config, args=args
    #     )
    # Get total training-time in seconds, and write to file
    total_time = time.time() - start
    args.total_time += total_time
    # time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
    t_file_name = 'weighted_best_exp1_'+args.cls_type+'_'+args.cl_alg+'_' \
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) + '_t' + str(args.D1orD2) \
                                            + '_hp'+str(args.hp)+'.pth'
    time_file = open("{}/time-{}.txt".format(args.r_dir, t_file_name), 'w')
    time_file.write('{}\n'.format(total_time))
    time_file.close()



    # -------------------------------------------------------------------------------------------------#

    # ----------------------#
    # ----- EVALUATION -----#
    # ----------------------#

#     print("\n\n--> Evaluation ({}-incremental learning scenario):".format(args.scenario))

#     # Evaluate precision of final model on full test-set
#     precs = [evaluate.validate(
#         model, test_datasets[i], verbose=False, test_size=None, task=i + 1, with_exemplars=False,
#         allowed_classes=list(range(classes_per_task * i, classes_per_task * (i + 1))) if scenario == "task" else None
#     ) for i in range(args.tasks)]
#     print("\n Precision on test-set (softmax classification):")
#     for i in range(args.tasks):
#         print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
#     average_precs = sum(precs) / args.tasks
#     test_total_prec = evaluate.validate(
#         model, test_total_dataset, verbose=False, test_size=None, task=2, with_exemplars=False,
#         allowed_classes=list(range(classes_per_task * i, classes_per_task * (i + 1))) if scenario == "task" else None
#     )
#     print('=> average precision over all {} tasks: {:.4f}'.format(args.tasks, average_precs))

#     # -with exemplars
#     if args.use_exemplars:
#         precs = [evaluate.validate(
#             model, test_datasets[i], verbose=False, test_size=None, task=i + 1, with_exemplars=True,
#             allowed_classes=list(
#                 range(classes_per_task * i, classes_per_task * (i + 1))) if scenario == "task" else None
#         ) for i in range(args.tasks)]
#         print("\n Precision on test-set (classification using exemplars):")
#         for i in range(args.tasks):
#             print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
#         average_precs_ex = sum(precs) / args.tasks
#         print('=> average precision over all {} tasks: {:.4f}'.format(args.tasks, average_precs_ex))
#     print("\n")

    # -------------------------------------------------------------------------------------------------#

    # ------------------#
    # ----- OUTPUT -----#
    # ------------------#

    # Average precision on full test set
#     output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
#     output_file.write('{}\n'.format(average_precs_ex if args.use_exemplars else average_precs))
#     output_file.close()
#     # -precision-dict
#     file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
#     utils.save_object(precision_dict_exemplars if args.use_exemplars else precision_dict, file_name)

#     # Average precision on full test set not evaluated using exemplars (i.e., using softmax on final layer)
#     if args.use_exemplars:
#         output_file = open("{}/prec_noex-{}.txt".format(args.r_dir, param_stamp), 'w')
#         output_file.write('{}\n'.format(average_precs))
#         output_file.close()
#         # -precision-dict:
#         file_name = "{}/dict_noex-{}".format(args.r_dir, param_stamp)
#         utils.save_object(precision_dict, file_name)

    # -------------------------------------------------------------------------------------------------#

    # --------------------#
    # ----- PLOTTING -----#
    # --------------------#

    # If requested, generate pdf
    if args.pdf:
        # -open pdf
        pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, param_stamp))

        # -show samples and reconstructions (either from main model or from separate generator)
        if args.feedback or args.replay == "generative":
            evaluate.show_samples(model if args.feedback else generator, config, size=args.sample_n, pdf=pp)
            for i in range(args.tasks):
                evaluate.show_reconstruction(model if args.feedback else generator, test_datasets[i], config, pdf=pp,
                                             task=i + 1)

        # -show metrics reflecting progression during training
        figure_list = []  # -> create list to store all figures to be plotted
        # -generate all figures (and store them in [figure_list])
        figure = visual_plt.plot_lines(
            precision_dict["all_tasks"], x_axes=precision_dict["x_task"],
            line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
        )
        figure_list.append(figure)
        figure = visual_plt.plot_lines(
            [precision_dict["average"]], x_axes=precision_dict["x_task"],
            line_names=['average all tasks so far']
        )
        figure_list.append(figure)
        if args.use_exemplars:
            figure = visual_plt.plot_lines(
                precision_dict_exemplars["all_tasks"], x_axes=precision_dict_exemplars["x_task"],
                line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
            )
            figure_list.append(figure)
        # -add figures to pdf (and close this pdf).
        for figure in figure_list:
            pp.savefig(figure)

        # -close pdf
        pp.close()
    print("precision_dict: ", precision_dict)
    # return model, precs[0] if args.D1orD2=='D1' else test_total_prec
    #return model, np.max(precision_dict['all_tasks'][0]) if args.D1orD2=='D1' else np.max(precision_dict['test_total_prec'])
    print('total_time: ', args.total_time / args.clsSeq)
    print('eval_time: ', args.eval_time / args.clsSeq)
    print('train_time_ewc_update: ', args.train_time_ewc_update / args.clsSeq)
    print('train_time_ewc_loss: ', args.train_time_ewc_loss / args.clsSeq)
    print('train_time_si_update: ', args.train_time_si_update / args.clsSeq)
    print('train_time_si_loss: ', args.train_time_si_loss / args.clsSeq)
    print('train_time_si_running_weights: ', args.train_time_si_running_weights / args.clsSeq)
    print('train_time_lwf_update: ', args.train_time_lwf_update / args.clsSeq)
    print('train_time_lwf_loss: ', args.train_time_lwf_loss / args.clsSeq)
    print('train_time_icarl_construct: ', args.train_time_icarl_construct / args.clsSeq)
    print('train_time_icarl_construct_class_mean: ', args.train_time_icarl_construct_class_mean / args.clsSeq)
    print('train_time_icarl_construct_argmin_loop: ', args.train_time_icarl_construct_argmin_loop / args.clsSeq)
    print('train_time_icarl_reduce: ', args.train_time_icarl_reduce / args.clsSeq)
    print('train_time_icarl_loss: ', args.train_time_icarl_loss / args.clsSeq)
    print('train_time_gem_update_memory: ', args.train_time_gem_update_memory / args.clsSeq)
    print('train_time_gem_compute_gradient: ', args.train_time_gem_compute_gradient / args.clsSeq)
    print('train_time_gem_violation_check: ', args.train_time_gem_violation_check / args.clsSeq)
    args.train_time_other_il = args.train_time_ewc_update + args.train_time_ewc_loss + \
            args.train_time_si_update + args.train_time_si_loss + args.train_time_si_running_weights + \
            args.train_time_lwf_update + args.train_time_lwf_loss + \
            args.train_time_icarl_construct + args.train_time_icarl_reduce + args.train_time_icarl_loss + \
            args.train_time_gem_update_memory + args.train_time_gem_compute_gradient + args.train_time_gem_violation_check

    print('only train_time: ', (args.total_time - args.eval_time - args.train_time_other_il)/args.clsSeq)
    

    return model, precision_dict, precision_dict_exemplars


class MyArgs():
    def __init__(self, cuda=True, cuda_num=0, parallel=1, quantize=32):
        self.get_stamp=False # help='print param_stamp & exit')
        self.seed=0 # help='random seed (for each random_module used)')
        self.cuda=cuda # help="don't use GPUs")
        self.cuda_num=cuda_num
        self.d_dir='./datasets' # help="default: %(default)s")
        self.p_dir='./plots' # help="default: %(default)s")
        self.r_dir='./results' # help="default: %(default)s")

        # expirimental task parameters
        self.experiment='sensor' # choices=['permMNIST', 'splitMNIST', 'sensor'])
        self.cls_type='mlp'
        self.scenario='class' # choices=['task', 'domain', 'class'])
        self.tasks=2 # help='number of tasks')
        self.dataset='opportunity' # ['opportunity', 'pamap2', 'hhar', etc.]
        self.patience = 200 # early-stopping parameter. can stand patience times of non-increase of performance.

        # specify loss functions to be used
        self.bce=False # help="use binary (instead of multi_class) classication loss")
        self.bce_distill=False # help='distilled loss on previous classes for new'' examples (only if __bce & __scenario="class")')

        # model architecture parameters
        self.fc_lay=3# help="# of fully_connected layers")
        self.fc_units=400 # help="# of units in first fc_layers")
        self.input_drop=0.2 # help="dropout probability for inputs")
        self.fc_drop=0.5 # help="dropout probability for fc_units")
        self.fc_bn='no'# help="use batch_norm in the fc_layers (no|yes)")
        self.fc_nl='relu' # choices=["relu", "leakyrelu","sigmoid","tanh"])
        self.singlehead=False# help="for Task_IL: use a 'single_headed' output layer   "" (instead of a 'multi_headed' one)")

        # training hyperparameters / initialization
        self.iters=500 # help="# batches to optimize solver")
        self.lr=0.001 # help="learning rate")
        self.lr2=0.001 # help="learning rate after D1")
        self.batch=32 # help="batch_size")
        self.optimizer='adam' # choices=['adam', 'adam_reset', 'sgd'], default='adam')

        # "memory replay" parameters
        self.feedback=False # help="equip model with feedback connections")
        self.z_dim=100 # help='size of latent representation (default: 100)')
        self.replay='none' # replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
        self.distill=False# help="use distillation for replay?")
        self.temp=2 #', type=float, default=2., dest='temp', # help="temperature for distillation")
        # _generative model parameters (if separate model)
        self.g_z_dim=100 #', type=int, default=100, # help='size of latent representation (default: 100)')
        self.g_fc_lay=None #', type=int, # help='[fc_layers] in generator (default: same as classifier)')
        self.g_fc_uni=None #', type=int, # help='[fc_units] in generator (default: same as classifier)')
        # _ hyper_parameters for generative model (if separate model)
        self.g_iters=None #', type=int, # help="# batches to train generator (default: as classifier)")
        self.lr_gen=None #', type=float, # help="learning rate generator (default: lr)")

        # "memory allocation" parameters
        self.ewc=False #', action='store_true', # help="use 'EWC' (Kirkpatrick et al, 2017)")
        self.ewc_lambda=5000 #', type=float, default=5000.,dest="ewc_lambda", # help="__> EWC: regularisation strength")
        self.fisher_n=None #', type=int, # help="__> EWC: sample size estimating Fisher Information")
        self.online=False #', action='store_true', # help="__> EWC: perform 'online EWC'")
        self.gamma=1.0 #', type=float, default=1., # help="__> EWC: forgetting coefficient (for 'online EWC')")
        self.emp_fi=False #', action='store_true', # help="__> EWC: estimate FI with provided labels")
        self.si=False #', action='store_true', # help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
        self.si_c=0.1 #', type=float, default=0.1, dest="si_c", # help="__> SI: regularisation strength")
        self.epsilon=0.1 #', type=float, default=0.1, dest="epsilon", # help="__> SI: dampening parameter")
        self.gating_prop=0.0 #', --xdg type=float, default=0., dest="gating_prop",# help="XdG: prop neurons per layer to gate")

        # exemplar parameters
        self.icarl=False #', action='store_true', # help="bce_distill, use_exemplars & add_exemplars")
        self.use_exemplars=False #', action='store_true', # help="use exemplars for classification")
        self.add_exemplars=False #', action='store_true', # help="add exemplars to current task dataset")
        self.budget=2000 #', type=int, default=2000, dest="budget", # help="how many exemplars can be stored?")
        self.herding=False #', action='store_true', # help="use herding to select exemplars (instead of random)")
        self.norm_exemplars=False #', action='store_true', # help="normalize features/averages of exemplars")
        self.gem=False
        self.n_memories=None
        self.memory_strength=None
        
        # evaluation parameters
        self.pdf=False #', action='store_true', # help="generate pdf with results")
        self.visdom=False #', action='store_true', # help="use visdom for on_the_fly plots")
        self.log_per_task=False #', action='store_true', # help="set all visdom_logs to [iters]")
        self.loss_log=200 #', type=int, default=200, metavar="N", # help="# iters after which to plot loss")
        self.prec_log=200 #', type=int, default=200, metavar="N", # help="# iters after which to plot precision")
        self.prec_n=1024 #', type=int, default=1024, # help="# samples for evaluating solver's precision")
        self.sample_log=500 #', type=int, default=500, metavar="N", # help="# iters after which to plot samples")
        self.sample_n= 64 #', type=int, default=64, # help="# images to show")

        # time & memory measure parameters
        self.total_time=0.0
        self.eval_time=0.0
        self.train_time_origin=0.0
        self.train_time_ewc_loss=0.0
        self.train_time_ewc_update=0.0
        self.train_time_ewc=0.0
        self.train_time_si_loss=0.0
        self.train_time_si_update=0.0
        self.train_time_si_running_weights=0.0
        self.train_time_si=0.0
        self.train_time_lwf_update=0.0
        self.train_time_lwf_loss=0.0
        self.train_time_lwf=0.0
        self.train_time_icarl_construct=0.0
        self.train_time_icarl_construct_class_mean=0.0
        self.train_time_icarl_construct_argmin_loop=0.0
        self.train_time_icarl_reduce=0.0
        self.train_time_icarl_loss=0.0
        self.train_time_icarl=0.0
        self.train_time_gem_update_memory=0.0
        self.train_time_gem_compute_gradient=0.0
        self.train_time_gem_violation_check=0.0
        self.train_time_gem=0.0
        self.train_time_other_il = 0.0

        self.total_time_l = []
        self.eval_time_l = []
        self.train_time_origin_l = []
        self.train_time_ewc_loss_l = []
        self.train_time_ewc_update_l = []
        self.train_time_ewc_l = []
        self.train_time_si_loss_l = []
        self.train_time_si_update_l = []
        self.train_time_si_running_weights_l = []
        self.train_time_si_l = []
        self.train_time_lwf_update_l = []
        self.train_time_lwf_loss_l = []
        self.train_time_lwf_l = []
        self.train_time_icarl_construct_l = []
        self.train_time_icarl_construct_class_mean_l = []
        self.train_time_icarl_construct_argmin_loop_l = []
        self.train_time_icarl_reduce_l = []
        self.train_time_icarl_loss_l = []
        self.train_time_icarl_l = []
        self.train_time_gem_update_memory_l = []
        self.train_time_gem_compute_gradient_l = []
        self.train_time_gem_violation_check_l = []
        self.train_time_gem_l = []
        self.train_time_other_il_l = []

        self.only_train_time_l = []

        self.memory_storage=0.0
        self.n_params_origin = 0
        self.n_params_il = 0

        self.exemplars_sets_indexes = []
        self.augment = -1
        self.scaling = 1
        self.parallel = 1
        self.quantize = quantize
        # self.ray = None if parallel <=1 else ray.init(num_cpus=parallel)


    def set_params_dataset(self, experiment, cls_type, scenario, tasks, dataset=None, d_dir=None, seed=None):
        self.experiment = experiment
        self.cls_type = cls_type
        self.scenario = scenario
        self.tasks = tasks
        self.d_dir = d_dir
        self.seed = seed
        self.dataset = dataset

    def set_params_D1(self, fc_lay, fc_units, fc_nl, lr, batch):
        self.fc_lay = fc_lay
        self.fc_units = fc_units
        self.fc_nl = fc_nl
        self.lr = lr
        self.batch = batch

    def set_params_D2_ewc(self, lr, ewc_lambda, batch):
        self.ewc = True
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.batch = batch
        
    def set_params_D2_ewc_online(self, lr, batch, ewc_lambda, gamma):
        self.ewc = True
        self.online = True
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.batch = batch
    
    def set_params_D1_si(self):
        self.si = True
    
    def set_params_D2_si(self, lr, batch, si_c, si_epsilon):
        self.lr = lr
        self.batch = batch
        self.si = True
        self.si_c = si_c
        self.epsilon = si_epsilon
        
    def set_params_D2_lwf(self, lr, batch):
        self.lr = lr
        self.batch = batch
        self.replay = 'current'
        self.distill = True
        
    def set_params_D12_icarl(self, lr, batch, budget, herding=True,parallel=1,augment=-1,scaling=1):
        self.lr = lr
        self.batch = batch
        self.icarl = True
        self.norm_exemplars = True
        self.herding = herding
        self.budget = budget
        self.augment = augment
        self.scaling = scaling
        self.parallel = parallel
    
    def set_params_D12_gem(self, lr, batch, n_memories, memory_strength):
        self.lr = lr
        self.batch = batch
        self.gem = True
        self.n_memories = n_memories
        self.memory_strength = memory_strength
    
    def set_params_train(self, iters):
        self.iters = iters

    def set_params_eval(self, prec_log, patience=200):
        self.prec_log = prec_log
        self.patience = patience


##################################################################
##################################################################

metrics_list = ['macro_f1','micro_f1','weighted_prec','macro_prec','micro_prec','weighted_rec','macro_rec','micro_rec','acc','weighted_f1']

def get_ewc_hp_num(n_layers, n_units):
    hp_num = 0
    if n_layers == 1:
        if n_units == 32:
            hp_num = 1
        else:
            hp_num = 2
    else:
        if n_units == 32:
            hp_num = 3
        else:
            hp_num = 4

    return hp_num

def get_icarl_hp_num(n_layers, n_units, budget_percent):
    hp_num = 0
    if n_layers == 1:
        if n_units == 32:
            hp_num = 1
        else:
            hp_num = 2
    else:
        if n_units == 32:
            hp_num = 3
        else:
            hp_num = 4

    return hp_num + 4 * (int(budget_percent/5))

def get_ewc_hparams(exp_setup='', rMeasureType=''):
    ewc_lambda_l = [1, 10, 100, 1000, 10000, 100000, 1000000]
    budget_size_l = [1, 5, 10, 20]
    if rMeasureType=='time':
        ewc_lambda_l = [10000]
    if exp_setup == 'time':
        fc_lay_l =[1]
        fc_units_l =[32]
        lr_l = [0.001]
        lr_l_2 = [0.0001]
        batch_l = [32]
    elif exp_setup == 'layers1_units32':
        fc_lay_l =[1]
        fc_units_l =[32]
        lr_l = [0.001]
        lr_l_2 = [0.0001]
        batch_l = [32]
    elif exp_setup == 'layers1_units64':
        fc_lay_l =[1]
        fc_units_l =[64]
        lr_l = [0.001]
        lr_l_2 = [0.0001]
        batch_l = [32]
    elif exp_setup == 'layers2_units32':
        fc_lay_l =[2]
        fc_units_l =[32]
        lr_l = [0.001]
        lr_l_2 = [0.0001]
        batch_l = [32]
    elif exp_setup == 'layers2_units64':
        fc_lay_l =[2]
        fc_units_l =[64]
        lr_l = [0.001]
        lr_l_2 = [0.0001]
        batch_l = [32]
    else:
        fc_lay_l = [2]
        fc_units_l = [64]
        lr_l = [0.001]
        lr_l_2 = [0.0001]
        batch_l = [32]
        pass

    return fc_lay_l,fc_units_l,lr_l,lr_l_2,batch_l,ewc_lambda_l,budget_size_l

def getYLabelName(name):
    if name == 'acc':
        return 'Accuracy'
    elif name == 'weighted_f1':
        return 'Weighted F1'
    elif name == 'macro_f1':
        return 'Macro F1'
    elif name == 'micro_f1':
        return 'Micro F1'
    elif name == 'weighted_prec':
        return 'Weighted Precision'
    elif name == 'macro_prec':
        return 'Macro Precision'
    elif name == 'micro_prec':
        return 'Micro Precision'
    elif name == 'weighted_rec':
        return 'Weighted Recall'
    elif name == 'macro_rec':
        return 'Macro Recall'
    elif name == 'micro_rec':
        return 'Micro Recall'
    else:
        return ''

def getExpType(args):
    if len(args.num_classes_per_task_l) <= 2:
        exp_type = 'exp1' if args.num_classes_per_task_l[-1] == 1 else 'exp2'
    else:
        exp_type = 'exp3'
    return exp_type

def rev_plot_all_metrics_summed(precision_dict_list_list, epoch=20, args=None, args_Dn_l=None):
    print('\n\n===== rev_plot_all_metrics_summed =====\n\n')
    
    x = [i+1 for i in range(2*epoch)]
    for metrics in metrics_list:
        y1_l = []
        y2_d1_l = []
        y2_d2_l = []
        y2_d1_d2_l = []
        for i, precision_dict_list in enumerate(precision_dict_list_list):
            y1 = precision_dict_list[0]['per_task_'+metrics][0]
            y2_d1 = precision_dict_list[1]['per_task_'+metrics][0] # d1 acc
            y2_d2 = precision_dict_list[1]['per_task_'+metrics][1] # d2 acc
            y2_d1_d2 = precision_dict_list[1]['all_tasks_'+metrics] # d1_d2 acc
            y1_l.append(y1)
            y2_d1_l.append(y2_d1)
            y2_d2_l.append(y2_d2)
            y2_d1_d2_l.append(y2_d1_d2)
        y2_l = [y2_d1_l,y2_d2_l,y2_d1_d2_l]
        
        if args is not None:
            if len(args.num_classes_per_task_l) <= 2:
                exp_type = 'exp1' if args.num_classes_per_task_l[-1] == 1 else 'exp2'
            else:
                exp_type = 'exp3'
            plotSummedBest(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/summed_best_'+metrics+'_'
                                      +exp_type+'_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset, 
                rLogAxis='no', rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
            plotSummedBestSd(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/sd_summed_best_'+metrics+'_'
                                      +exp_type+'_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset, 
                rLogAxis='no', rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
        else:
            plotSummedBest(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/summed_best_test', rLogAxis='no',
                rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
            plotSummedBestSd(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/sd_summed_best_test', rLogAxis='no',
                rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
    print('args: ', args)
    print('precision_dict_list_list:', precision_dict_list_list)
    print('args_Dn_l:', args_Dn_l)



def plot_all_metrics_summed(precision_dict_list_list, epoch=20, args=None):
    print('\n\n===== plot_all_metrics_summed =====\n\n')
    
    x = [i+1 for i in range(2*epoch)]
    for metrics in metrics_list:
        y1_l = []
        y2_d1_l = []
        y2_d2_l = []
        y2_d1_d2_l = []
        for i, precision_dict_list in enumerate(precision_dict_list_list):
            y1 = precision_dict_list[0]['per_task_'+metrics][0]
            y2_d1 = precision_dict_list[1]['per_task_'+metrics][0] # d1 acc
            y2_d2 = precision_dict_list[1]['per_task_'+metrics][1] # d2 acc
            y2_d1_d2 = precision_dict_list[1]['all_tasks_'+metrics] # d1_d2 acc
            y1_l.append(y1)
            y2_d1_l.append(y2_d1)
            y2_d2_l.append(y2_d2)
            y2_d1_d2_l.append(y2_d1_d2)
        y2_l = [y2_d1_l,y2_d2_l,y2_d1_d2_l]
        
        if args is not None:
            if len(args.num_classes_per_task_l) <= 2:
                exp_type = 'exp1' if args.num_classes_per_task_l[-1] == 1 else 'exp2'
            else:
                exp_type = 'exp3'
            plotSummedBest(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/summed_best_'+metrics+'_'
                                      +exp_type+'_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset, 
                rLogAxis='no', rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
            plotSummedBestSd(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/sd_summed_best_'+metrics+'_'
                                      +exp_type+'_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset, 
                rLogAxis='no', rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
        else:
            plotSummedBest(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/summed_best_test', rLogAxis='no',
                rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
            plotSummedBestSd(rX=x,rY=[y1_l,y2_l],rStdErrN='depends',
                rFileName='../plots/sd_summed_best_test', rLogAxis='no',
                rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
    print('args: ', args)
    print('precision_dict_list_list:', precision_dict_list_list)

def plot_all_metrics(precision_dict_list, epoch=20, args=None):
    metrics_list = ['acc','weighted_f1','macro_f1','micro_f1','weighted_prec','macro_prec','micro_prec','weighted_rec','macro_rec','micro_rec']
    x = [i+1 for i in range(2*epoch)]
    print('precision_dict_list:', precision_dict_list)
    for metrics in metrics_list:
        y1 = precision_dict_list[0]['per_task_'+metrics][0]
        y2_d1 = precision_dict_list[1]['per_task_'+metrics][0] # d1 acc
        y2_d2 = precision_dict_list[1]['per_task_'+metrics][1] # d2 acc
        y2_d1_d2 = precision_dict_list[1]['all_tasks_'+metrics] # d1_d2 acc
        y2 = [y2_d1, y2_d2, y2_d1_d2]
        if metrics == 'acc':
            print('accuracy')
            print('y1 max: ', np.max(y1))
            print('y2 d1 max: ', np.max(y2_d1))
            print('y2 d2 max: ', np.max(y2_d2))
            print('y2 d1&d2 max: ', np.max(y2_d1_d2))
        if metrics == 'weighted_f1':
            print('weighted f1 scores')
            print('y1 max: ', np.max(y1))
            print('y2 d1 max: ', np.max(y2_d1))
            print('y2 d2 max: ', np.max(y2_d2))
            print('y2 d1&d2 max: ', np.max(y2_d1_d2))
        
        if args is not None:
            if len(args.num_classes_per_task_l) <= 2:
                exp_type = 'exp1' if args.num_classes_per_task_l[-1] == 1 else 'exp2'
            else:
                exp_type = 'exp3'
            plotMovingDistChurnProba4(rX=x,rY=[y1,y2],rStdErrN='depends',
                rFileName='../plots/best_'+metrics+'_'
                                      +exp_type+'_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) 
                                          + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp), 
                rLogAxis='no', rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch)
        else:
            plotMovingDistChurnProba4(rX=x,rY=[y1,y2],rStdErrN='depends',
                rFileName='../plots/test', rLogAxis='no',
                rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch)


def plot_D1_metrics(precision_dict_list, epoch=20, args=None):
    metrics_list = ['acc','weighted_f1','macro_f1','micro_f1','weighted_prec','macro_prec','micro_prec','weighted_rec','macro_rec','micro_rec']
    x = [i+1 for i in range(epoch)]
    print('precision_dict_list:', precision_dict_list)
    for metrics in metrics_list:
        y1 = precision_dict_list[0]['per_task_'+metrics][0]
        if metrics == 'acc':
            print('accuracy')
            print('y1 max: ', np.max(y1))
        if metrics == 'weighted_f1':
            print('weighted f1 scores')
            print('y1 max: ', np.max(y1))

        if args is not None:
            if len(args.num_classes_per_task_l) <= 2:
                exp_type = 'exp1' if args.num_classes_per_task_l[-1] == 1 else 'exp2'
            else:
                exp_type = 'exp3'
            plotD1Metrics(rX=x,rY=y1,rStdErrN='depends',
                rFileName='../plots/best_'+metrics+'_'
                                      +exp_type+'_'+args.cls_type+'_'+args.cl_alg+'_'
                                            + args.dataset+'_'+'clsSeq'+str(args.clsSeq) 
                                          + '_t' + str(args.D1orD2)
                                            + '_hp'+str(args.hp), 
                rLogAxis='no', rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)
        else:
            plotD1Metrics(rX=x,rY=y1,rStdErrN='depends',
                rFileName='../plots/test', rLogAxis='no',
                rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=epoch,metrics=metrics,args=args)


def plotSummedBestSd(rX,rY,rStdErrN,rFileName,rLogAxis,rXLabel,rYLabel,epoch,metrics,args):
    x = rX
    df_l = []
    desc_l = []
    y_l = []
    e_l = []
    y1 = rY[0]
    df = pd.DataFrame(y1)
    desc = df.describe()
    df_l.append(df)
    desc_l.append(desc)
    print('\n===== y1 =====')
    print('desc: ', desc)

    y = desc.iloc[1, :]
    e = getStdOrStdErr(desc, df, 'sd')
    y_l.append(y)
    e_l.append(e)
    
    y2 = rY[1]
    print('\n===== y2 =====')
    for i, l_l in enumerate(y2):
        df = pd.DataFrame(l_l)
        desc = df.describe()
        df_l.append(df)
        desc_l.append(desc)
        print('\nData: D', i+1, ' // desc: ', desc)

        y = desc.iloc[1, :]
        e = getStdOrStdErr(desc, df, 'sd')
        y_l.append(y)
        e_l.append(e)

    plt.errorbar(x[epoch:], y_l[3], yerr=e_l[3], linestyle=':', color='C3', label=None)
    plt.plot(x[epoch:], y_l[3], linestyle='-.', color='C3', label='$D1 \cup D2$', marker='$*$', markersize=15, zorder=3)
    plt.errorbar(x[epoch:], y_l[2], yerr=e_l[2], linestyle=':', color='C2', label=None)
    plt.plot(x[epoch:], y_l[2], linestyle=':', color='C2', label='$D2$', marker='d', markersize=9, zorder=3)
    y0 = pd.concat([y_l[0],y_l[1]], ignore_index=True)
    e0 = pd.concat([e_l[0],e_l[1]], ignore_index=True)
    plt.errorbar(x, y0, yerr=e0, linestyle=':', color='C0', label=None)
    plt.plot(x, y0, linestyle='-', color='C0',label='$D1$', marker='o', markersize=9, zorder=3)
    
    if rLogAxis == 'x':
        plt.xscale('log')
    elif rLogAxis == 'y':
        plt.yscale('log')
    elif rLogAxis == 'xy':
        plt.xscale('log')
        plt.yscale('log')

    plt.grid(True, which="both", ls=":")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.ylabel(rYLabel)
    plt.xlabel(rXLabel)
    legend_properties = {'weight':'bold'}
    
    leg = plt.legend(loc='best', numpoints=1, ncol=1, fancybox=True, borderpad=1, framealpha=0.0, prop=legend_properties)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin, xmax, -0.05, 1.05])
    step_ = 5 if epoch < 40 else 10
    plt.xticks([i for i in range(1, 2*epoch, step_)] + [2*epoch])
    if (len(rFileName) > 10) and (rFileName != 'not now'):
        plt.savefig(rFileName + '.pdf', bbox_inches='tight')
    plt.show()


def plotSummedBest(rX,rY,rStdErrN,rFileName,rLogAxis,rXLabel,rYLabel,epoch,metrics,args):
    metrics_set = {'acc','weighted_f1'}
    x = rX
    df_l = []
    desc_l = []
    y_l = []
    e_l = []
    y1 = rY[0]
    df = pd.DataFrame(y1)
    desc = df.describe()
    df_l.append(df)
    desc_l.append(desc)
    print('\n===== y1 =====')
    print('desc: ', desc)

    y = desc.iloc[1, :]
    e = getStdOrStdErr(desc, df, 'sd')
    y_l.append(y)
    e_l.append(e)
    
    y2 = rY[1]
    print('\n===== y2 =====')
    for i, l_l in enumerate(y2):
        df = pd.DataFrame(l_l)
        desc = df.describe()
        df_l.append(df)
        desc_l.append(desc)
        print('\nData: D', i+1, ' // desc: ', desc)

        y = desc.iloc[1, :]
        e = getStdOrStdErr(desc, df, 'sd')
        y_l.append(y)
        e_l.append(e)

    if metrics in metrics_set:
        print(('' if args==None else args.cl_alg+': ')+rYLabel+': D1 Max: {} (epoch: {}), D2 Max: {} (epoch: {})'.format(
            np.max(y_l[0]), np.argmax(y_l[0])+1, np.max(y_l[3]), np.argmax(y_l[3])+1
        ))
    
    plt.plot(x[epoch:], y_l[3], linestyle='-.', color='C3', label='$D1 \cup D2$', marker='$*$', markersize=15, zorder=3)
    plt.plot(x[epoch:], y_l[2], linestyle=':', color='C2', label='$D2$', marker='d', markersize=9, zorder=3)
    y0 = pd.concat([y_l[0],y_l[1]], ignore_index=True)
    plt.plot(x, y0, linestyle='-', color='C0',label='$D1$', marker='o', markersize=9, zorder=3)
    
    if rLogAxis == 'x':
        plt.xscale('log')
    elif rLogAxis == 'y':
        plt.yscale('log')
    elif rLogAxis == 'xy':
        plt.xscale('log')
        plt.yscale('log')

    plt.grid(True, which="both", ls=":")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.ylabel(rYLabel)
    plt.xlabel(rXLabel)
    legend_properties = {'weight':'bold'}
    
    leg = plt.legend(loc='best', numpoints=1, ncol=1, fancybox=True, borderpad=1, framealpha=0.0, prop=legend_properties)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin, xmax, -0.05, 1.05])
    step_ = 5 if epoch < 40 else 10
    plt.xticks([i for i in range(1, 2*epoch, step_)] + [2*epoch])
    if (len(rFileName) > 10) and (rFileName != 'not now'):
        plt.savefig(rFileName + '.pdf', bbox_inches='tight')
    plt.show()


def plotMovingDistChurnProba4(rX,rY,rStdErrN,rFileName,rLogAxis,rXLabel,rYLabel,epoch):
    x = rX
    y1 = rY[0]
    y2 = rY[1]
    y2_d1 = [] # d1 acc
    y2_d2 = [] # d2 acc
    y2_d1_d2 = [] # d1_d2 acc
    for i in range(0, epoch):
        y2_d1.append(y2[0][i])
        y2_d2.append(y2[1][i])
        y2_d1_d2.append(y2[2][i])
#     plt.plot(x, Y[4], linestyle='-.', color='C4', marker='s', markersize=10,markerfacecolor='none', zorder=3, label='$k = 50$')
#     plt.plot(x, Y[3], linestyle=':', color='C3', marker='v', markersize=10,markerfacecolor='none', zorder=3, label='$k = 40$')
#     plt.plot(x, Y[2], linestyle=':', color='C2', marker='$*$', markersize=10,markerfacecolor='none', zorder=3, label='$k = 30$')
#     plt.plot(x, Y[1], linestyle='-.', color='C1', marker='^', markersize=10,markerfacecolor='none', zorder=3, label='$k = 20$')
#     plt.plot(x, Y[0], linestyle=':', color='C0', marker='o', markersize=10,markerfacecolor='none', zorder=3, label='$k = 10$')    
    #plt.plot(x, Y[4], linestyle='-.', color='C4', marker='s', markersize=10, zorder=3, label='$k = 50$')
    #plt.plot(x[10:], y2_d1_d2, linestyle=':', color='C3', marker='v', markersize=10, zorder=3, label='$k = 40$')
    plt.plot(x[epoch:], y2_d1_d2, linestyle='-.', color='C3', marker='$*$', markersize=13, zorder=3, label='$D1 \cup D2$')
    plt.plot(x[epoch:], y2_d2, linestyle=':', color='C2', marker='d', markersize=8, zorder=3, label='$D2$')
    plt.plot(x, y1+y2_d1, linestyle='-', color='C0', marker='o', markersize=8, zorder=3, label='$D1$')

#     plt.axis([-5, 205, 0.1, 0.6])
    if rLogAxis == 'x':
        plt.xscale('log')
#         plt.axis([4, 210, 0.1, 0.6])
    elif rLogAxis == 'y':
        plt.yscale('log')
    elif rLogAxis == 'xy':
        plt.xscale('log')
        plt.yscale('log')
    else:
        pass
    
    plt.grid(True, which="both", ls=":")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.ylabel(rYLabel)
    plt.xlabel(rXLabel)
#     leg = plt.legend()
    legend_properties = {'weight':'bold'}
    
    leg = plt.legend(loc='best', numpoints=1, ncol=1, fancybox=True, borderpad=1, framealpha=0.0, prop=legend_properties)
#     leg.get_frame().set_linewidth(1)
#     leg.get_frame().set_edgecolor('b')
#     leg.get_frame().set_facecolor('white')

    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin, xmax, -0.05, 1.05])
    step_ = 5 if epoch < 40 else 10
    plt.xticks([i for i in range(1, 2*epoch, step_)] + [2*epoch])
#     plt.xticks(x, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], rotation=90)
#     plt.xticks(x, ['0%', '20%','40%', '60%', '80%', '100%'])
#     plt.xticks(['1', '2','3', '4'])
#     plt.axis([5, 220, 0.15, 0.55])
#     plt.axis([0, , 0.6, 0.8])
    if (len(rFileName) > 10) and (rFileName != 'not now'):
        plt.savefig(rFileName + '.pdf', bbox_inches='tight')
    plt.show()


def plotD1Metrics(rX,rY,rStdErrN,rFileName,rLogAxis,rXLabel,rYLabel,epoch,metrics,args):
    metrics_set = {'acc','weighted_f1'}
    x = rX
    y1 = rY
    if metrics in metrics_set:
        print( ('' if args==None else args.cl_alg+': ') + rYLabel+': D1 Max: {} (epoch: {})'.format(
            np.max(y1), np.argmax(y1)+1
        ))
    plt.plot(x, y1, linestyle='--', color='C0', marker='o', markersize=8, zorder=3, label='$D1$')

    if rLogAxis == 'x':
        plt.xscale('log')
    elif rLogAxis == 'y':
        plt.yscale('log')
    elif rLogAxis == 'xy':
        plt.xscale('log')
        plt.yscale('log')
    else:
        pass
    
    plt.grid(True, which="both", ls=":")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.ylabel(rYLabel)
    plt.xlabel(rXLabel)
    legend_properties = {'weight':'bold'}
    
    leg = plt.legend(loc='best', numpoints=1, ncol=1, fancybox=True, borderpad=1, framealpha=0.0, prop=legend_properties)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin, xmax, -0.05, 1.05])
    step_ = 5 if epoch < 40 else 10
    plt.xticks([i for i in range(1, epoch, step_)] + [epoch])
    if (len(rFileName) > 10) and (rFileName != 'not now'):
        plt.savefig(rFileName + '.pdf', bbox_inches='tight')
    plt.show()

    
def getStdOrStdErr(desc, df, rStdErrN):
    if isinstance(rStdErrN, str):
        if rStdErrN == 'sd':
            e = desc.iloc[2, :]
        else:
            e = desc.iloc[2, :] / math.sqrt(df.shape[0])
    else:
        e = desc.iloc[2, :] / math.sqrt(rStdErrN)
    return e

def plotWithErrorBarMultiTasks(rXLabel,rYLabel,rX,rY,rStdErrN,rFileName,rLogAxis):
    x = rX
    df_l = []
    desc_l = []
    y_l = []
    e_l = []
    for i, l_l in enumerate(rY):
        df = pd.DataFrame(l_l)
        desc = df.describe()
        df_l.append(df)
        desc_l.append(desc)
        print('idx: ', i, ' desc: ', desc)

        y = desc.iloc[1, :]
        e = getStdOrStdErr(desc, df, rStdErrN)
        y_l.append(y)
        e_l.append(e)
    
    plt.errorbar(x, y_l[0], yerr=e_l[0], linestyle=':', color='C0', label=None)
    plt.plot(x, y_l[0], linestyle=':', color='C0', label='Alg. 0', marker='o', markersize=10, zorder=3)
    plt.errorbar(x, y_l[1], yerr=e_l[1], linestyle=':', color='C1', label=None)
    plt.plot(x, y_l[1], linestyle=':', color='C1', label='Alg. 1', marker='o', markersize=10, zorder=3)
#     plt.errorbar(x, y[0], yerr=e[0], linestyle=':', color='C0', label=None)
#     plt.plot(x, y[0], linestyle=':', color='C0', , label='Alg. 1', marker='o', markersize=10, zorder=3)
#     plt.errorbar(x, y[0], yerr=e[0], linestyle=':', color='C0', label=None)
#     plt.plot(x, y[0], linestyle=':', color='C0', , label='Alg. 1', marker='o', markersize=10, zorder=3)

    if rLogAxis == 'x':
        plt.xscale('log')
    elif rLogAxis == 'y':
        plt.yscale('log')
    elif rLogAxis == 'xy':
        plt.xscale('log')
        plt.yscale('log')
    
    plt.grid(True, which="both", ls=":")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.ylabel(rYLabel)
    plt.xlabel(rXLabel)
    plt.legend(loc='best', numpoints=1, ncol=1, fancybox=True)
    xmin, xmax, ymin, ymax = plt.axis()
    # plt.axis([8, 102, 0.0, 1.01])
    plt.xticks(list(range(1, len(x)+1)))
    #plt.xticks(x, ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], rotation=90)
    if (len(rFileName) > 10) and (rFileName != 'not now'):
        plt.savefig(rFileName + '.pdf', bbox_inches='tight')
    plt.show()


def makeClassSetD1(class_set_D2, num_classes):
    class_list = []
    for i in range(num_classes):
        if i not in class_set_D2:
            class_list.append(i)
    return set(class_list)


def makeClassSeqList(class_D2_list, rDataset, max_num_cases=1):
    np.random.seed(0)
    config = DATASET_CONFIGS[rDataset]
    num_classes = config['classes']
    input_class_seq_l = []
    for clsSeq, class_D2_l in enumerate(class_D2_list):
        class_D1_seed_set = makeClassSetD1(set(class_D2_l), num_classes)
        class_D1_seed_list = list(class_D1_seed_set)
        class_Dn_seed_set = set(class_D2_l)
        class_Dn_seed_list = class_D2_l
        # print('class_D1_seed_list: ', class_D1_seed_list)
        # print('class_Dn_seed_set: ', class_Dn_seed_set)

        fact = 1
        for i in range(1,len(class_Dn_seed_set)+1):
            fact = fact * i

        num_cases = min(fact, max_num_cases)
        for i in range(num_cases):
            input_class_seq_l.append([])
            input_class_seq_l[i+clsSeq*num_cases].append(class_D1_seed_list)
        
        if num_cases == 1:
            for i in range(num_cases):
                for j, idx in enumerate(class_Dn_seed_list):
                    input_class_seq_l[i+clsSeq*num_cases].append([idx])
        else:
            i = 0
            indeces_selected_dict = {}
            while(i<num_cases):
                indeces_selected = np.random.choice(class_Dn_seed_list, size=len(class_Dn_seed_list), replace=False)
                s = ""
                for val in indeces_selected:
                    s += str(val)
                if s not in indeces_selected_dict:
                    for idx in indeces_selected:
                        input_class_seq_l[i+clsSeq*num_cases].append([idx])
                    indeces_selected_dict[s] = 1
                    i+=1

    print('input_class_seq_l: ')
    for l in input_class_seq_l:
        print(l)
    return input_class_seq_l


def computeAvgAccForgetAkk(model, datasets, precision_dict_list, args):
    # precision_dict_list = [ # tasks x {'all_tasks_weighted_f1', , etc.}]
    # EX: precision_dict_list[task_idx]['all_tasks_weighted_f1'] 
    # compute Ak,j in j = 1, k
    temp_dict = {}
    n_tasks = len(precision_dict_list)
    metrics_dict_list = []
    for i in range(n_tasks):
        current_task = i+1
        allowed_classes = list(range(np.sum(args.num_classes_per_task_l[:current_task])))
        metrics_dict_list.append(evaluate.validate(
            model, datasets[-1], test_size=None, verbose=False,
           allowed_classes=allowed_classes, with_exemplars=True if args.cl_alg[:5]=='icarl' else False,
           no_task_mask=False, task=i+1
            )) 
    print(metrics_dict_list)
    A_kj_dict = dict2list(metrics_dict_list)
    # test_total_metrics_dict = validate(model, datasets[-1], test_size=test_size, verbose=verbose,
    #                                allowed_classes=allowed_classes, with_exemplars=with_exemplars,
    #                                no_task_mask=no_task_mask, task=n_tasks)
    

    ##### get inverse Intransigence A_k,k
    A_kk = np.max(precision_dict_list[-1]['all_tasks_weighted_f1'])

    ##### get A_k,j in j = 1, k
    # print('A_kj_dict')
    # # A_kj_dict = dict2list(metrics_dict_list)
    # A_kj = A_kj_dict['weighted_f1']
    # # A_kj.append(A_kk)
    # print('A_kj: ', A_kj)
    # print('A_kk: ', A_kk)

    # ##### get average metrics A_k
    # A_K = np.mean(A_kj)


    # ##### get A_j,j in j = 1, k
    # A_jj = []
    # A_jj.append(np.max(precision_dict_list[0]['per_task_weighted_f1'][0]))
    # for i in range(n_tasks-1):
    #     A_jj.append(np.max(precision_dict_list[i+1]['all_tasks_weighted_f1']))

    # ##### get average forget F_k
    # F_j = []
    # for i in range(n_tasks-1):
    #     F_j.append(A_jj[i] - A_kj[i])
    # F_K = np.mean(F_j)

    # temp_dict['weighted_f1'+'_A_K'] = A_K
    # temp_dict['weighted_f1'+'_F_K'] = F_K
    # temp_dict['weighted_f1'+'_A_kk'] = A_kk



    for metrics in metrics_list:
        ##### get inverse Intransigence A_k,k
        A_kk = np.max(precision_dict_list[-1]['all_tasks_'+metrics])


        ##### get A_k,j in j = 1, k
        A_kj = A_kj_dict[metrics]
        # A_kj.append(A_kk)

        ##### get average metrics A_k
        A_K = np.mean(A_kj)


        ##### get A_j,j in j = 1, k
        A_jj = []
        if args.cl_alg[:3] == 'gem':
            A_jj.append(np.max(precision_dict_list[0]['all_tasks_'+metrics]))
        else:
            if 'per_task_' + metrics in precision_dict_list[0]:
                A_jj.append(np.max(precision_dict_list[0]['per_task_'+metrics][0]))
            else:
                A_jj.append(np.max(precision_dict_list[0]['all_tasks_'+metrics]))
        for i in range(n_tasks-1):
            A_jj.append(np.max(precision_dict_list[i+1]['all_tasks_'+metrics]))

        ##### get average forget F_k
        F_j = []
        for i in range(n_tasks-1):
            F_j.append(A_jj[i] - A_kj[i])
        F_K = np.mean(F_j)

        temp_dict[metrics+'_A_K'] = A_K
        temp_dict[metrics+'_F_K'] = F_K
        temp_dict[metrics+'_A_kk'] = A_kk

        print(metrics+'    A_kj_dict')
        print('A_kj: ', A_kj)
        print('F_j: ', F_j)
        print('A_K: ', A_K)
        print('F_K: ', F_K)
        print('A_kk: ', A_kk)
    return temp_dict


def dict2list(avgAccForgetAkk_l):
    temp = {}
    for metrics in metrics_list:
        temp[metrics] = []
        for i in range(len(avgAccForgetAkk_l)):
            temp[metrics].append(avgAccForgetAkk_l[i][metrics])

        print('\n{}:, Mean: {}, SD: {}'.format(metrics, np.mean(temp[metrics]),np.std(temp[metrics])))
    return temp


def summaryAvgAccForget(avgAccForgetAkk_l):
    temp = {}
    for metrics in metrics_list:
        temp[metrics+'_A_K'] = []
        temp[metrics+'_F_K'] = []
        temp[metrics+'_A_kk'] = []
        for i in range(len(avgAccForgetAkk_l)):
            temp[metrics+'_A_K'].append(avgAccForgetAkk_l[i][metrics+'_A_K'])
            temp[metrics+'_F_K'].append(avgAccForgetAkk_l[i][metrics+'_F_K'])
            temp[metrics+'_A_kk'].append(avgAccForgetAkk_l[i][metrics+'_A_kk'])

        print('\n{}:, Mean: {}, SD: {}'.format(metrics+'_A_K', np.mean(temp[metrics+'_A_K']),np.std(temp[metrics+'_A_K'])))
        print('{}:, Mean: {}, SD: {}'.format(metrics+'_F_K', np.mean(temp[metrics+'_F_K']),np.std(temp[metrics+'_F_K'])))
        print('{}:, Mean: {}, SD: {}\n'.format(metrics+'_A_kk', np.mean(temp[metrics+'_A_kk']),np.std(temp[metrics+'_A_kk'])))
    return temp







def plot_all_methods_wl_exp3(precision_dict_list_list_l,joint, dataset,methods, args=None, exp_setup=''):
    print('\n\n===== plot_all_metrics_summed / Dataset: ' + dataset + ' / Scenario: '+str(3) +' =====\n')
    metrics_list = ['acc','weighted_f1']
    x = [i+1 for i in range( len(precision_dict_list_list_l[0][0]) )]
    for metrics in metrics_list:
        #y1_l_l = []
        y2_l_l = []
        for i, precision_dict_list_list in enumerate(precision_dict_list_list_l):
            #y1_l = []
            y2_l = []
            for j, precision_dict_list in enumerate(precision_dict_list_list):
                A_jj = []
                if methods[i][:3] == 'GEM':
                    A_jj.append(np.max(precision_dict_list[0]['all_tasks_'+metrics]))
                else:
                    if 'per_task_' + metrics in precision_dict_list[0]:
                        A_jj.append(np.max(precision_dict_list[0]['per_task_'+metrics][0]))
                    else:
                        A_jj.append(np.max(precision_dict_list[0]['all_tasks_'+metrics]))
                for jj in range(len(x)-1):
                    A_jj.append(np.max(precision_dict_list[jj+1]['all_tasks_'+metrics]))
                
                    
                    
                #y1 = precision_dict_list[0]['per_task_'+metrics][0]
                #y1_l.append(y1)
                #y2 = precision_dict_list[1]['all_tasks_'+metrics] # d1_d2 acc
                y2_l.append(A_jj)
            #y1_l_l.append(y1_l)
            y2_l_l.append(y2_l)
        plotMethodsWlExp3(rX=x,rY=['',y2_l_l,joint],methods=methods,rStdErrN='depends',
                rFileName='../plots/weighted_methods_'+metrics+'_'
                                    + exp_setup  +'exp'+str(3)+'_'+'lstm'+'_'+ dataset, 
                rLogAxis='no', rXLabel='Epoch',rYLabel=getYLabelName(metrics),epoch=None,metrics=metrics)


def plotMethodsWlExp3(rX,rY,methods,rStdErrN,rFileName,rLogAxis,rXLabel,rYLabel,epoch,metrics):
    x = rX
    y1_l_l = rY[0]
    y2_l_l = rY[1]
    joint = rY[2]
    cnt = 0
    markers_set = ['o', 'v','^','s','<','>','1','2','3','4','d']
    markers = markers_set[:len(methods)]
    # plt.plot(x, [acc_f1_max] * len(x), linestyle='--', color='k',label='Joint', zorder=3)
    print('\n===== basic stats =====\n')
    print('Joint: Max: {} (epoch: {})'.format(
            np.max(joint[0]['per_task_'+metrics][0]), np.argmax(joint[0]['per_task_'+metrics][0])+1
        ))
    # plt.plot(x, joint[0]['per_task_'+metrics][0], linestyle='--', color='k',label='Joint', zorder=3)
    plt.plot(x, [np.max(joint[0]['per_task_'+metrics][0])]*len(x), linestyle='--', color='k', zorder=3)
    # y1_l_ewc = y1_l_l[0]
    for y2_l, method in zip(y2_l_l, methods):
        # if len(method) <= 4:
        #     y1_l = y1_l_ewc
        df_l = []
        desc_l = []
        # y_l = []
        # e_l = []
        
        # df = pd.DataFrame(y1_l)
        # desc = df.describe()
        # df_l.append(df)
        # desc_l.append(desc)
        # print('\n===== y1 =====')
        # print('desc: ', desc)

        # y = desc.iloc[1, :]
        # e = getStdOrStdErr(desc, df, 'sd')
        # y_l.append(y)
        # e_l.append(e)

        # print('\n===== y2 =====')
        df = pd.DataFrame(y2_l)
        desc = df.describe()
        df_l.append(df)
        desc_l.append(desc)
        # print('desc: ', desc)

        y = desc.iloc[1, :]
        e = getStdOrStdErr(desc, df, 'sd')
        # y_l.append(y)
        # e_l.append(e)
        
        
        # y0 = pd.concat([y_l[0],y_l[1]], ignore_index=True)
        print(method+': A_jj:', np.array(y))
        if method == 'None':
            plt.plot(x, y, linestyle='--', color='C3',label=method,zorder=3)
            #plt.plot(x, y, linestyle='--', color='C3',zorder=3)
        else:
            plt.plot(x, y, linestyle='-', color='C'+str(cnt),label=method,marker=markers[cnt], markersize=4, zorder=3,
                markerfacecolor=(1, 1, 1, 1), markeredgewidth=1,  markeredgecolor=(0, 0, 0, .5))
            #plt.plot(x, y, linestyle='-', color='C'+str(cnt),marker=markers[cnt], markersize=4, zorder=3,
            #    markerfacecolor=(1, 1, 1, 1), markeredgewidth=1,  markeredgecolor=(0, 0, 0, .5))
            cnt += 1

    if rLogAxis == 'x':
        plt.xscale('log')
    elif rLogAxis == 'y':
        plt.yscale('log')
    elif rLogAxis == 'xy':
        plt.xscale('log')
        plt.yscale('log')

    plt.grid(True, which="both", ls=":")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.ylabel(rYLabel)
    plt.xlabel(rXLabel)
    legend_properties = {'weight':'bold'}
    
    # leg = plt.legend(loc='best', numpoints=1, ncol=int(len(methods+1)/2), fancybox=True, borderpad=1, framealpha=0.0, prop=legend_properties)
#     leg = plt.legend(loc='lower left', numpoints=1, ncol=int( (len(methods)+2)/2), bbox_to_anchor=(0, 1), prop=legend_properties)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([1-0.05, len(x)+0.05, -0.03, 1.03])
    # plt.axvspan(epoch+1, xmax, 0.0, 1.0, facecolor='gray', alpha=0.2)
    
    
    # step_ = 5 if epoch < 40 else 10
    plt.xticks([i for i in range(1,len(x)+1)])
    if (len(rFileName) > 10) and (rFileName != 'not now'):
        plt.savefig(rFileName + '.pdf', bbox_inches='tight')
    plt.show()

def putTime2List(args):
    args.total_time_l.append(args.total_time)
    args.eval_time_l.append(args.eval_time)
    args.train_time_origin_l.append(args.train_time_origin)
    args.train_time_ewc_update_l.append(args.train_time_ewc_update)
    args.train_time_ewc_loss_l.append(args.train_time_ewc_loss)
    args.train_time_ewc_l.append(args.train_time_ewc_loss+args.train_time_ewc_update)
    args.train_time_si_update_l.append(args.train_time_si_update)
    args.train_time_si_loss_l.append(args.train_time_si_loss)
    args.train_time_si_running_weights_l.append(args.train_time_si_running_weights)
    args.train_time_si_l.append(args.train_time_si_loss+args.train_time_si_update+args.train_time_si_running_weights)
    args.train_time_lwf_update_l.append(args.train_time_lwf_update)
    args.train_time_lwf_loss_l.append(args.train_time_lwf_loss)
    args.train_time_lwf_l.append(args.train_time_lwf_update + args.train_time_lwf_loss)
    args.train_time_icarl_construct_l.append(args.train_time_icarl_construct)
    args.train_time_icarl_construct_class_mean_l.append(args.train_time_icarl_construct_class_mean)
    args.train_time_icarl_construct_argmin_loop_l.append(args.train_time_icarl_construct_argmin_loop)
    args.train_time_icarl_reduce_l.append(args.train_time_icarl_reduce)
    args.train_time_icarl_loss_l.append(args.train_time_icarl_loss)
    args.train_time_icarl_l.append(args.train_time_icarl_construct+args.train_time_icarl_reduce+args.train_time_icarl_loss)
    args.train_time_gem_update_memory_l.append(args.train_time_gem_update_memory)
    args.train_time_gem_compute_gradient_l.append(args.train_time_gem_compute_gradient)
    args.train_time_gem_violation_check_l.append(args.train_time_gem_violation_check)
    args.train_time_gem_l.append(args.train_time_gem_update_memory+args.train_time_gem_compute_gradient+args.train_time_gem_violation_check)

    # args.train_time_other_il = args.train_time_ewc_update + args.train_time_ewc_loss + \
    #         args.train_time_si_update + args.train_time_si_loss + args.train_time_lwf + \
    #         args.train_time_icarl_construct + args.train_time_icarl_reduce + args.train_time_gem
    args.train_time_other_il = args.train_time_ewc_update + args.train_time_ewc_loss + \
            args.train_time_si_update + args.train_time_si_loss + args.train_time_si_running_weights + \
            args.train_time_lwf_update + args.train_time_lwf_loss + \
            args.train_time_icarl_construct + args.train_time_icarl_reduce + args.train_time_icarl_loss + \
            args.train_time_gem_update_memory + args.train_time_gem_compute_gradient + args.train_time_gem_violation_check

    args.train_time_other_il_l.append(args.train_time_other_il)
    args.only_train_time_l.append(args.total_time - args.eval_time - args.train_time_other_il)

    args.total_time=0.0
    args.eval_time=0.0
    args.train_time_origin=0.0
    args.train_time_ewc_loss=0.0
    args.train_time_ewc_update=0.0
    args.train_time_ewc=0.0
    args.train_time_si_loss=0.0
    args.train_time_si_update=0.0
    args.train_time_si_running_weights=0.0
    args.train_time_si=0.0
    args.train_time_lwf_update=0.0
    args.train_time_lwf_loss=0.0
    args.train_time_lwf=0.0
    args.train_time_icarl_construct=0.0
    args.train_time_icarl_construct_class_mean=0.0
    args.train_time_icarl_construct_argmin_loop=0.0
    args.train_time_icarl_reduce=0.0
    args.train_time_icarl_loss=0.0
    args.train_time_icarl=0.0
    args.train_time_gem_update_memory=0.0
    args.train_time_gem_compute_gradient=0.0
    args.train_time_gem_violation_check=0.0
    args.train_time_gem=0.0
    args.train_time_other_il=0.0

def reportTimeMem(model, args, method):
    ### report time
    print('\n===== report time =====')
    print('method: ', method)
    print('total_time: ', args.total_time_l )
    print('eval_time: ', args.eval_time_l )
    print('train_time_ewc: ', args.train_time_ewc_l )
    print('train_time_ewc_update: ', args.train_time_ewc_update_l )
    print('train_time_ewc_loss: ', args.train_time_ewc_loss_l )
    print('train_time_si: ', args.train_time_si_l )
    print('train_time_si_update: ', args.train_time_si_update_l )
    print('train_time_si_loss: ', args.train_time_si_loss_l )
    print('train_time_si_running_weights: ', args.train_time_si_running_weights_l )
    print('train_time_lwf: ', args.train_time_lwf_l )
    print('train_time_lwf_update: ', args.train_time_lwf_update_l )
    print('train_time_lwf_loss: ', args.train_time_lwf_loss_l )
    print('train_time_icarl: ', args.train_time_icarl_l )
    print('train_time_icarl_construct: ', args.train_time_icarl_construct_l )
    print('train_time_icarl_construct_class_mean: ', args.train_time_icarl_construct_class_mean_l )
    print('train_time_icarl_construct_argmin_loop: ', args.train_time_icarl_construct_argmin_loop_l )
    print('train_time_icarl_reduce: ', args.train_time_icarl_reduce_l )
    print('train_time_icarl_loss: ', args.train_time_icarl_loss_l )
    print('train_time_gem: ', args.train_time_gem_l )
    print('train_time_gem_update_memory: ', args.train_time_gem_update_memory_l )
    print('train_time_gem_compute_gradient: ', args.train_time_gem_compute_gradient_l )
    print('train_time_gem_violation_check: ', args.train_time_gem_violation_check_l )

    print('total_time: mean: {}, sd: {}'.format(np.mean(args.total_time_l), np.std(args.total_time_l)) )
    print('eval_time: mean: {}, sd: {}'.format(np.mean(args.eval_time_l), np.std(args.eval_time_l)) )
    print('train_time_ewc: mean: {}, sd: {}'.format(np.mean(args.train_time_ewc_l), np.std(args.train_time_ewc_l)) )
    print('train_time_ewc_update: mean: {}, sd: {}'.format(np.mean(args.train_time_ewc_update_l), np.std(args.train_time_ewc_update_l)) )
    print('train_time_ewc_loss: mean: {}, sd: {}'.format(np.mean(args.train_time_ewc_loss_l), np.std(args.train_time_ewc_loss_l)) )
    print('train_time_si: mean: {}, sd: {}'.format(np.mean(args.train_time_si_l), np.std(args.train_time_si_l)) )
    print('train_time_si_update: mean: {}, sd: {}'.format(np.mean(args.train_time_si_update_l), np.std(args.train_time_si_update_l)) )
    print('train_time_si_loss: mean: {}, sd: {}'.format(np.mean(args.train_time_si_loss_l), np.std(args.train_time_si_loss_l)) )
    print('train_time_si_running_weights: mean: {}, sd: {}'.format(np.mean(args.train_time_si_running_weights_l), np.std(args.train_time_si_running_weights_l)) )
    print('train_time_lwf: mean: {}, sd: {}'.format(np.mean(args.train_time_lwf_l), np.std(args.train_time_lwf_l)) )
    print('train_time_lwf_update: mean: {}, sd: {}'.format(np.mean(args.train_time_lwf_update_l), np.std(args.train_time_lwf_update_l)) )
    print('train_time_lwf_loss: mean: {}, sd: {}'.format(np.mean(args.train_time_lwf_loss_l), np.std(args.train_time_lwf_loss_l)) )
    print('train_time_icarl: mean: {}, sd: {}'.format(np.mean(args.train_time_icarl_l), np.std(args.train_time_icarl_l)) )
    print('train_time_icarl_construct: mean: {}, sd: {}'.format(np.mean(args.train_time_icarl_construct_l), np.std(args.train_time_icarl_construct_l)) )
    print('train_time_icarl_construct_class_mean: mean: {}, sd: {}'.format(np.mean(args.train_time_icarl_construct_class_mean_l), np.std(args.train_time_icarl_construct_class_mean_l)) )
    print('train_time_icarl_construct_argmin_loop: mean: {}, sd: {}'.format(np.mean(args.train_time_icarl_construct_argmin_loop_l), np.std(args.train_time_icarl_construct_argmin_loop_l)) )
    print('train_time_icarl_reduce: mean: {}, sd: {}'.format(np.mean(args.train_time_icarl_reduce_l), np.std(args.train_time_icarl_reduce_l)) )
    print('train_time_icarl_loss: mean: {}, sd: {}'.format(np.mean(args.train_time_icarl_loss_l), np.std(args.train_time_icarl_loss_l)) )
    print('train_time_gem: mean: {}, sd: {}'.format(np.mean(args.train_time_gem_l), np.std(args.train_time_gem_l)) )
    print('train_time_gem_update_memory: mean: {}, sd: {}'.format(np.mean(args.train_time_gem_update_memory_l), np.std(args.train_time_gem_update_memory_l)) )
    print('train_time_gem_compute_gradient: mean: {}, sd: {}'.format(np.mean(args.train_time_gem_compute_gradient_l), np.std(args.train_time_gem_compute_gradient_l)) )
    print('train_time_gem_violation_check: mean: {}, sd: {}'.format(np.mean(args.train_time_gem_violation_check_l), np.std(args.train_time_gem_violation_check_l)) )
    
    print('')
    print('train_time_other_il: ', args.train_time_other_il_l )
    print('train_time_other_il: mean: {}, sd: {}'.format(np.mean(args.train_time_other_il_l), np.std(args.train_time_other_il_l) ) )
    print('only train_time: ', args.only_train_time_l )
    print('only train_time: mean: {}, sd: {}'.format(np.mean(args.only_train_time_l), np.std(args.only_train_time_l) ) )

    ### report memory usage for each method
    model_size = 0.0
    IL_method_size = 0.0
    budget_size = 0.0
    exemplar_means = 0.0
    # model parameters (MB)
    numel_l = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel_l.append(param.data.numel())
    model_size = np.sum(numel_l) * 4 / 1000000
    if method == 'none' or method=='joint':
        IL_method_size = 0.0
    elif method == 'ewc':
        IL_method_size = (2*model_size) * args.tasks
    elif method == 'ewc_online':
        IL_method_size = 2*model_size
    elif method == 'si':
        IL_method_size = 3*model_size
    elif method == 'lwf':
        IL_method_size = model_size
    elif method[:2] == 'ic':
        # icarl exemplars_mean (MB)
        numel_l = []
        for i in range(len(model.exemplar_means)):
            numel = 1
            for dim in model.exemplar_means[i].shape:
                numel *= dim
            numel_l.append(numel)
        exemplar_means = np.sum(numel_l) * 4 / 1000000
        # icarl memory budget (MB)
        numel_l = []
        for i in range(len(model.exemplar_sets)):
            numel = 1
            for dim in model.exemplar_sets[i].shape:
                numel *= dim
            numel_l.append(numel)
        budget_size = np.sum(numel_l) * 4 / 1000000
        # add model size M
        IL_method_size = model_size + budget_size + exemplar_means
    elif method[:2] == 'ge':
        gradient_size = model_size * args.tasks
        numel_memory = model.memory_data.numel()
        numel_labs = model.memory_labs.numel()
        budget_size = (numel_memory * 4 + numel_labs * 8) / 1000000
        IL_method_size = gradient_size + budget_size

    print('\n===== report memory =====')
    print('# tasks: {}'.format(args.tasks))
    print('model_size: {} MB, budget_size: {}, IL_method_size (total): {} MB'.format(model_size, budget_size, IL_method_size))
    if method[:2] == 'ge':
        print('numel_memory: {}, numel_labs: {}'.format(numel_memory, numel_labs))



##### n_classes = 5
class_D2_list_task1_c5 = [
    [0],
    [1],
    [2],
    [3],
    [4]
]
class_D2_list_task1_c5_2 = [
    [0],
    [1]
]

class_D2_list_task2_c5 = [
    [2,3,4],
    [1,2,3],
    [1,3,4],
    [1,2,4],
    [0,3,4],
    [0,2,4],
    [0,2,3],
    [0,1,2],
    [0,1,3],
    [0,1,4]
]
class_D2_list_task2_c5_2 = [
    [2,3,4],
    [1,2,3]
]

##### n_classes = 6
class_D2_list_task1_c6 = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5]
]
class_D2_list_task1_c6_2 = [
    [0],
    [1]
]

class_D2_list_task2_c6 = [
    [0,1,2],
    [0,2,3],
    [0,3,4],
    [0,4,5],
    [1,2,3],
    [1,3,4],
    [1,3,5],
    [1,4,5],
    [2,3,4],
    [3,4,5]
]
class_D2_list_task2_c6_2 = [
    [0,1,2],
    [0,2,3]
]

# class_D2_list_task2_hhar = [
#     [0,1,2],[0,1,3],[0,1,4],[0,1,5],
#     [0,2,3],[0,2,4],[0,2,5],
#     [0,3,4],[0,3,5],
#     [0,4,5],
#     [1,2,3],[1,2,4],[1,2,5],
#     [1,3,4],[1,3,5],
#     [1,4,5],
#     [2,3,4],[2,3,5],[2,4,5],
#     [3,4,5]
# ]


##### n_classes = 12
class_D2_list_task1_c12 = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11]
]
class_D2_list_task1_c12_2 = [
    [0],
    [1]
]

class_D2_list_task2_c12 = [
    [0,1,2,3,4,5],
    [6,7,8,9,10,11],
    [1,2,3,4,5,6],
    [0,2,4,6,8,10],
    [1,3,5,7,9,11],
    [0,3,6,9,1,4],
    [1,4,7,10,2,5],
    [2,5,8,11,3,6],
    [1,2,3,9,10,11],
    [4,5,6,9,10,11]
]
class_D2_list_task2_c12_2 = [
    [0,1,2,3,4,5],
    [6,7,8,9,10,11]
]

##### n_classes = 10
class_D2_list_task1_c10 = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9]
]
class_D2_list_task1_c10_2 = [
    [0],
    [1]
]

class_D2_list_task2_c10 = [
    [0,1,2,3,4],
    [5,6,7,8,9],
    [1,2,3,4,5],
    [0,2,4,6,8],
    [1,3,5,7,9],
    [0,3,6,9,1],
    [1,4,7,9,2],
    [2,5,8,3,6],
    [0,1,3,8,9],
    [4,5,6,8,9]
]
class_D2_list_task2_c10_2 = [
    [0,1,2,3,4],
    [5,6,7,8,9]
]


##### n_classes = 11
class_D2_list_task1_c11 = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
]


class_D2_list_task2_c11 = [
    [0,1,2,3,4],
    [6,7,8,9,10],
    [1,2,3,4,5],
    [0,2,4,6,8],
    [1,3,5,7,9],
    [0,3,6,9,1],
    [1,4,7,10,2],
    [2,5,8,3,6],
    [1,2,3,9,10],
    [4,5,6,9,10]
]


##### n_classes = 14
class_D2_list_task1_c14 = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11],
    [12],
    [13]
]
class_D2_list_task1_c14_2 = [
    [0],
    [1]
]

class_D2_list_task2_c14 = [
    [0,1,2,3,4,5,6],
    [7,8,9,10,11,12,13],
    [0,2,4,6,8,10,12],
    [1,3,5,7,9,11,13],
    [0,3,6,9,12,1,4],
    [1,4,7,13,2,5,8],
    [2,5,8,11,0,3,6],
    [0,4,8,12,2,6,10],
    [1,5,9,13,3,7,11],
    [6,7,8,12,13,4,5]
]
class_D2_list_task2_c14_2 = [
    [0,1,2,3,4,5,6],
    [7,8,9,10,11,12,13]
]



def rev_get_class_D2_list(args):

    if args.rDataset == 'hhar-noaug':
        if args.rExp == 1:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task1_c6_2
            else:
                return class_D2_list_task1_c6
        else:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task2_c6_2
            else:
                return class_D2_list_task2_c6
    elif args.rDataset == 'pamap2':
        if args.rExp == 1:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task1_c12_2
            else:
                return class_D2_list_task1_c12
        else:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task2_c12_2
            else:
                return class_D2_list_task2_c12
    elif args.rDataset == 'skoda' or args.rDataset=='ninapro-db2-c10' or \
            args.rDataset == 'ninapro-db3-c10':
        if args.rExp == 1:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task1_c10_2
            else:
                return class_D2_list_task1_c10
        else:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task2_c10_2
            else:
                return class_D2_list_task2_c10
    elif args.rDataset[:14] == 'emotion-fbanks':
        if args.rExp == 1:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task1_c14_2
            else:
                return class_D2_list_task1_c14
        else:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task2_c14_2
            else:
                return class_D2_list_task2_c14
    elif args.rDataset[:11] == 'emotion-all':
        if args.rExp == 1:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task1_c5_2
            else:
                return class_D2_list_task1_c5
        else:
            if args.rIsSimpleExp == 'yes':
                return class_D2_list_task2_c5_2
            else:
                return class_D2_list_task2_c5
    else:
        assert ("no available datasets")


# saved_model epoch: 
path='../data/saved_model/'
path_weighted_best_model = {
    'exp2':{
        'hhar-noaug':{
            'none':[
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq6_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq8_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq9_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq10_t1_hp2.pth'],
            'ewc':[
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq6_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq8_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq9_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq10_t1_hp2.pth'],
            'ewc_online':[
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq6_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq8_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq9_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq10_t1_hp2.pth'],
            'si':[
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq6_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq8_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq9_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_hhar-noaug_clsSeq10_t1_hp2.pth'],
            'lwf':[
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq6_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq8_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq9_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_hhar-noaug_clsSeq10_t1_hp2.pth'],
            'icarl1':[
                path+'weighted_best_exp2_lstm_icarl1_hhar-noaug_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl1_hhar-noaug_clsSeq2_t1_hp3.pth',
                ],
            'icarl5':[
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq2_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq6_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq9_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl5_hhar-noaug_clsSeq10_t1_hp2.pth'],
            'icarl10':[
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq2_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq6_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq9_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_hhar-noaug_clsSeq10_t1_hp2.pth'],
            'icarl20':[
                path+'weighted_best_exp2_lstm_icarl20_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl20_hhar-noaug_clsSeq2_t1_hp3.pth'
                ],
            'icarl40':[
                path+'weighted_best_exp2_lstm_icarl40_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl40_hhar-noaug_clsSeq2_t1_hp3.pth'
                ],
            'gem1':[ # not yet
                path+'weighted_best_exp2_lstm_gem1_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem1_hhar-noaug_clsSeq2_t1_hp4.pth'],
            'gem5':[
                path+'weighted_best_exp2_lstm_gem5_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem5_hhar-noaug_clsSeq2_t1_hp2.pth'],
            'gem10':[
                path+'weighted_best_exp2_lstm_gem10_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem10_hhar-noaug_clsSeq2_t1_hp2.pth'],
            'gem20':[ # not yet
                path+'weighted_best_exp2_lstm_gem20_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem20_hhar-noaug_clsSeq2_t1_hp4.pth'],
            'gem40':[ # not yet
                path+'weighted_best_exp2_lstm_gem40_hhar-noaug_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem40_hhar-noaug_clsSeq2_t1_hp4.pth']
            },
        'pamap2':{
            'none':[
                path+'weighted_best_exp2_lstm_none_pamap2_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq3_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq4_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq6_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq9_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq10_t1_hp4.pth'],
            'ewc':[
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq3_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq4_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq6_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq9_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq10_t1_hp4.pth'],
            'ewc_online':[
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq3_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq4_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq6_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq9_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq10_t1_hp4.pth'],
            'si':[
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq3_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq4_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq6_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq9_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_pamap2_clsSeq10_t1_hp4.pth'],
            'lwf':[
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq3_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq4_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq6_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq9_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_pamap2_clsSeq10_t1_hp4.pth'],
            'icarl1':[
                path+'weighted_best_exp2_lstm_icarl1_pamap2_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl1_pamap2_clsSeq2_t1_hp2.pth',
                ],
            'icarl5':[
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq3_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq4_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq6_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq8_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq9_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_pamap2_clsSeq10_t1_hp1.pth'],
            'icarl10':[
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq2_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq3_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq4_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq5_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq6_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq7_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq8_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq9_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_pamap2_clsSeq10_t1_hp1.pth'],
            'icarl20':[ # 3 cases
                path+'weighted_best_exp2_lstm_icarl20_pamap2_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl20_pamap2_clsSeq2_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl20_pamap2_clsSeq3_t1_hp1.pth'
                ],
            'icarl40':[
                path+'weighted_best_exp2_lstm_icarl40_pamap2_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl40_pamap2_clsSeq2_t1_hp2.pth'
                ],
            'gem1':[ # not yet
                path+'weighted_best_exp2_lstm_gem1_pamap2_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem1_pamap2_clsSeq2_t1_hp4.pth'],
            'gem5':[
                path+'weighted_best_exp2_lstm_gem5_pamap2_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_gem5_pamap2_clsSeq2_t1_hp2.pth'],
            'gem10':[
                path+'weighted_best_exp2_lstm_gem10_pamap2_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_gem10_pamap2_clsSeq2_t1_hp2.pth'],
            'gem20':[ # not yet
                path+'weighted_best_exp2_lstm_gem20_pamap2_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem20_pamap2_clsSeq2_t1_hp4.pth'],
            'gem40':[ # not yet
                path+'weighted_best_exp2_lstm_gem40_pamap2_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_gem40_pamap2_clsSeq2_t1_hp4.pth']
            },
        'skoda':{
            'none':[
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq2_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq5_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq6_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq7_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq9_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq10_t1_hp4.pth'],
            'ewc':[
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq2_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq5_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq6_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq7_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq9_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq10_t1_hp4.pth'],
            'ewc_online':[
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq2_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq5_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq6_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq7_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq9_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq10_t1_hp4.pth'],
            'si':[
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq2_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq5_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq6_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq7_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq9_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_si_skoda_clsSeq10_t1_hp4.pth'],
            'lwf':[
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq2_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq4_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq5_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq6_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq7_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq9_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_ewc_skoda_clsSeq10_t1_hp4.pth'],
            'icarl1':[
                path+'weighted_best_exp2_lstm_icarl1_skoda_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl1_skoda_clsSeq2_t1_hp4.pth',
                ],
            'icarl5':[
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq2_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq3_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq4_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq5_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq6_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq9_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl5_skoda_clsSeq10_t1_hp3.pth'],
            'icarl10':[
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq2_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq3_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq4_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq5_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq6_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq7_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq8_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq9_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl10_skoda_clsSeq10_t1_hp3.pth'],
            'icarl20':[ # 3 cases
                path+'weighted_best_exp2_lstm_icarl20_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl20_skoda_clsSeq2_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl20_skoda_clsSeq3_t1_hp1.pth'
                ],
            'icarl40':[ # 3 caess
                path+'weighted_best_exp2_lstm_icarl40_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl40_skoda_clsSeq2_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl40_skoda_clsSeq3_t1_hp1.pth'
                ],
            'gem1':[ # not yet
                path+'weighted_best_exp2_lstm_gem1_skoda_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_gem1_skoda_clsSeq2_t1_hp2.pth'],
            'gem5':[
                path+'weighted_best_exp2_lstm_gem5_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem5_skoda_clsSeq2_t1_hp2.pth'],
            'gem10':[
                path+'weighted_best_exp2_lstm_gem10_skoda_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem10_skoda_clsSeq2_t1_hp2.pth'],
            'gem20':[ # not yet
                path+'weighted_best_exp2_lstm_gem20_skoda_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_gem20_skoda_clsSeq2_t1_hp2.pth'],
            'gem40':[ # not yet
                path+'weighted_best_exp2_lstm_gem40_skoda_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_gem40_skoda_clsSeq2_t1_hp2.pth']
            },
        'ninapro-db2-c10':{
            'none':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'ewc':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'ewc_online':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'si':[
                path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq2_t1_hp1.pth'],
            'lwf':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'icarl1':[
                path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'icarl5':[ # 8 cases
                path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq1_t1_hp2.pth',
                path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq2_t1_hp2.pth'],
            'icarl10':[ # 8 cases
                path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq2_t1_hp1.pth'],
            'icarl20':[
                path+'weighted_best_exp2_lstm_icarl20_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl20_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'icarl40':[
                path+'weighted_best_exp2_lstm_icarl40_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl40_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'gem1':[ # not yet
                path+'weighted_best_exp2_lstm_gem1_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_gem1_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'gem5':[ 
                path+'weighted_best_exp2_lstm_gem5_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_gem5_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'gem10':[
                path+'weighted_best_exp2_lstm_gem10_ninapro-db2-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem10_ninapro-db2-c10_clsSeq2_t1_hp1.pth'],
            'gem20':[ # not yet
                path+'weighted_best_exp2_lstm_gem20_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_gem20_ninapro-db2-c10_clsSeq2_t1_hp3.pth'],
            'gem40':[ # not yet
                path+'weighted_best_exp2_lstm_gem40_ninapro-db2-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_gem40_ninapro-db2-c10_clsSeq2_t1_hp3.pth']
            },
        'per-subjectninapro-db2-c10':{
            'none':[
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'ewc':[
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'ewc_online':[
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'si':[
                path+'per-subjectweighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'lwf':[
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'icarl1':[
                path+'per-subjectweighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'icarl5':[ # 8 cases
                path+'per-subjectweighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'icarl10':[ # 8 cases
                path+'per-subjectweighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'icarl20':[
                path+'per-subjectweighted_best_exp2_lstm_icarl20_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_icarl20_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'icarl40':[
                path+'per-subjectweighted_best_exp2_lstm_icarl40_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_icarl40_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'gem1':[ # not yet
                path+'per-subjectweighted_best_exp2_lstm_gem1_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_gem1_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'gem5':[ 
                path+'per-subjectweighted_best_exp2_lstm_gem5_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_gem5_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'gem10':[
                path+'per-subjectweighted_best_exp2_lstm_gem10_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_gem10_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'gem20':[ # not yet
                path+'per-subjectweighted_best_exp2_lstm_gem20_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_gem20_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
            'gem40':[ # not yet
                path+'per-subjectweighted_best_exp2_lstm_gem40_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
                path+'per-subjectweighted_best_exp2_lstm_gem40_ninapro-db2-c10_clsSeq2_t1_hp4.pth']
            },
        'ninapro-db3-c10':{
            'none':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'ewc':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'ewc_online':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'si':[
                path+'weighted_best_exp2_lstm_si_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_si_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'lwf':[
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_ewc_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'icarl1':[
                path+'weighted_best_exp2_lstm_icarl1_ninapro-db3-c10_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl1_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'icarl5':[ # 8 cases
                path+'weighted_best_exp2_lstm_icarl5_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl5_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'icarl10':[ # 8 cases
                path+'weighted_best_exp2_lstm_icarl10_ninapro-db3-c10_clsSeq1_t1_hp3.pth',
                path+'weighted_best_exp2_lstm_icarl10_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'icarl20':[
                path+'weighted_best_exp2_lstm_icarl20_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_icarl20_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'icarl40':[
                path+'weighted_best_exp2_lstm_icarl40_ninapro-db3-c10_clsSeq1_t1_hp4.pth',
                path+'weighted_best_exp2_lstm_icarl40_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'gem1':[ # not yet
                path+'weighted_best_exp2_lstm_gem1_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem1_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'gem5':[ 
                path+'weighted_best_exp2_lstm_gem5_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem5_ninapro-db3-c10_clsSeq2_t1_hp2.pth'],
            'gem10':[
                path+'weighted_best_exp2_lstm_gem10_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem10_ninapro-db3-c10_clsSeq2_t1_hp4.pth'],
            'gem20':[ # not yet
                path+'weighted_best_exp2_lstm_gem20_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem20_ninapro-db3-c10_clsSeq2_t1_hp4.pth'],
            'gem40':[ # not yet
                path+'weighted_best_exp2_lstm_gem40_ninapro-db3-c10_clsSeq1_t1_hp1.pth',
                path+'weighted_best_exp2_lstm_gem40_ninapro-db3-c10_clsSeq2_t1_hp2.pth']
            }
    }
}



# 'ninapro-db2-c10':{
#             'none':[
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq3_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq4_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq8_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq9_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq10_t1_hp4.pth'],
#             'ewc':[
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq3_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq4_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq8_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq9_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq10_t1_hp4.pth'],
#             'ewc_online':[
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq3_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq4_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq8_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq9_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq10_t1_hp4.pth'],
#             'si':[
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq2_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq3_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq4_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq8_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq9_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_si_ninapro-db2-c10_clsSeq10_t1_hp4.pth'],
#             'lwf':[
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq2_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq3_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq4_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq8_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq9_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_ewc_ninapro-db2-c10_clsSeq10_t1_hp4.pth'],
#             'icarl1':[
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq1_t1_hp2.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq2_t1_hp2.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq3_t1_hp1.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq4_t1_hp2.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq8_t1_hp1.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq9_t1_hp1.pth',
#                 path+'weighted_best_exp2_lstm_icarl1_ninapro-db2-c10_clsSeq10_t1_hp4.pth'],
#             'icarl5':[ # 8 cases
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq2_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq3_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq4_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl5_ninapro-db2-c10_clsSeq8_t1_hp4.pth'],
#             'icarl10':[ # 8 cases
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq2_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq3_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq4_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq5_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq6_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq7_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl10_ninapro-db2-c10_clsSeq8_t1_hp4.pth'
#                 ],
#             'icarl20':[
#                 path+'weighted_best_exp2_lstm_icarl20_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl20_ninapro-db2-c10_clsSeq2_t1_hp4.pth'
#                 ],
#             'icarl40':[
#                 path+'weighted_best_exp2_lstm_icarl40_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_icarl40_ninapro-db2-c10_clsSeq2_t1_hp4.pth'
#                 ],
#             'gem1':[ # not yet
#                 path+'weighted_best_exp2_lstm_gem1_ninapro-db2-c10_clsSeq1_t1_hp2.pth',
#                 path+'weighted_best_exp2_lstm_gem1_ninapro-db2-c10_clsSeq2_t1_hp2.pth'],
#             'gem5':[ 
#                 path+'weighted_best_exp2_lstm_gem5_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_gem5_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
#             'gem10':[
#                 path+'weighted_best_exp2_lstm_gem10_ninapro-db2-c10_clsSeq1_t1_hp4.pth',
#                 path+'weighted_best_exp2_lstm_gem10_ninapro-db2-c10_clsSeq2_t1_hp4.pth'],
#             'gem20':[ # not yet
#                 path+'weighted_best_exp2_lstm_gem20_ninapro-db2-c10_clsSeq1_t1_hp2.pth',
#                 path+'weighted_best_exp2_lstm_gem20_ninapro-db2-c10_clsSeq2_t1_hp2.pth'],
#             'gem40':[ # not yet
#                 path+'weighted_best_exp2_lstm_gem40_ninapro-db2-c10_clsSeq1_t1_hp2.pth',
#                 path+'weighted_best_exp2_lstm_gem40_ninapro-db2-c10_clsSeq2_t1_hp2.pth']
#             },
        

##### specify all params which will be fixed after D1 #####
fc_lay_l =[1,2]
fc_lay_l_2 =[1]
fc_units_l =[32,64]
fc_units_l_2 =[64]
lr_l = [0.001]
lr_l_2 = [0.0005,0.0001,0.00001]
batch_l = [32]

##### specify all params which will be changing all the time D1, D2 #####
lr2_l = [0.001, 0.0001]
lr2_l_ewc = [0.0001]

##### specify algorithm-specific params #####
##### EWC: from 10^3 to 10^7 #####
ewc_lambda_l = [1, 10, 100, 1000, 10000, 100000, 1000000]
#     ewc_lambda_l = [100000]

##### Online EWC #####
# ewc_online_lambda_l = [1, 10, 100, 1000, 10000, 100000, 1000000]
ewc_online_lambda_l = [10000]
# ewc_online_gamma_l = [0.5, 1.0]
ewc_online_gamma_l = [0.5]
# ewc_online_gamma_l_2 = [0.2, 0.4, 0.6, 0.8, 1.0]

##### SI #####
si_c_l = [0.2, 0.4, 0.6, 0.8, 1.0]
si_epsilon_l = [0.1]


##### LwF #####
# no parameter... just put 'current' into replay variable, invoke param_D2_lwf function

##### icarl #####
# total # of examples summed up with all classes. default 2000
budget_l = [2000] # for total number of stored data 

##### GEM #####
budget_l = [2000] # for total number of stored data 

##### ref: XDG
gating_prop_l = [0.0]


def test_run():
    # Use cuda?
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device)

if __name__ == '__main__':
    test_run()









