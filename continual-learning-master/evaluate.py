import numpy as np
import torch
import visual_visdom
import visual_plt
import utils
from sklearn.metrics import *

####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             with_exemplars=False, no_task_mask=False, task=None, args=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    allowed_classes_set = set(allowed_classes) if allowed_classes is not None else None
    print('allowed_classes_set: ', allowed_classes_set)
    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
    total_tested = total_correct = 0
    total_labels = torch.Tensor().to(model._device()).long()
    total_predicted = torch.Tensor().to(model._device()).long()
    for data, labels in data_loader: # note that data, labels are torch tensor! not np array!
        # -break on [test_size] (if "None", full dataset is used) ; currently None from precision function
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        # if model.experiment == 'sensor':
        # print('data.shape: ', data.shape)
        # print('labels.shape: ', labels.shape)
        if allowed_classes is not None:
            allowed_idx = [i for i in range(len(labels)) if labels[i].item() in allowed_classes_set]
            data, labels = data[allowed_idx, :], labels[allowed_idx]
            # print('data.shape: ', data.shape)
            # print('labels.shape: ', labels.shape)
        data, labels = data.to(model._device()), labels.to(model._device())
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        with torch.no_grad():
            if with_exemplars:
                predicted = model.classify_with_exemplars(data, allowed_classes=allowed_classes, args=args)
                # - in case of Domain-IL scenario, collapse all corresponding domains into same class
                if max(predicted).item() >= model.classes:
                    predicted = predicted % model.classes
            else:
                scores = model(data) if (allowed_classes is None) else model(data)[:, allowed_classes]
                _, predicted = torch.max(scores, 1)
        # -update statistics

        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
        # concatenate tensors into total tensors
        total_predicted = torch.cat((total_predicted, predicted), dim=0)
        total_labels = torch.cat((total_labels, labels), dim=0)
    # print('total_predicted shape: ', total_predicted.shape)
    # print(torch.unique(total_predicted))
    # print('total_labels: ', total_labels.shape)
    # print(torch.unique(total_labels))

    # calculate 10 different metrics 1: accuracy, 2~10: (weighted macro micro) (f1, prec, rec)
    np_total_predicted = total_predicted.to("cpu").numpy() if model._is_on_cuda() else total_predicted.numpy()
    np_total_labels = total_labels.to("cpu").numpy() if model._is_on_cuda() else total_labels.numpy()
    accuracy = total_correct / total_tested
    metrics_dict = {}
    metrics_dict['acc'] = accuracy
    weighted = precision_recall_fscore_support(np_total_labels, np_total_predicted, average='weighted')
    macro = precision_recall_fscore_support(np_total_labels, np_total_predicted, average='macro')
    micro = precision_recall_fscore_support(np_total_labels, np_total_predicted, average='micro')

    metrics_dict['weighted_f1'] = weighted[2]
    metrics_dict['macro_f1'] = macro[2]
    metrics_dict['micro_f1'] = micro[2]
    metrics_dict['weighted_prec'] = weighted[0]
    metrics_dict['macro_prec'] = macro[0]
    metrics_dict['micro_prec'] = micro[0]
    metrics_dict['weighted_rec'] = weighted[1]
    metrics_dict['macro_rec'] = macro[1]
    metrics_dict['micro_rec'] = micro[1]

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    if verbose:
        print('=> accuracy: {:.3f}'.format(accuracy))
    return metrics_dict


def initiate_precision_dict(n_tasks):
    '''Initiate <dict> with all precision-measures to keep track of.'''
    if n_tasks <= 2:
        precision = {}
        precision["per_task_acc"] = [[] for _ in range(n_tasks)]
        precision["per_task_weighted_f1"] = [[] for _ in range(n_tasks)]
        precision["per_task_macro_f1"] = [[] for _ in range(n_tasks)]
        precision["per_task_micro_f1"] = [[] for _ in range(n_tasks)]
        precision["per_task_weighted_prec"] = [[] for _ in range(n_tasks)]
        precision["per_task_macro_prec"] = [[] for _ in range(n_tasks)]
        precision["per_task_micro_prec"] = [[] for _ in range(n_tasks)]
        precision["per_task_weighted_rec"] = [[] for _ in range(n_tasks)]
        precision["per_task_macro_rec"] = [[] for _ in range(n_tasks)]
        precision["per_task_micro_rec"] = [[] for _ in range(n_tasks)]

        # precision["average"] = []
        precision["x_iteration"] = []
        precision["x_task"] = []

        precision["all_tasks_acc"] = []
        precision["all_tasks_weighted_f1"] = []
        precision["all_tasks_macro_f1"] = []
        precision["all_tasks_micro_f1"] = []
        precision["all_tasks_weighted_prec"] = []
        precision["all_tasks_macro_prec"] = []
        precision["all_tasks_micro_prec"] = []
        precision["all_tasks_weighted_rec"] = []
        precision["all_tasks_macro_rec"] = []
        precision["all_tasks_micro_rec"] = []
    ########################
    ##### n_tasks >= 3 #####
    else: 
        precision = {}
        
        precision["x_iteration"] = []
        precision["x_task"] = []

        precision["all_tasks_acc"] = []
        precision["all_tasks_weighted_f1"] = []
        precision["all_tasks_macro_f1"] = []
        precision["all_tasks_micro_f1"] = []
        precision["all_tasks_weighted_prec"] = []
        precision["all_tasks_macro_prec"] = []
        precision["all_tasks_micro_prec"] = []
        precision["all_tasks_weighted_rec"] = []
        precision["all_tasks_macro_rec"] = []
        precision["all_tasks_micro_rec"] = []

    return precision


def precision(model, datasets, current_task, iteration, classes_per_task=None, scenario="class",
              precision_dict=None, test_size=None, visdom=None, verbose=False, summary_graph=True,
              num_classes_per_task_l=None, experiment=None,
              with_exemplars=False, no_task_mask=False, args=None):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [precision_dict]    None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [scenario]          <str> how to decide which classes to include during evaluating precision
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    n_tasks = len(datasets)-1
    if n_tasks <= 2:
        metrics_dict_list = []
        for i in range(n_tasks):
            if i+1 <= current_task:
                if scenario=='domain':
                    allowed_classes = None
                elif scenario=='task':
                    allowed_classes = list(range(classes_per_task*i, classes_per_task*(i+1)))
                elif scenario=='class':
                    if experiment == 'sensor':
                        allowed_classes = list(range(np.sum(num_classes_per_task_l[:current_task])))
                    else:
                        allowed_classes = list(range(classes_per_task*current_task))
                metrics_dict_list.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                      allowed_classes=allowed_classes, with_exemplars=with_exemplars,
                                      no_task_mask=no_task_mask, task=i+1, args=args))
            else:
                metrics_dict_list.append(0)
        # average_metrics_dict_list = sum([metrics_dict_list[task_id] for task_id in range(current_task)]) / current_task
        if current_task > 1:
            test_total_metrics_dict = validate(model, datasets[-1], test_size=test_size, verbose=verbose,
                                   allowed_classes=allowed_classes, with_exemplars=with_exemplars,
                                   no_task_mask=no_task_mask, task=n_tasks, args=args)
        # Print results on screen
        # if verbose:
        #     print(' => ave precision: {:.3f}'.format(average_metrics_dict_list))

        # Send results to visdom server
        names = ['task {}'.format(i + 1) for i in range(n_tasks)]
        # if visdom is not None:
        #     visual_visdom.visualize_scalars(
        #         metrics_dict_list, names=names, title="precision ({})".format(visdom["graph"]),
        #         iteration=iteration, env=visdom["env"], ylabel="test precision"
        #     )
        #     if n_tasks>1 and summary_graph:
        #         visual_visdom.visualize_scalars(
        #             [average_metrics_dict_list], names=["ave"], title="ave precision ({})".format(visdom["graph"]),
        #             iteration=iteration, env=visdom["env"], ylabel="test precision"
        #         )


        # Append results to [progress]-dictionary and return
        if precision_dict is not None:
            # precision_dict["average"].append(average_metrics_dict_list)
            precision_dict["x_iteration"].append(iteration)
            precision_dict["x_task"].append(current_task)
            for task_id, _ in enumerate(names):
                if task_id+1 <= current_task:
                    precision_dict["per_task_acc"][task_id].append(metrics_dict_list[task_id]['acc'])
                    precision_dict["per_task_weighted_f1"][task_id].append(metrics_dict_list[task_id]['weighted_f1'])
                    precision_dict["per_task_macro_f1"][task_id].append(metrics_dict_list[task_id]['macro_f1'])
                    precision_dict["per_task_micro_f1"][task_id].append(metrics_dict_list[task_id]['micro_f1'])
                    precision_dict["per_task_weighted_prec"][task_id].append(metrics_dict_list[task_id]['weighted_prec'])
                    precision_dict["per_task_micro_prec"][task_id].append(metrics_dict_list[task_id]['micro_prec'])
                    precision_dict["per_task_macro_prec"][task_id].append(metrics_dict_list[task_id]['macro_prec'])
                    precision_dict["per_task_weighted_rec"][task_id].append(metrics_dict_list[task_id]['weighted_rec'])
                    precision_dict["per_task_macro_rec"][task_id].append(metrics_dict_list[task_id]['macro_rec'])
                    precision_dict["per_task_micro_rec"][task_id].append(metrics_dict_list[task_id]['micro_rec'])
            if current_task > 1:
                precision_dict["all_tasks_acc"].append(test_total_metrics_dict['acc'])
                precision_dict["all_tasks_weighted_f1"].append(test_total_metrics_dict['weighted_f1'])
                precision_dict["all_tasks_macro_f1"].append(test_total_metrics_dict['macro_f1'])
                precision_dict["all_tasks_micro_f1"].append(test_total_metrics_dict['micro_f1'])
                precision_dict["all_tasks_weighted_prec"].append(test_total_metrics_dict['weighted_prec'])
                precision_dict["all_tasks_micro_prec"].append(test_total_metrics_dict['micro_prec'])
                precision_dict["all_tasks_macro_prec"].append(test_total_metrics_dict['macro_prec'])
                precision_dict["all_tasks_weighted_rec"].append(test_total_metrics_dict['weighted_rec'])
                precision_dict["all_tasks_macro_rec"].append(test_total_metrics_dict['macro_rec'])
                precision_dict["all_tasks_micro_rec"].append(test_total_metrics_dict['micro_rec'])

    ########################
    ##### n_tasks >= 3 #####
    else:
        metrics_dict_list = []
        for i in range(n_tasks):
            if i+1 <= current_task:
                if scenario=='domain':
                    allowed_classes = None
                elif scenario=='task':
                    allowed_classes = list(range(classes_per_task*i, classes_per_task*(i+1)))
                elif scenario=='class':
                    if experiment == 'sensor':
                        allowed_classes = list(range(np.sum(num_classes_per_task_l[:current_task])))
                    else:
                        allowed_classes = list(range(classes_per_task*current_task))

        # if current_task > 1:
        test_total_metrics_dict = validate(model, datasets[-1], test_size=test_size, verbose=verbose,
                                       allowed_classes=allowed_classes, with_exemplars=with_exemplars,
                                       no_task_mask=no_task_mask, task=i+1, args=args)

        # Send results to visdom server
        names = ['task {}'.format(i + 1) for i in range(n_tasks)]
        
        # Append results to [progress]-dictionary and return
        if precision_dict is not None:
            precision_dict["x_iteration"].append(iteration)
            precision_dict["x_task"].append(current_task)

            # if current_task > 1:
            precision_dict["all_tasks_acc"].append(test_total_metrics_dict['acc'])
            precision_dict["all_tasks_weighted_f1"].append(test_total_metrics_dict['weighted_f1'])
            precision_dict["all_tasks_macro_f1"].append(test_total_metrics_dict['macro_f1'])
            precision_dict["all_tasks_micro_f1"].append(test_total_metrics_dict['micro_f1'])
            precision_dict["all_tasks_weighted_prec"].append(test_total_metrics_dict['weighted_prec'])
            precision_dict["all_tasks_micro_prec"].append(test_total_metrics_dict['micro_prec'])
            precision_dict["all_tasks_macro_prec"].append(test_total_metrics_dict['macro_prec'])
            precision_dict["all_tasks_weighted_rec"].append(test_total_metrics_dict['weighted_rec'])
            precision_dict["all_tasks_macro_rec"].append(test_total_metrics_dict['macro_rec'])
            precision_dict["all_tasks_micro_rec"].append(test_total_metrics_dict['micro_rec'])
    # store precision_dict into model so that we can fetch
    if with_exemplars:
        model.precision_dict_exemplars = precision_dict
    else:
        model.precision_dict = precision_dict
    return precision_dict



####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----GENERATION EVALUATION----####
####-----------------------------####


def show_samples(model, config, pdf=None, visdom=None, size=32, title="Generated images"):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Generate samples from the model
    sample = model.sample(size)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        visual_plt.plot_images_from_tensor(image_tensor, pdf, title=title, nrow=nrow)
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, name='Generated samples ({})'.format(visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)



####--------------------------------------------------------------------------------------------------------------####

####---------------------------------####
####----RECONSTRUCTION EVALUATION----####
####---------------------------------####


def show_reconstruction(model, dataset, config, pdf=None, visdom=None, size=32, task=None, collate_fn=None):
    '''Plot reconstructed examples by an auto-encoder [model] on [dataset], in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Get data
    data_loader = utils.get_data_loader(dataset, size, cuda=model._is_on_cuda(), collate_fn=collate_fn)
    (data, labels) = next(iter(data_loader))
    data, labels = data.to(model._device()), labels.to(model._device())

    # Evaluate model
    with torch.no_grad():
        recon_batch, y_hat, mu, logvar, z = model(data, full=True)

    # Plot original and reconstructed images
    comparison = torch.cat(
        [data.view(-1, config['channels'], config['size'], config['size'])[:size],
         recon_batch.view(-1, config['channels'], config['size'], config['size'])[:size]]
    ).cpu()
    image_tensor = comparison.view(-1, config['channels'], config['size'], config['size'])
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size*2)))
    # -make plots
    if pdf is not None:
        task_stm = "" if task is None else " (task {})".format(task)
        visual_plt.plot_images_from_tensor(
            image_tensor, pdf, nrow=nrow, title="Reconstructions" + task_stm
        )
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, name='Reconstructions ({})'.format(visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)