import torch
import torch.nn as nn
from torch.nn import functional as F
from linear_nets import MLP,fc_layer
from exemplars import ExemplarHandler
from continual_learner import ContinualLearner
from replayer import Replayer
import utils
import time
# Auxiliary functions useful for GEM's inner optimization.
import numpy as np
import quadprog


def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def my_compute_offsets(task, num_classes_per_task_l):
    # active_classes = list(range(sum(num_classes_per_task_l[:(task+1)])))
    offset1 = 0
    offset2 = sum(num_classes_per_task_l[:(task+1)])
    return offset1, offset2

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

class Classifier(ContinualLearner, Replayer, ExemplarHandler):
    '''Model for classifying images, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    def __init__(self, num_features, num_seq, classes,
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", gated=False,
                 bias=True, excitability=None, excit_buffer=False, binaryCE=False, binaryCE_distill=False,
                 experiment='splitMNIST', cls_type='mlp', args=None):

        # configurations
        super().__init__()
        self.num_features = num_features
        self.num_seq = num_seq
        self.classes = classes
        self.label = "Classifier"
        self.fc_layers = fc_layers
        self.hidden_dim = fc_units
        self.layer_dim = fc_layers-1
        self.cuda = None if args is None else args.cuda
        self.device = args.device
        self.weights_per_class = None if args is None else torch.FloatTensor(args.weights_per_class).to(args.device)

        # store precision_dict into model so that we can fetch
        # self.precision_dict_list = [[] for i in range(len(args.num_classes_per_task_l))]
        # self.precision_dict = {}

        # settings for training
        self.binaryCE = binaryCE
        self.binaryCE_distill = binaryCE_distill

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")


        ######------SPECIFY MODEL------######
        self.cls_type = cls_type
        self.experiment = experiment
        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        # fully connected hidden layers
        if experiment=='sensor':
            if self.cls_type == 'mlp':
                self.fcE = MLP(input_size=num_seq * num_features, output_size=fc_units, layers=fc_layers - 1,
                           hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                           excitability=excitability, excit_buffer=excit_buffer, gated=gated)
            elif self.cls_type == 'lstm':
                self.lstm_input_dropout = nn.Dropout(args.input_drop)
                self.lstm = nn.LSTM(input_size=num_features,hidden_size=fc_units,num_layers=fc_layers-1,
                                    dropout=0.0 if (fc_layers-1)==1 else fc_drop, batch_first=True)
                # self.name = "LSTM([{} X {} X {}])".format(num_features, num_seq, classes) if self.fc_layers > 0 else ""
        else:
            self.fcE = MLP(input_size=num_seq*num_features**2, output_size=fc_units, layers=fc_layers-1,
                           hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                           excitability=excitability, excit_buffer=excit_buffer, gated=gated)
        # classifier
        if self.cls_type == 'mlp':
            mlp_output_size = fc_units if fc_layers>1 else num_seq*num_features**2
            self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop)
        elif self.cls_type == 'lstm':
            self.lstm_fc = nn.Linear(fc_units, classes)

        #################
        # +++++ GEM +++++
        #####
        if args.gem:
            print('this is test for GEM ')
            self.margin = args.memory_strength
            self.ce = nn.CrossEntropyLoss()
            self.n_outputs = classes
            self.n_memories = args.n_memories
            self.gpu = args.cuda
            n_tasks = len(args.num_classes_per_task_l)
            # allocate episodic memory
            self.memory_data = torch.FloatTensor(
                n_tasks, self.n_memories, self.num_seq, self.num_features)
            self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
            if args.cuda:
                # self.memory_data = self.memory_data.cuda()
                self.memory_data = self.memory_data.to(self.device)
                # self.memory_labs = self.memory_labs.cuda()
                self.memory_labs = self.memory_labs.to(self.device)

            # allocate temporary synaptic memory
            self.grad_dims = []
            for param in self.parameters():
                self.grad_dims.append(param.data.numel())
            self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
            if args.cuda:
                # self.grads = self.grads.cuda()
                self.grads = self.grads.to(self.device)

            # allocate counters
            self.observed_tasks = []
            self.old_task = -1
            self.mem_cnt = 0

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.cls_type == 'mlp':
            return "{}_c{}".format(self.fcE.name, self.classes)
        elif self.cls_type == 'lstm':
            return "LSTM([{} X {}]_c{})".format(self.num_seq, self.num_features, self.classes)


    def forward(self, x):
        if self.cls_type == 'mlp':
            final_features = self.fcE(self.flatten(x))
            return self.classifier(final_features)
        elif self.cls_type == 'lstm':
            x = self.lstm_input_dropout(x)
            h0, c0 = self.init_hidden(x)
            out, (hn, cn) = self.lstm(x, (h0, c0))
            return self.lstm_fc(out[:, -1, :])
            # lstm_out, hidden = self.lstm(x)
            # print(lstm_out.size())
            # print(x.size())
            # print(lstm_out[-1].size())
            # return self.lstm_fc(lstm_out[-1].view(x.size(0), -1))


    def feature_extractor(self, x):
        if self.cls_type == 'mlp':
            return self.fcE(self.flatten(x))
        elif self.cls_type == 'lstm':
            x = self.lstm_input_dropout(x)
            h0, c0 = self.init_hidden(x)
            out, (hn, cn) = self.lstm(x, (h0, c0))
            return out[:, -1, :]
            # lstm_out, hidden = self.lstm(images)
            # return lstm_out[-1]

    def forward_from_hidden_layer(self, x):
        if self.cls_type == 'mlp': return self.classifier(x)
        elif self.cls_type == 'lstm': return self.lstm_fc(x)

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)] if self.cuda else (h0, c0)

    def train_a_batch(self, x, y, x_=None, y_=None, x_ex=None, y_ex=None,
                    scores=None, scores_=None, rnt=0.5,
                      active_classes=None, num_classes_per_task_l=None, task=1,
                      args=None):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [x_ex]              None or (<list> of) <tensor> batch of exemplars inputs
        [y_ex]              None or (<list> of) <tensor> batch of exemplars inputs' labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        y.long()
        if y_ is not None:
            y_.long()
        if y_ex is not None:
            y_ex.long()

        # Set model to training-mode
        self.train()

        if args.gem:
            start_gem = time.time()
            t = task - 1
            # update memory
            if t != self.old_task:
                self.observed_tasks.append(t)
                self.old_task = t

            # Update ring buffer storing examples from current task
            bsz = y.data.size(0)
            endcnt = min(self.mem_cnt + bsz, self.n_memories)
            effbsz = endcnt - self.mem_cnt
            self.memory_data[t, self.mem_cnt: endcnt].copy_(
                x.data[: effbsz])
            if bsz == 1:
                self.memory_labs[t, self.mem_cnt] = y.data[0]
            else:
                self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                    y.data[: effbsz])
            self.mem_cnt += effbsz
            if self.mem_cnt == self.n_memories:
                self.mem_cnt = 0
            args.train_time_gem_update_memory += time.time() - start_gem

            start_gem = time.time()
            # compute gradient on previous tasks
            if len(self.observed_tasks) > 1:
                # print('self.observed_tasks: ', self.observed_tasks)
                for tt in range(len(self.observed_tasks) - 1):
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]

                    offset1, offset2 = my_compute_offsets(task=past_task, num_classes_per_task_l=num_classes_per_task_l)
                    # ptloss = self.ce(
                    #     input=self.forward(self.memory_data[past_task])[:, offset1: offset2],
                    #     target=self.memory_labs[past_task] - offset1)
                    ptloss = F.cross_entropy(
                            input=self.forward(self.memory_data[past_task])[:, offset1: offset2],
                            target=self.memory_labs[past_task] - offset1,
                            weight=self.weights_per_class[offset1:offset2])
                    ptloss.backward()
                    store_grad(self.parameters, self.grads, self.grad_dims,
                               past_task)

            args.train_time_gem_compute_gradient += time.time() - start_gem
            # now compute the grad on the current minibatch
            self.zero_grad()

            # print(t, num_classes_per_task_l)
            offset1, offset2 = my_compute_offsets(task=t, num_classes_per_task_l=num_classes_per_task_l)
            # print(self.forward(x)[:, offset1: offset2].size())
            # print(y.size())
            # loss = self.ce(
            #     input=self.forward(x)[:, offset1: offset2],
            #     target=y - offset1)
            loss = F.cross_entropy(
                input=self.forward(x)[:, offset1: offset2],
                target=y - offset1,
                weight=self.weights_per_class[offset1:offset2])
            loss.backward()

            # check if gradient violates constraints
            start_gem = time.time()
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                # indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                #     else torch.LongTensor(self.observed_tasks[:-1])
                indx = torch.LongTensor(self.observed_tasks[:-1]).to(self.device) if self.gpu \
                    else torch.LongTensor(self.observed_tasks[:-1])
                # print(indx)
                dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads[:, t].unsqueeze(1),
                                  self.grads.index_select(1, indx), self.margin)
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, t],
                                   self.grad_dims)
            args.train_time_gem_violation_check += time.time() - start_gem
            self.optimizer.step()

            return {
                'loss_total': loss.item(),
                'loss_current': 0,
                'loss_replay': 0,
                'pred': 0,
                'pred_r': 0,
                'distil_r': 0,
                'ewc': 0., 'si_loss': 0.,
                'precision': 0.,
            }
        else:
            # Reset optimizer
            self.optimizer.zero_grad()
            ##--(1)-- CURRENT DATA --##

            if x is not None:
                # If requested, apply correct task-specific mask
                if self.mask_dict is not None:
                    self.apply_XdGmask(task=task)

                # Run model
                if args.augment < 0:
                    y_hat = self(x)
                else:
                    features = self.feature_extractor(x)
                    features_ex = self.feature_extractor(x_ex)
                    # y_hat = self.forward_from_hidden_layer(features)
                    # y_hat_ex = self.forward_from_hidden_layer(features_ex)

                #####
                ##### perform augmentation on feature space #####
                #####
                if args.augment == 0: # random augmentation

                    features = torch.cat([features, features_ex])
                    y = torch.cat([y, y_ex])
                    # now let's add some noise!
                    for i in range(args.scaling-1):
                        features = torch.cat([features, features_ex+torch.randn(features_ex.shape).to(self.device)*args.sd])
                        y = torch.cat([y, y_ex])
                        if features.shape[0] > args.batch:
                            features = features[:args.batch,:]
                            y = y[:args.batch]
                            break

                    y_hat = self.forward_from_hidden_layer(features)
                elif args.augment == 1: # feature augmentation based on standard deviation
                    pass
                elif args.augment == 2: # feature augmentation based on SMOTE method
                    pass

                # -if needed, remove predictions for classes not in current task
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                    y_hat = y_hat[:, class_entries]
                    # if args.augment >= 0:
                    #     y_hat_ex = y_hat_ex[:, class_entries]

                # Calculate prediction loss
                if self.binaryCE: # ICARL goes into this. binaryCE is True.
                    # -binary prediction loss
                    # print('x.shape: ', x.shape)
                    # print('y.shape: ', y.shape)
                    # print('y_hat.shape: ', y_hat.shape)
                    # print('y: ', y)
                    start_icarl = time.time()
                    binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                    # binary_targets = [0, 0, 1, ... , 0] <=> class = 2
                    # print(binary_targets.size()) # [128 x 17]
                    if self.binaryCE_distill and (scores is not None): # ICARL does not go into this cuz scores is None
                        if args.experiment == 'sensor':
                            binary_targets = binary_targets[:, sum(num_classes_per_task_l[:(task-1)]):]
                        else:
                            classes_per_task = int(y_hat.size(1) / task)
                            binary_targets = binary_targets[:, -(classes_per_task):]
                        #     print(classes_per_task) # 8
                        # print(binary_targets.size()) # [128 x 1]
                        binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
                        # print(binary_targets.size()) # [128 x 17] => this supposed to be [128 x 17 (16 + 1)]
                        # print(scores.size()) # [128 x 16]
                        # print(self.KD_temp) # 2
                    predL = None if y is None else F.binary_cross_entropy_with_logits(
                        input=y_hat, target=binary_targets, reduction='none'
                    ).sum(dim=1).mean()     #--> sum over classes, then average over batch
                    args.train_time_icarl_loss += time.time() - start_icarl
                else:
                    # -multiclass prediction loss
                    # print("x", x.shape, x)
                    # print("y_hat", y_hat.shape, y_hat)
                    # print("y", y.shape, y)
                    predL = None if y is None else F.cross_entropy(
                        input=y_hat,target=y,weight=self.weights_per_class[class_entries],reduction='elementwise_mean')

                # Weigh losses
                loss_cur = predL

                # Calculate training-precision
                precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

                # If XdG is combined with replay, backward-pass needs to be done before new task-mask is applied
                if (self.mask_dict is not None) and (x_ is not None):
                    weighted_current_loss = rnt*loss_cur
                    weighted_current_loss.backward()
            else:
                precision = predL = None
                # -> it's possible there is only "replay" [i.e., for offline with incremental task learning]


            ##--(2)-- REPLAYED DATA --##

            if x_ is not None:
                # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
                # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
                start_lwf = time.time()

                TaskIL = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
                if not TaskIL:
                    y_ = [y_]
                    scores_ = [scores_]
                    active_classes = [active_classes] if (active_classes is not None) else None
                n_replays = len(y_) if (y_ is not None) else len(scores_)

                # Prepare lists to store losses for each replay
                loss_replay = [None]*n_replays
                predL_r = [None]*n_replays
                distilL_r = [None]*n_replays

                # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
                if (not type(x_)==list) and (self.mask_dict is None):
                    y_hat_all = self(x_)

                # Loop to evalute predictions on replay according to each previous task
                for replay_id in range(n_replays):

                    # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                    if (type(x_)==list) or (self.mask_dict is not None):
                        x_temp_ = x_[replay_id] if type(x_)==list else x_
                        if self.mask_dict is not None:
                            self.apply_XdGmask(task=replay_id+1)
                        y_hat_all = self(x_temp_)

                    # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in replayed task
                    y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                    # Calculate losses
                    if (y_ is not None) and (y_[replay_id] is not None):
                        if self.binaryCE:
                            binary_targets_ = utils.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(y_[replay_id].device)
                            predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                                input=y_hat, target=binary_targets_, reduction='none'
                            ).sum(dim=1).mean()     #--> sum over classes, then average over batch
                        else:
                            predL_r[replay_id] = F.cross_entropy(
                                input=y_hat, target=y_[replay_id], weight=self.weights_per_class[active_classes[replay_id]], reduction='elementwise_mean')
                    if (scores_ is not None) and (scores_[replay_id] is not None):
                        # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                        n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
                        kd_fn = utils.loss_fn_kd_binary if self.binaryCE else utils.loss_fn_kd
                        distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                     target_scores=scores_[replay_id], T=self.KD_temp)
                    # Weigh losses
                    if self.replay_targets=="hard":
                        loss_replay[replay_id] = predL_r[replay_id]
                    elif self.replay_targets=="soft":
                        loss_replay[replay_id] = distilL_r[replay_id]

                    # If task-specific mask, backward pass needs to be performed before next task-mask is applied
                    if self.mask_dict is not None:
                        weighted_replay_loss_this_task = (1 - rnt) * loss_replay[replay_id] / n_replays
                        weighted_replay_loss_this_task.backward()
                args.train_time_lwf_loss += time.time() - start_lwf

            # Calculate total loss with replay loss if it exists.
            if x_ is None:
                loss_replay = None
            else:
                start_lwf = time.time()
                loss_replay = sum(loss_replay)/n_replays
                args.train_time_lwf_loss += time.time() - start_lwf
            if x is None:
                start_lwf = time.time()
                loss_total = loss_replay
                args.train_time_lwf_loss += time.time() - start_lwf
            else:
                if x_ is None:
                    loss_total = loss_cur
                else:
                    start_lwf = time.time()
                    loss_total = rnt*loss_cur+(1-rnt)*loss_replay
                    args.train_time_lwf_loss += time.time() - start_lwf

            # loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays
            # loss_total = loss_replay if (x is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)


            ##--(3)-- ALLOCATION LOSSES --##

            # Add SI-loss (Zenke et al., 2017)
            if self.si_c>0:
                start_si = time.time()
                surrogate_loss = self.surrogate_loss()
                loss_total += self.si_c * surrogate_loss
                args.train_time_si_loss += time.time() - start_si


            # Add EWC-loss
            if self.ewc_lambda>0:
                start_ewc = time.time()
                ewc_loss = self.ewc_loss()
                loss_total += self.ewc_lambda * ewc_loss
                args.train_time_ewc_loss += time.time() - start_ewc


            # Backpropagate errors (if not yet done)
            if (self.mask_dict is None) or (x_ is None):
                loss_total.backward()
            # Take optimization-step
            self.optimizer.step()


            # Return the dictionary with different training-loss split in categories
            return {
                'loss_total': loss_total.item(),
                'loss_current': loss_cur.item() if x is not None else 0,
                'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
                'pred': predL.item() if predL is not None else 0,
                'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
                'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
                'ewc': ewc_loss.item() if self.ewc_lambda > 0 else 0.0, 
                'si_loss': surrogate_loss.item() if self.si_c > 0 else 0.0,
                'precision': precision if precision is not None else 0.,
            }



