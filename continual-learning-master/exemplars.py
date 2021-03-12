import abc
import torch
from torch import nn
from torch.nn import functional as F
import utils
import copy
import numpy as np
import time
import heapq
from functools import total_ordering
from collections import namedtuple

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

@total_ordering
class MaxHeapObj(object):
    def __init__(self, idx, dist): self.idx, self.dist = idx, dist
    def __eq__(self, other): return self.dist == other.dist
    def __lt__(self, other): return self.dist > other.dist

class MinHeap(object):
    def __init__(self): self.h = []
    def heappush(self, x): heapq.heappush(self.h, x)
    def heappop(self): return heapq.heappop(self.h)
    def __getitem__(self, i): return self.h[i]
    def __len__(self): return len(self.h)

class MaxHeap(MinHeap):
    def heappush(self, idx, dist): heapq.heappush(self.h, MaxHeapObj(idx, dist))
    def heappop(self): 
        maxHeapObj = heapq.heappop(self.h)
        return maxHeapObj.idx, maxHeapObj.dist
    def __getitem__(self, i): return self.h[i].idx, self.h[i].dist

class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self):
        super().__init__()

        # list with exemplar-sets
        self.quantize = 32
        self.exemplar_sets = []   #--> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True

        # settings
        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    def init_exemplar_set(self, num_classes, quantize=32):
        self.quantize = quantize
        # for i in range(num_classes):
        #     self.exemplar_sets.append([])

    def init_exemplar_dict(self):
        self.exemplar_dict = {}
        self.parallel_flag = True

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n, class_id, args=None):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        if args.parallel <= 1:
            self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding == 3: # FastICARL (Parallel.)
            pass
            # compute features in parallel 
            #### Within Child Process ####
            # compute features_sum in each child process

            #### In Mother Process ####
            # sum up features_sum in mother process

            # divide by # child processes to get class_mean, i.e., features_mean
            # normalize class_mean

            #### Within Child Process ####
            # loop over features in each child process 

            # retain m exemplars (k = min(n, n_max or # features in Child Process))

            #### In Mother Process ####
            # Combine m x #ChildProcesses exemplars from each child process
            # select m exemplars
            # sort m exemplars

            #### let's compare FastICARL (Parallel.) produces the same m exemplars as FastICARL (Serial.)


        elif self.herding == 1 or self.herding == 2:
            # compute features for each example in [dataset]
            start = time.time()

            first_entry = True
            dataloader = utils.get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for (image_batch, y) in dataloader:
                image_batch = image_batch.to(self._device())
                # print('image_batch on cuda?', image_batch.is_cuda)
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch)
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            args.train_time_icarl_construct_class_mean += time.time() - start
            # print('features on cuda?', features.is_cuda)
            # print('class_mean on cuda?', class_mean.is_cuda)

            start = time.time()
            set_of_selected = set()
            if self.herding == 1:
                # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
                exemplar_features = torch.zeros_like(features[:min(n, n_max)])
                # print('exemplar_features on cuda?', exemplar_features.is_cuda)
                # list_of_selected = []
                args.exemplars_sets_indexes[class_id] = [-1]*min(n, n_max)
                for k in range(min(n, n_max)):
                    if k>0:
                        exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                        features_means = (features + exemplar_sum)/(k+1)
                        # if self.norm_exemplars:
                        #     # perhaps this normalization should be left out??
                        #     features_means = F.normalize(features_means, p=2, dim=1)
                        features_dists = features_means - class_mean
                    else:
                        features_dists = features - class_mean
                    # print('features_dists on cuda?', features_dists.is_cuda)
                    index_selected = np.argmin((torch.norm(features_dists, p=2, dim=1).cpu())).item()
                    # if index_selected in list_of_selected:
                    if index_selected in set_of_selected:
                        raise ValueError("Exemplars should not be repeated!!!!")
                    # list_of_selected.append(index_selected)
                    set_of_selected.add(index_selected)

                    # print('set_of_selected: ', set_of_selected)
                    # print('class_id: ', class_id,' / index_selected: ', index_selected)
                    if args.quantize == 32:
                        exemplar_set.append(dataset[index_selected][0].numpy())
                    elif args.quantize == 16:
                        exemplar_set.append(dataset[index_selected][0].numpy().astype(np.float16))
                    elif args.quantize < 16:
                        exemplar_set.append(args.quantizer.quantize_tensor( dataset[index_selected][0].to(args.device),
                                                                            num_bits=args.quantize,
                                                                            min_val=args.quantizer.min_val,
                                                                            max_val=args.quantizer.max_val) )
                        # quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

                    exemplar_features[k] = copy.deepcopy(features[index_selected])
                    args.exemplars_sets_indexes[class_id][k] = index_selected

                    # make sure this example won't be selected again
                    features[index_selected] = features[index_selected] + 10000
            else: # when self.herding == 2 : FastICARL (Serial Ver.)
                features_dists = features - class_mean
                dists = torch.norm(features_dists, p=2, dim=1).cpu()
                # Loop over all dataset X & retain 'k' exemplars in max heap
                k = min(n, n_max)
                maxh = MaxHeap()
                for idx in range(0,k):
                    maxh.heappush(idx, dists[idx])

                for idx in range(k, len(dists)):
                    if dists[idx] < maxh[0][1]: # if dist is smaller than max value in maxHeap,
                        maxh.heappop()  # then replace it, i.e., pop one element
                        maxh.heappush(idx, dists[idx])    # push the new element

                # Sort 'k' exemplars in max heap & put them into a exemplar_set
                # initialize exemplar_set in advance
                for i in range(len(maxh)):
                    exemplar_set.append([])

                args.exemplars_sets_indexes[class_id] = [-1]*len(maxh)
                # Note that iterate from the last element of exemplar_set cuz heappop gives the max dist
                for i in range(len(maxh)-1, -1, -1):
                    index_selected, dist = maxh.heappop()
                    if args.quantize == 32:
                        exemplar_set[i] = dataset[index_selected][0].numpy()
                    elif args.quantize == 16:
                        exemplar_set[i] = dataset[index_selected][0].numpy().astype(np.float16)
                    elif args.quantize < 16:
                        exemplar_set[i] = args.quantizer.quantize_tensor(dataset[index_selected][0].to(args.device),
                                                           num_bits=args.quantize,
                                                           min_val=args.quantizer.min_val,
                                                           max_val=args.quantizer.max_val)

                    args.exemplars_sets_indexes[class_id][i] = index_selected
                    # print('class_id: ', class_id,' / index_selected: ', index_selected)
                # print('class_id: ', class_id,' / index_selected_list: ',args.exemplars_sets_indexes[class_id])
            # args.exemplars_sets_indexes[class_id] = list(set_of_selected)
            args.train_time_icarl_construct_argmin_loop += time.time() - start
        else:
            start = time.time()

            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            args.exemplars_sets_indexes[class_id] = list(indeces_selected)
            for k in indeces_selected:
                if args.quantize == 32:
                    exemplar_set.append(dataset[k][0].numpy())
                elif args.quantize == 16:
                    exemplar_set.append(dataset[k][0].numpy().astype(np.float16))
                elif args.quantize < 16:
                    exemplar_set.append(args.quantizer.quantize_tensor(dataset[k][0].to(args.device),
                                                                       num_bits=args.quantize,
                                                                       min_val=args.quantizer.min_val,
                                                                       max_val=args.quantizer.max_val))

                # print('class_id: ', class_id,' / index_selected: ', k)
                # print(dataset[k][0].size()) # [24 x 113] [# seq x # features]
                # print(dataset[k].size())
                # break

            args.train_time_icarl_construct_argmin_loop += time.time() - start


        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]

        if args.parallel <= 1 or args.D1orD2 > 1:
            if args.quantize >= 16:
                if len(self.exemplar_sets) > class_id:
                    self.exemplar_sets[class_id] = np.array(exemplar_set)
                else:
                    self.exemplar_sets.append(np.array(exemplar_set))
            else: # quantize even smaller than 16 bits / don't convert to numpy but stay as tensor all the time
                if len(self.exemplar_sets) > class_id:
                    self.exemplar_sets[class_id] = torch.stack(exemplar_set).to("cpu")
                else:
                    # print(len(exemplar_set))
                    # temp = args.quantizer.dequantize_tensor(QTensor(tensor=exemplar_set[0], scale=args.quantizer.scale, zero_point=args.quantizer.zero_point))
                    # print(exemplar_set.shape)
                    self.exemplar_sets.append(torch.stack(exemplar_set).to("cpu"))
            # self.exemplar_sets[class_id] = np.array(exemplar_set)
            # print("adding more classes to exemplar sets")
            # print("len of sets: {} , class_id: {}".format(len(self.exemplar_sets[class_id]), class_id))
        else:
            if args.quantize >= 16:
                self.exemplar_dict[class_id] = np.array(exemplar_set)
                print("adding more classes to exemplar dict")
                print("len of dict: {} , class_id: {}".format(len(self.exemplar_dict[class_id]), class_id))
            else:
                self.exemplar_dict[class_id] = torch.stack(exemplar_set).to("cpu")
        # set mode of model back
        if args.parallel <= 1:
            self.train(mode=mode)

    def construct_exemplar_set_parallel(self, dataset, n, class_id, args=None):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        if args.parallel <= 1:
            self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding == 3: # FastICARL (Parallel.)
            pass
            # compute features in parallel 
            #### Within Child Process ####
            # compute features_sum in each child process

            #### In Mother Process ####
            # sum up features_sum in mother process

            # divide by # child processes to get class_mean, i.e., features_mean
            # normalize class_mean

            #### Within Child Process ####
            # loop over features in each child process 

            # retain m exemplars (k = min(n, n_max or # features in Child Process))

            #### In Mother Process ####
            # Combine m x #ChildProcesses exemplars from each child process
            # select m exemplars
            # sort m exemplars

            #### let's compare FastICARL (Parallel.) produces the same m exemplars as FastICARL (Serial.)

        # set mode of model back
        if args.parallel <= 1:
            self.train(mode=mode)

    ####----CLASSIFICATION----####

    def classify_with_exemplars(self, x, allowed_classes=None, args=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        if self.compute_means:
            exemplar_means = []  #--> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            # if self.parallel_flag == True:
            #     for allowed_class in range(le):
            #         P_y = self.exemplar_sets[allowed_class]
            #         # Collect all exemplars in P_y into a <tensor> and extract their features
            #         exemplars = []
            #         for ex in P_y:
            #             exemplars.append(torch.from_numpy(ex))
            #         exemplars = torch.stack(exemplars).to(self._device())
            #         with torch.no_grad():
            #             features = self.feature_extractor(exemplars)
            #         if self.norm_exemplars:
            #             features = F.normalize(features, p=2, dim=1)
            #         # Calculate their mean and add to list
            #         mu_y = features.mean(dim=0, keepdim=True)
            #         if self.norm_exemplars:
            #             mu_y = F.normalize(mu_y, p=2, dim=1)
            #         exemplar_means.append(mu_y.squeeze())       # -> squeeze removes all dimensions of size 1
            # else:
            for P_y in self.exemplar_sets:
                if len(P_y) == 0:
                    break
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                for ex in P_y:
                    if self.quantize == 32:
                        exemplars.append(torch.from_numpy(ex))
                    elif self.quantize == 16:
                        exemplars.append(torch.from_numpy(ex.astype(np.float32)))
                    elif self.quantize < 16:
                        exemplars.append(args.quantizer.dequantize_tensor(QTensor(tensor=ex, scale=args.quantizer.scale, zero_point=args.quantizer.zero_point)) )
                        # print(" ")
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())       # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # print('exemplars: exemplar mean:', len(self.exemplar_means))
        # print('exemplars: allowed_classes:', allowed_classes)
        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)        # (n_classes, feature_size)
        # print(means.size()) # [16 x 64]
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        # print(means.size()) # [128 x 16 x 64]
        means = means.transpose(1, 2)              # (batch_size, feature_size, n_classes)
        # print(means.size()) # [128 x 64 x 16]
        # Extract features for input data (and reorganize)
        # print(x.size()) # [128 x 24 x 113]
        with torch.no_grad():
            feature = self.feature_extractor(x)    # (batch_size, feature_size)
        # print(feature.size())  # [128 x 64]
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)             # (batch_size, feature_size, 1)
        # print(feature.size()) # [128 x 64 x 1]
        feature = feature.expand_as(means)         # (batch_size, feature_size, n_classes)
        # print(feature.size())  # [128 x 64 x 16]

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze(1)  # (batch_size, n_classes)
        _, preds = dists.min(1)
        # print(preds.size())


        # Set mode of model back
        self.train(mode=mode)

        return preds

