import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torch
from scipy.io import loadmat
from sliding_window import sliding_window
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight 
from collections import namedtuple

def reshape(x):
    return x.reshape((x.shape[0],) + (1,) + x.shape[1:])

def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name=='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

#----------------------------------------------------------------------------------------------------------#

def makeClassSetD1(class_set_D2, num_classes):
    class_list = []
    for i in range(num_classes):
        if i not in class_set_D2:
            class_list.append(i)
    return set(class_list)


def makePermu(classes_per_task_list):
    temp = []
    for classes_per_task in classes_per_task_list:
        classes_l = list(classes_per_task)
        classes_l.sort()
        for cls in classes_l:
            temp.append(cls)
    # print('temp: ', temp)
    # flip index and class
    permu = np.zeros(len(temp), dtype = int)
    for i, cls in enumerate(temp):
        permu[cls] = i
    print('permu: ', permu)
    return permu

def applyPermu(weights_per_class, permu):
    print('weights_per_class: ', weights_per_class)
    temp = np.zeros(len(weights_per_class))
    for i in range(len(weights_per_class)):
        temp[permu[i]] = weights_per_class[i]
    print('permuted weights_per_class: ', temp)
    return temp

def opp_sliding_window(data_x, data_y, ws, ss):
    """
    Obtaining the windowed data from the HAR data
    :param data_x: sensory data
    :param data_y: labels
    :param ws: window size
    :param ss: stride
    :return: windows from the sensory data (based on window size and stride)
    """
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.reshape(data_y, (len(data_y), ))  # Just making it a vector if it was a 2D matrix
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)



def load_and_segment_data(name, config=None, subject_idx=None, exp_setup=None):
    # default window size would be 1s, and stride size is 50%.
    if config is not None:
        window_size = config['seq']
        stride_size = int(window_size / 2)

    if name == 'opportunity':
        path = '../DeepConvLSTM-master/'
        X_train = np.load(path+"data/opp_X_train.npy")
        X_test = np.load(path+"data/opp_X_test.npy")
        y_train = np.load(path+"data/opp_y_train.npy")
        y_test = np.load(path+"data/opp_y_test.npy")
        # exclude null class
        X_train = X_train[y_train[:] > 0]
        y_train = y_train[y_train[:] > 0]
        y_train = y_train[:] - 1
        X_test = X_test[y_test[:] > 0]
        y_test = y_test[y_test[:] > 0]
        y_test = y_test[:] - 1
    elif name[:4] == 'nina':
        path = '../data/emg/ninapro/'
        # if exp_setup==None:
        #     if name[:11] == 'ninapro-db2':
        #         full_path = path + 'db2/processed/'
        #     elif name[:11] == 'ninapro-db3':
        #         full_path = path + 'db3/processed/'
        #     else:
        #         full_path = ''
        # elif exp_setup=='leave-one-user-out':
        if 'ninapro-db2-c50' in name:
            full_path = path + 'db2/processed_c50_2000Hz/'
            data_raw = load_dataset_ninapro(full_path, name, subject_idx, exp_setup)
            X_train, y_train = data_raw['train_data'], data_raw['train_labels']
            X_eval, y_eval = data_raw['val_data'], data_raw['val_labels']
            X_test, y_test = data_raw['test_data'], data_raw['test_labels']
            X_train = np.concatenate((X_train, X_eval), axis=0)
            y_train = np.concatenate((y_train, y_eval), axis=0)
            return X_train, y_train, X_test, y_test
        elif name[:11] == 'ninapro-db2':
            full_path = path + 'db2/processed_2/'
        elif name[:11] == 'ninapro-db3':
            full_path = path + 'db3/processed_2/'
        else:
            full_path = ''
        data_raw = load_dataset_ninapro(full_path, name, subject_idx, exp_setup)
        # Obtaining the segmented data
        X_train, y_train = opp_sliding_window(data_raw['train' + '_data'], data_raw['train' + '_labels'],
                                                    window_size, stride_size)
        X_eval, y_eval = opp_sliding_window(data_raw['val' + '_data'], data_raw['val' + '_labels'],
                                          window_size, stride_size)
        X_test, y_test = opp_sliding_window(data_raw['test' + '_data'], data_raw['test' + '_labels'],
                                          window_size, stride_size)
        # Stacking train and eval sets
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_train = np.concatenate((y_train, y_eval), axis=0)
    elif name[:4] == 'emot':
        path = '../data/audio_emotion/Splits/Emotions-all/'
        full_path = path + name + '.mat'

        # in audio dataset, we don't do overlapping so we can save more steps in LSTMs.
        stride_size = window_size

        data_raw = load_dataset(full_path)
        # Obtaining the segmented data
        X_train, y_train = opp_sliding_window(data_raw['train' + '_data'], data_raw['train' + '_labels'],
                                                    window_size, stride_size)
        X_eval, y_eval = opp_sliding_window(data_raw['val' + '_data'], data_raw['val' + '_labels'],
                                          window_size, stride_size)
        X_test, y_test = opp_sliding_window(data_raw['test' + '_data'], data_raw['test' + '_labels'],
                                          window_size, stride_size)
        # Stacking train and eval sets
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_train = np.concatenate((y_train, y_eval), axis=0)
    elif name[:4] == 'hhar':
        path = '../data/hhar/'
        if name == 'hhar-noaug':
            full_path = path + 'noaug_leave_one_user_out/'
        elif name == 'hhar-aug':
            full_path = path + 'aug_leave_one_user_out/'
        elif name == 'hhar-raw':
            full_path = path + 'raw_leave_one_user_out/'
        else:
            full_path = ''
        X_train = np.load(full_path + "a_train_X.npy").astype(np.float32)
        X_test = np.load(full_path + "a_test_X.npy").astype(np.float32)
        y_train = np.load(full_path + "a_train_y.npy").astype(np.uint8)
        y_test = np.load(full_path + "a_test_y.npy").astype(np.uint8)
    else:
        path = '../data/data_harish/'
        if name == 'pamap2':
            full_path = path + 'pamap2/PAMAP2.mat'
        elif name == 'skoda':
            full_path = path + 'skoda/Skoda_nonull.mat'
        elif name == 'usc-had':
            full_path = path + 'usc-had/usc-had.mat'
        elif name == 'opp_thomas':
            full_path = path + 'opportunity/opportunity_hammerla.mat'
        else:
            full_path = ''
        data_raw = load_dataset(full_path)
        # Obtaining the segmented data
        X_train, y_train = opp_sliding_window(data_raw['train' + '_data'], data_raw['train' + '_labels'],
                                                    window_size, stride_size)
        X_eval, y_eval = opp_sliding_window(data_raw['val' + '_data'], data_raw['val' + '_labels'],
                                          window_size, stride_size)
        X_test, y_test = opp_sliding_window(data_raw['test' + '_data'], data_raw['test' + '_labels'],
                                          window_size, stride_size)
        # Stacking train and eval sets
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_train = np.concatenate((y_train, y_eval), axis=0)

    return X_train, y_train, X_test, y_test
    # return torch.from_numpy(X_train), torch.from_numpy(y_train).long(), torch.from_numpy(X_test), torch.from_numpy(y_test).long()

# specify configurations of available data-sets.
# DATASET_CONFIGS = {
#     'mnist': {'features': 32, 'seq': 1, 'classes': 10},
#     'mnist28': {'features': 28, 'seq': 1, 'classes': 10},
#     'opportunity': {'features': 113, 'seq': 24, 'classes': 17}, # w/o null class
#     'hhar-raw': {'features': 6, 'seq': 50, 'classes': 6}, # no null class
#     'hhar-noaug': {'features': 120, 'seq': 20, 'classes': 6}, # no null class
#     'hhar-aug': {'features': 120, 'seq': 20, 'classes': 6}, # no null class
#     'opp_thomas': {'features': 77, 'seq': 30, 'classes': 18}, # w/ null class
#     'pamap2': {'features': 52, 'seq': 33, 'classes': 12}, # w/o null class
#     'skoda': {'features': 60, 'seq': 33, 'classes': 10}, # w/ null class
#     'usc-had': {'features': 6, 'seq': 33, 'classes': 12},
#     'ninapro-db2-c10': {'features': 12, 'seq': 40, 'classes': 10}, # w/o null class
#     'ninapro-db3-c10': {'features': 12, 'seq': 40, 'classes': 10}, # w/o null class
#     'emotion-all-frames10': {'features': 24, 'seq': 50, 'classes': 14},
#     'emotion-all-frames20': {'features': 24, 'seq': 25, 'classes': 14},
#     'emotion-all-frames25': {'features': 24, 'seq': 20, 'classes': 14},
#     'emotion-all-frames50': {'features': 24, 'seq': 10, 'classes': 14}
# }

def load_dataset_ninapro(full_path, filename, subject_idx=None, exp_setup=None):
    # target_gest_dict = {
    # 13: 0, 14:1, 10:2, 9:3, 5:4, 30:5, 34:6, 37:7, 32:8, 21:9
    # }
    target_gest_dict = {
    13: 0, 14:1, 12:2, 11:3, 5:4, 33:5, 34:6, 19:7, 32:8, 21:9
    }
    target_user_set_db2 = {11, 31, 17, 39, 7, 26, 35, 28, 4, 14}
    # target_user_set_db2 = {11, 31, 17, 39, 7, 26, 35, 28, 4, 14,
    #                         32,19,38,24,12,18,23,21,30,3}
    # templist = list(range(1,41))
    # target_user_set_db2 = set(templist)
    target_user_set_db3 = {1, 2, 3, 4, 5, 6, 9, 10, 11}
    X = []
    y = []
    rep = []
    if exp_setup==None:
        if filename[:11] == 'ninapro-db3':
            n_subject = 11
        else:
            n_subject = 40
        if filename[-3:] == 'c10':
            for subject in range(n_subject):
                data = loadmat(full_path+'S'+str(subject+1)+'_c10_200Hz.mat')
                X.extend(data['emg'])
                y.extend(data['restimulus'][0])
                rep.extend(data['rerepetition'][0])
        else:
            for subject in range(n_subject):
                data = loadmat(full_path+'S'+str(subject+1)+'_all_200Hz.mat')
                X.extend(data['emg'])
                y.extend(data['restimulus'])
                rep.extend(data['rerepetition'])

    elif exp_setup=='per-subject': 
        if 'c50' in filename:
            data = loadmat(full_path+'S'+str(subject_idx)+'_2000Hz_3.mat')
            X.extend(data['emg'])
            y.extend(data['restimulus'].T[0])
            rep.extend(data['rerepetition'].T[0])
        elif filename[-3:] == 'c10':
            data = loadmat(full_path+'S'+str(subject_idx)+'_c10_200Hz.mat')
            X.extend(data['emg'])
            y.extend(data['restimulus'][0])
            rep.extend(data['rerepetition'][0])
        else:
            data = loadmat(full_path+'S'+str(subject_idx)+'_all_200Hz.mat')
            X.extend(data['emg'])
            y.extend(data['restimulus'])
            rep.extend(data['rerepetition'])

    elif exp_setup=='leave-one-user-out':
        if filename[:11] == 'ninapro-db3':
            n_subject = 11
            target_user_set = target_user_set_db3
        else:
            n_subject = 40
            target_user_set = target_user_set_db2
        X_eval = []
        X_test = []
        y_eval = []
        y_test = []
        if filename[-3:] == 'c10':
            for subject in range(1, n_subject+1):
                if subject in target_user_set:
                    data = loadmat(full_path+'S'+str(subject)+'_c10_200Hz.mat')
                    if subject == subject_idx: # for test set
                        X_test.extend(data['emg'])
                        y_test.extend(data['restimulus'][0])
                    elif subject == 11: # for eval set
                        X_eval.extend(data['emg'])
                        y_eval.extend(data['restimulus'][0])
                    else:
                        X.extend(data['emg'])
                        y.extend(data['restimulus'][0])
        else:
            for subject in range(1, n_subject+1):
                if subject in target_user_set:
                    data = loadmat(full_path+'S'+str(subject)+'_all_200Hz.mat')
                    if subject == subject_idx: # for test set
                        X_test.extend(data['emg'])
                        y_test.extend(data['restimulus'][0])
                    elif subject == 11: # for eval set
                        X_eval.extend(data['emg'])
                        y_eval.extend(data['restimulus'][0])
                    else:
                        X.extend(data['emg'])
                        y.extend(data['restimulus'][0])
    
    if exp_setup=='leave-one-user-out':
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        for i in range(len(y)):
            y[i] = target_gest_dict[y[i]]
        for i in range(len(y_test)):
            y_test[i] = target_gest_dict[y_test[i]]
        for i in range(len(y_eval)):
            y_eval[i] = target_gest_dict[y_eval[i]]
        data_raw = {'train_data': X, 'train_labels': y,
                'val_data': X_eval, 'val_labels': y_eval,
                'test_data': X_test, 'test_labels': y_test
               }
    else:
        X = np.array(X)
        y = np.array(y)
        train_idx = []
        val_idx = []
        test_idx = []
        for i in range(len(rep)):
            if y[i] in target_gest_dict:
                if exp_setup == 'per-subject': # one model per user
                    if rep[i] == 5:
                        test_idx.append(i)
                    elif rep[i] == 3:
                        val_idx.append(i)
                    else:
                        train_idx.append(i)
                else: # one model for all users
                    if rep[i] == 2 or rep[i] == 5:
                        test_idx.append(i)
                    elif rep[i] == 3:
                        val_idx.append(i)
                    else:
                        train_idx.append(i)
                y[i] = target_gest_dict[y[i]]

        data_raw = {'train_data': X[train_idx, :], 'train_labels': y[train_idx],
                'val_data': X[val_idx, :], 'val_labels': y[val_idx],
                'test_data': X[test_idx, :], 'test_labels': y[test_idx]
               }

    for dset in ['train', 'val', 'test']:
        print('The shape of the {} dataset is {} x {}, and the labels is {}'.format(dset, len(data_raw[dset + '_data']), 
                                                    len(data_raw[dset + '_data'][0]), len(data_raw[dset + '_labels']) ))
        data_raw[dset + '_data'] = data_raw[dset + '_data'].astype(np.float32)
        data_raw[dset + '_labels'] = data_raw[dset + '_labels'].astype(np.uint8)
    
    return data_raw


def load_dataset(filename):
    """
    Loading the .mat file and creating a dictionary based on the phase
    :param filename: name of the .mat file
    :return: dictionary containing the sensory data
    """
    # Load the data from the .mat file
    data = loadmat(filename)

    # Putting together the data into a dictionary for easy retrieval
    data_raw = {'train_data': data['X_train'], 'train_labels': np.transpose(data['y_train']),
                'val_data': data['X_valid'],
                'val_labels': np.transpose(data['y_valid']), 'test_data': data['X_test'],
                'test_labels': np.transpose(data['y_test'])}

    # Setting the variable types for the data and labels
    for dset in ['train', 'val', 'test']:
        print('The shape of the {} dataset is {}, and the labels is {}'.format(dset, len(data_raw[dset + '_data'][0]),
                                                                               len(data_raw[dset + '_labels']) ))
        data_raw[dset + '_data'] = data_raw[dset + '_data'].astype(np.float32)
        data_raw[dset + '_labels'] = data_raw[dset + '_labels'].astype(np.uint8)

    return data_raw

def split_train_val_multi_tasks(X, y, input_class_l, task_idx):
    
    if task_idx == -1: # total test data case
        input_data = X
        target_data = y
    elif task_idx >= 0: # other general case
        print('input_class_l: ', input_class_l)
        input_class_set = set(input_class_l)
        input_data = X[[True if x in input_class_set else False for x in y]]
        target_data = y[[True if x in input_class_set else False for x in y]]
    else:
        print('wrong task idx input')

    data_size = input_data.shape[0]
    print("total data size of task idx {} is ".format(task_idx) + str(data_size))
    return torch.from_numpy(input_data), torch.from_numpy(target_data).long()

class MyDataLoaderMultiTasks(Dataset):
    """docstring for DataLoader"""
    def __init__(self, X, y, input_class_l, task_idx, target_transform = None, args=None):
        super(MyDataLoaderMultiTasks, self).__init__()
        self.input, self.target = split_train_val_multi_tasks(X, y, input_class_l, task_idx)
        self.target_transform = target_transform
        # if args.cuda_num >= 0:
        #     self.input = self.input.to("cuda:" + str(args.cuda_num))
        #     self.target = self.target.to("cuda:" + str(args.cuda_num))
        #     torch.device("cuda:" + str(args.cuda_num) if args.cuda_num >= 0 else "cpu")

    def __getitem__(self, index):
        if self.target_transform:
            # print('index: ', index)
            # print('target[index]: ', self.target[index])
            # print('input[index,:]: ', self.input[index,:])
            return (self.input[index, :], self.target_transform(self.target[index]))
        else:
            return (self.input[index, :], self.target[index])

    def __len__(self):
        return self.input.shape[0]

def slit_train_val2(X, y, class_set_D1, class_set_D2, name):
    if name == 'train_D1':
        input_data = X[[True if x in class_set_D1 else False for x in y]]
        target_data = y[[True if x in class_set_D1 else False for x in y]]
    elif name == 'train_D2':
        input_data = X[[True if x in class_set_D2 else False for x in y]]
        target_data = y[[True if x in class_set_D2 else False for x in y]]
    elif name == 'test_D1':
        input_data = X[[True if x in class_set_D1 else False for x in y]]
        target_data = y[[True if x in class_set_D1 else False for x in y]]
    elif name == 'test_D2':
        input_data = X[[True if x in class_set_D2 else False for x in y]]
        target_data = y[[True if x in class_set_D2 else False for x in y]]
    elif name == 'test_total':
        input_data = X
        target_data = y
    else:
        print('wrong name')
    data_size = input_data.shape[0]
    print("total data size of {} is ".format(name) + str(data_size))
    # return input_data, target_data
    return torch.from_numpy(input_data), torch.from_numpy(target_data).long()

class MyDataLoader2(Dataset):
    """docstring for DataLoader"""
    def __init__(self, X, y, class_set_D1, class_set_D2, name, target_transform = None):
        super(MyDataLoader2, self).__init__()
        self.input, self.target = slit_train_val2(X, y, class_set_D1, class_set_D2, name)
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.target_transform:
            return (self.input[index, :], self.target_transform(self.target[index]))
        else:
            return (self.input[index, :], self.target[index])

    def __len__(self):
        return self.input.shape[0]

class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None, quantize=32, args=None):
        super().__init__()
        self.quantize = quantize
        if quantize == 32: # incoming exemplar_sets are numpy array
            self.exemplar_sets = exemplar_sets
        elif quantize == 16: # incoming exemplar_sets are numpy array
            self.exemplar_sets = []
            for arr in exemplar_sets:
                self.exemplar_sets.append(arr.astype(np.float32))
        elif quantize < 16: # incoming exemplar_sets are torch.Tensor in qint8
            self.exemplar_sets = []
            for arr in exemplar_sets:
                self.exemplar_sets.append(args.quantizer.dequantize_tensor(QTensor(tensor=arr, scale=args.quantizer.scale, zero_point=args.quantizer.zero_point)))

        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        try:
            if self.quantize >=16:
                image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
            else:
                image = self.exemplar_sets[class_id][exemplar_id]
            return (image, class_id_to_return)
        except:
            # print('index: ', index)
            # print('class_id: ', class_id)
            # print('total: ', total)
            # print('exemplars_in_this_class: ', exemplars_in_this_class)
            if self.quantize >= 16:
                image = torch.from_numpy(self.exemplar_sets[class_id][0])
            else:
                image = self.exemplar_sets[class_id][0]
            return (image, class_id)


#----------------------------------------------------------------------------------------------------------#



# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'features': 32, 'seq': 1, 'classes': 10},
    'mnist28': {'features': 28, 'seq': 1, 'classes': 10},
    'opportunity': {'features': 113, 'seq': 24, 'classes': 17}, # w/o null class
    'hhar-raw': {'features': 6, 'seq': 50, 'classes': 6}, # no null class
    'hhar-noaug': {'features': 120, 'seq': 20, 'classes': 6}, # no null class
    'hhar-aug': {'features': 120, 'seq': 20, 'classes': 6}, # no null class
    'opp_thomas': {'features': 77, 'seq': 30, 'classes': 18}, # w/ null class
    'pamap2': {'features': 52, 'seq': 33, 'classes': 12}, # w/o null class
    'skoda': {'features': 60, 'seq': 33, 'classes': 10}, # w/ null class
    'usc-had': {'features': 6, 'seq': 33, 'classes': 12},
    'ninapro-db2-c10': {'features': 12, 'seq': 40, 'classes': 10}, # w/o null class
    'ninapro-db2-c50': {'features': 372, 'seq': 1, 'classes': 10}, # w/o null class
    'ninapro-db3-c10': {'features': 12, 'seq': 40, 'classes': 10}, # w/o null class
    'emotion-fbanks-frames10': {'features': 24, 'seq': 50, 'classes': 14},
    'emotion-fbanks-frames20': {'features': 24, 'seq': 25, 'classes': 14},
    'emotion-fbanks-frames25': {'features': 24, 'seq': 20, 'classes': 14},
    'emotion-fbanks-frames50': {'features': 24, 'seq': 10, 'classes': 14},
    'emotion-all-frames10': {'features': 24, 'seq': 50, 'classes': 5},
    'emotion-all-frames20': {'features': 24, 'seq': 25, 'classes': 5},
    'emotion-all-frames25': {'features': 24, 'seq': 20, 'classes': 5},
    'emotion-all-frames50': {'features': 24, 'seq': 10, 'classes': 5}
}

#----------------------------------------------------------------------------------------------------------#

def get_singletask_experiment(name, scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                             exception=False, dataset='opportunity', class_D2_l=None, subject_idx=None, exp_setup=None,
                             args=None):
    if name == 'sensor':
        # configurations
        config = DATASET_CONFIGS[dataset]
        if not only_config:
            X_train, y_train, X_test, y_test = load_and_segment_data(name=dataset, config=config, subject_idx=subject_idx, exp_setup=exp_setup)
            
            for i, v in Counter(y_train).items():
                print("class: {} , counts: {}".format(i,v))
            print("Total sum of train labels : ", len(y_train))
            weights_per_class = compute_class_weight('balanced',np.unique(y_train),y_train)
            for i, v in Counter(y_test).items():
                print("class: {} , counts: {}".format(i,v))
            print("Total sum of test labels : ", len(y_test))
            class_set_D2 = {}
            class_set_D1 = makeClassSetD1(class_set_D2, num_classes = config['classes'])
            num_classes_per_task_l = [len(class_set_D1), len(class_set_D2)]
            classes_per_task = config['classes']

            if 'cnn' in args.cls_type:
                X_train = reshape(X_train)
                X_test = reshape(X_test)
            elif 'mlp' in args.cls_type:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            test_total_dataset = MyDataLoader2(X_test, y_test, class_set_D1, class_set_D2, 'test_total', None)

            for i in range(1):
                train_datasets.append(
                    MyDataLoader2(X_train, y_train, class_set_D1, class_set_D2, 'train_D1' if i == 0 else 'train_D2',None))
                test_datasets.append(
                    MyDataLoader2(X_test, y_test, class_set_D1, class_set_D2, 'test_D1' if i == 0 else 'test_D2',None))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    return config if only_config else ((train_datasets, test_datasets, test_total_dataset), config, classes_per_task, num_classes_per_task_l, weights_per_class)




def get_multitask_experiment(name, scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                             exception=False, dataset='opportunity', class_D2_l=[0], subject_idx=None, exp_setup=None,
                             args=None):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['features']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['features']**2) for _ in range(tasks)]
            # prepare datasets
            train_datasets = []
            test_datasets = []
            for task_id, p in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(get_dataset('mnist', type="train", permutation=p, dir=data_dir,
                                                  target_transform=target_transform, verbose=verbose))
                test_datasets.append(get_dataset('mnist', type="test", permutation=p, dir=data_dir,
                                                 target_transform=target_transform, verbose=verbose))
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'sensor':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS[dataset]
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:

            X_train, y_train, X_test, y_test = load_and_segment_data(name=dataset, config=config, subject_idx=subject_idx, exp_setup=exp_setup)
            class_set_D2 = set(class_D2_l)
            class_set_D1 = makeClassSetD1(class_set_D2, num_classes = config['classes'])
            if len(class_set_D2) > 2:
                print("==== Adding Half More Classes ====")
            else:
                print("==== Adding One More Classes ====")
            print("class dict D1: ", class_set_D1)
            print("class dict D2: ", class_set_D2)
            print("Count of train labels : ", Counter(y_train))
            
            config['train_size'] = len(y_train)
            for i, v in Counter(y_train).items():
                print("class: {} , counts: {}".format(i,v))
            print("Total sum of train labels : ", len(y_train))

            weights_per_class = compute_class_weight('balanced',np.unique(y_train),y_train)
            for i, v in Counter(y_test).items():
                print("class: {} , counts: {}".format(i,v))
            print("Total sum of test labels : ", len(y_test))
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = makePermu([class_set_D1, class_set_D2])
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            weights_per_class = applyPermu(weights_per_class, permutation)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            num_classes_per_task_l = [len(class_set_D1), len(class_set_D2)]

            if 'cnn' in args.cls_type:
                X_train = reshape(X_train)
                X_test = reshape(X_test)
            elif 'mlp' in args.cls_type:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            test_total_dataset = MyDataLoader2(X_test, y_test, class_set_D1, class_set_D2, 'test_total', target_transform)

            for i, labels in enumerate(labels_per_task):
                # target_transform = transforms.Lambda(
                #     lambda y, x=labels[0]: y - x
                # ) if scenario == 'domain' else None
                train_datasets.append(
                    MyDataLoader2(X_train, y_train, class_set_D1, class_set_D2, 'train_D1' if i == 0 else 'train_D2',target_transform))
                test_datasets.append(
                    MyDataLoader2(X_test, y_test, class_set_D1, class_set_D2, 'test_D1' if i == 0 else 'test_D2',target_transform))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    # config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets, test_total_dataset), config, classes_per_task, num_classes_per_task_l, weights_per_class)


def get_multitask_experiment_multi_tasks(name, scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                                        exception=False, dataset='opportunity', input_class_seq=None, subject_idx=None, exp_setup=None,
                                         args=None):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    if name == 'sensor':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS[dataset]
        print(config)
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            X_train, y_train, X_test, y_test = load_and_segment_data(name=dataset, config=config, subject_idx=subject_idx, exp_setup=exp_setup)

            config['train_size'] = len(y_train)
            for i, v in Counter(y_train).items():
                print("class: {} , counts: {}".format(i,v))
            print("Total sum of train labels : ", len(y_train))
            weights_per_class = compute_class_weight('balanced',np.unique(y_train),y_train)
            for i, v in Counter(y_test).items():
                print("class: {} , counts: {}".format(i,v))
            print("Total sum of test labels : ", len(y_test))
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = makePermu(input_class_seq)
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            weights_per_class = applyPermu(weights_per_class, permutation)
            # generate number of classes per task
            num_classes_per_task_l = []
            for input_class_l in input_class_seq:
                num_classes_per_task_l.append(len(input_class_l))

            if 'cnn' in args.cls_type:
                X_train = reshape(X_train)
                X_test = reshape(X_test)
            elif 'mlp' in args.cls_type:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            test_total_dataset = MyDataLoaderMultiTasks(X_test, y_test, input_class_seq, -1, target_transform, args)
            for task_idx, input_class_l in enumerate(input_class_seq):
                train_datasets.append(
                    MyDataLoaderMultiTasks(X_train, y_train, input_class_l, task_idx, target_transform, args))
                test_datasets.append(
                    MyDataLoaderMultiTasks(X_test, y_test, input_class_l, task_idx, target_transform, args))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets, test_total_dataset), config, classes_per_task, num_classes_per_task_l, weights_per_class)