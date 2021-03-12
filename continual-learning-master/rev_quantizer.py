from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import namedtuple
import utils
import numpy as np


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

class Quantizer():
    def __init__(self):
        super().__init__()
        # self.QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])
        self.scale = 0
        self.zero_point = 0
        self.min_val = 0
        self.max_val = 0


    def calcScaleZeroPoint(self, min_val, max_val, num_bits=8):
        # Calc Scale and zero point of next
        qmin = 0.
        qmax = 2. ** num_bits - 1.

        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = int(zero_point)

        return scale, zero_point

    def quantize_tensor(self, x, num_bits=8, min_val=None, max_val=None):

        if not min_val and not max_val:
            min_val, max_val = x.min(), x.max()

        qmin = 0.
        qmax = 2. ** num_bits - 1.

        # scale, zero_point = self.calcScaleZeroPoint(min_val, max_val, num_bits)
        q_x = self.zero_point + x / self.scale
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()

        return q_x

    def dequantize_tensor(self, q_x):
        return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

    # Get Min and max of x tensor, and stores it
    def updateStats(self, x, stats, key):
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)

        if key not in stats:
            stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 64}
        else:
            stats[key]['max'] += max_val.sum().item()
            stats[key]['min'] += min_val.sum().item()
            stats[key]['total'] += 64

        return stats

    # Reworked Forward Pass to access activation Stats through updateStats function
    def gatherActivationStats(self, x, stats):
        stats = self.updateStats(x.clone().view(x.shape[0], -1), stats, 'input')
        return stats

    # Entry function to get stats of all functions.
    def gatherStats(self, dataset, cuda_num=None, num_bits=8):

        stats = {}
        data_loader = iter(utils.get_data_loader(dataset, 64, cuda=True if cuda_num else False, drop_last=True))
        iters_left = len(data_loader)
        device = torch.device("cuda:" + str(cuda_num) if cuda_num >= 0 else "cpu")

        with torch.no_grad():
            for i in range(iters_left):
                data, target = next(data_loader)  # --> sample training data of current task
                # data, target = data.to(device), target.to(device)
                stats = self.gatherActivationStats(data, stats)

        final_stats = {}
        for key, value in stats.items():
            final_stats[key] = {"max": value["max"] / value["total"], "min": value["min"] / value["total"]}
            self.min_val = value["min"] / value["total"]
            self.max_val = value["max"] / value["total"]
            self.scale, self.zero_point = self.calcScaleZeroPoint(self.min_val, self.max_val, num_bits)
        return final_stats


# stats = gatherStats(q_model, test_loader, args.device)
# print(stats)
#
# x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
# x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
