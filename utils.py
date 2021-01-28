import numpy as np
import copy
import os
import torch

def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')

def save_checkpoint(state, save_path):
    torch.save(state, save_path)

def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def get_train_directories(dataset_dir):
    dirs = {}
    dirs['trainA'] = './input/trainA'
    dirs['trainB'] = './input/trainB'
    return dirs

def get_test_directories(dataset_dir):
    dirs = {}
    dirs['testA'] = './input/testA'
    dirs['testB'] = './input/testB'
    return dirs

class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)


class PoolSample(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, items):
        return_items = []
        for item in items:
            if self.cur_elements < self.max_elements:
                self.items.append(item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = item
                    return_items.append(tmp)
                else:
                    return_items.append(item)

        return return_items