import numpy as np
import copy


def get_dataset_directories():
    dirs = {}
    dirs['trainA'] = './dataset/trainA'
    dirs['trainB'] = './dataset/trainB'


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