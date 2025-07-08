'''
    Datasplit for training
'''
import numpy as np
import os

def get_data(train_list, train_path):
    # Load data & labels
    data_list = []
    data_label = []
    lines = open(train_list).read().splitlines()
    dictkeys = list(set([x.split()[0] for x in lines]))
    dictkeys.sort()
    dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
    for index, line in enumerate(lines):
        speaker_label = dictkeys[line.split()[0]]
        file_name = os.path.join(train_path, line.split()[1])
        data_label.append(speaker_label)
        data_list.append(file_name)
    return data_list, data_label

def split_ssl_data(data, target, lb_samples_per_class, num_classes, include_lb_to_ulb=False):
    """
    data & target is splitted into labeled and unlabeld data.

    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx = sample_labeled_data(data, target, lb_samples_per_class, num_classes)

    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]
def sample_labeled_data(data, target,
                        lb_samples_per_class, num_classes):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''

    lb_data = []
    lbs = []
    lb_idx = []

    for c in range(num_classes):

        idx = np.where(target == c)[0]

        # idx = np.random.choice(idx, lb_samples_per_class, False)
        idx = np.random.choice(idx, int(lb_samples_per_class * len(idx)), False)

        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])
    return np.array(lb_data), np.array(lbs), np.array(lb_idx)
