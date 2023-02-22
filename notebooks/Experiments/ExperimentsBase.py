import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

'''
Function for loading train, test, validation datasets
`params`:
    `path_to_train`:Dataset: path to dataset for training
    `path_to_test`:Dataset: path to dataset for test
    `path_to_val`:Dataset: path to dataset for validation
return:
    tuple of 3 types of dataset
'''
def load_ttv_datasets(path_to_train = 'train_dataset.pt',
                      path_to_test = 'test_dataset.pt',
                      path_to_val = 'val_dataset.pt'):
    train_dataset = torch.load(path_to_train)
    test_dataset = torch.load(path_to_test)
    val_dataset = torch.load(path_to_val)
    return train_dataset, test_dataset, val_dataset


def savefig_train_val(train_loss, val_loss, path, title_suffix=''):
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    axes.set_title(' '.join(['Loss', title_suffix]))
    axes.plot(train_loss, label='train')
    axes.plot(val_loss, label='validation')
    axes.legend()

    plt.savefig(path)
