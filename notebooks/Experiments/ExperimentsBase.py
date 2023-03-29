import sys
from typing import Union, Any

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

import nmslib

from tqdm import tqdm

from JuristEngine.Models import RetrieverDouble, TFIDFNet, LinearScore, CosineScore

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

def parse_experiments_args():
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-4
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    max_quest_features = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    max_doc_features = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    return lr, epochs, batch_size, max_quest_features, max_doc_features

'''

`params`:
    `doc_vectors`:Union[torch.tensor, np.ndarray]: data with shapes (n, features);
    `method`:str: see in `nmslib.init` documentation;
    `space`:str: see in `nmslib.init` documentation.

return:
    created nmslib index.
'''
def create_nms_index(doc_vectors:Union[torch.tensor, np.ndarray], method:str = 'hnsw', space:str = 'cosinesimil'):
    index = nmslib.init(method = method, space = space)
    index.addDataPointBatch(doc_vectors)
    index.createIndex()
    return index

'''
`params`:
    `data`:pd.DataFrame: dataframe, which contain text of question;
    `model`:torch.nn.Module: model for extraction question features;
    `index`:: index from nmslib for searching;
    `data_doc:pd.DataFrame: dataframe, which contain text of documents;
    `target`:Any: name of column from data;
    `K`:int: count of searching documents.
return
    1d tensor with shapes which equal data.shape[0]
'''
def find_docs_for_queries(data:pd.DataFrame, model:torch.nn.Module, index, data_doc, target:Any = 'question', target_doc = 'article_id', K = 3):
    indeces = []
    with torch.no_grad():
        for i in tqdm(data.index):
            part_ids, _ = index.knnQuery(model(data.loc[i:i+1][target]), k = K)
            indeces.append(data_doc.iloc[part_ids].index)
    return torch.tensor(indeces)

'''
`params`:
    `data`:pd.DataFrame: dataframe which contain needed feature with text;
    `model`:torch.nn.Module: model which obtain batch with `STR` and output torch features;
    `target`:Any: value of column with text information from `data` argument;
    `batch_size`:int: size of batch for feature extraction.

return:
    torch.Tensor with shape (data.shape[0], N1, N2, ..., Nn), where other demension dependend 
    from size outputed model features.
'''
def create_doc_features(data:pd.DataFrame, model:torch.nn.Module, target:Any = 'paragraph', batch_size:int = 256):
    were_training = model.training
    if model.training:
        model.eval()

    total_batches = data.shape[0]//batch_size
    embedded_vecs = []
    with torch.no_grad():
        for i in tqdm(range(total_batches)):
            batched_vecs = model(data.iloc[i*batch_size:(i+1)*batch_size][target])
            embedded_vecs.extend(batched_vecs)

    if were_training:
        model.train()

    return embedded_vecs

'''
Calculating Precision@K

`params`:
    `predictions`:Union[torch.tensor, np.ndarray]: values, which predicted your algs;
    `ground_truth`:Union[torch.tensor, np.ndarray]: ground truth values from your dataset;
    `K`:int: hyperparameter of Precision@K metrics, which define how much of first K data 
             should be get for evaluating (K value should be less then size of `predictions`);
    `check_limit`:bool: using for limiting K value if `ground_truth` or `predictions` less then
                        `K` will equal min size.

return: 
    metric value
'''
def PrecisionK(predictions:Union[torch.tensor, np.ndarray], ground_truth:Union[torch.tensor, np.ndarray], K:int = 5, check_limit:bool = True):
    ground_truth = torch.tensor(ground_truth)
    if check_limit:
        K = min(min(ground_truth.shape[0], predictions.shape[0]), K)
    ground_truth = pd.DataFrame(ground_truth[:K])
    predictions = pd.DataFrame(predictions[:K])
    predictions.index = predictions[0].values
    ground_truth.index = ground_truth[0].values
    gp_join = predictions.join(ground_truth, how = 'inner', lsuffix='-')
    return gp_join.shape[0]/ground_truth.shape[0]


'''
Calculating Mean Precision@K
Because for my model ranking is not important and important that values in topk, 
that MAP it's not valueble, but Mean Precision is corresponding for this task

`params`:
    `queries_predictions`:Union[torch.tensor, np.ndarray]: predictions for all queries;
    `queries_ground_truth`:Union[torch.tensor, np.ndarray]: ground_truth predictions for all queries
    `MAX_K`:int: maximum k for precision
return:
    float mean value
'''
def MP(queries_predictions:Union[torch.tensor, np.ndarray], queries_ground_truth:Union[torch.tensor, np.ndarray], MAX_K:int = 5):
    return torch.tensor([PrecisionK(predictions, ground_truth, K = cur_k)
                         for cur_k in range(1, MAX_K + 1)
                         for predictions, ground_truth
                         in zip(queries_predictions, queries_ground_truth)]).mean(-1).mean(-1)

