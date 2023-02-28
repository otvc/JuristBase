import sys
sys.path.insert(0, '..')

import argparse
import pickle

import pandas as pd
import numpy as np

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from JuristEngine.Instruments import df_to_tfidf_vectors

from notebooks.base import load_yaml_config, mlflow_model_path, load_torch_model

def get_model_path():
    parser = argparse.ArgumentParser(prog = 'create vectorized df')
    parser.add_argument('-cp', '--config_path', default = f'./config/{sys.argv[0][:-2]}yaml')
    args = parser.parse_args()
    return args.config_path

if __name__ == '__main__':
    config_path = get_model_path()
    arguments = load_yaml_config(config_path)
    
    model_path = mlflow_model_path(arguments['model']['meta_model_path'])
    model = load_torch_model(model_path)
    
    df = pd.read_csv(arguments['data']['df_path'], index_col = arguments['data']['index_col'])
    
    df_extr = df_to_tfidf_vectors(df, arguments['data']['target_column'], model.model_doc.vectorizer)
    
    with open(arguments['data']['path_to_save'], 'wb') as f:
        pickle.dump(df_extr, f)
