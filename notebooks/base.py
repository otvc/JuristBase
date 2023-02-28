import yaml
from yaml.loader import SafeLoader

import torch

def load_yaml_config(path):
    with open(path, 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=SafeLoader)
    return config

def mlflow_model_path(path_to_yaml, model_name = 'model.pth'):
    model_ver_config = load_yaml_config(path_to_yaml)
    src_model_fol = model_ver_config['source']+'/data/' + model_name
    if src_model_fol[:4] == 'file':
        src_model_fol = src_model_fol[7:]
    return src_model_fol

def load_torch_model(model_path):
    return torch.load(model_path)
