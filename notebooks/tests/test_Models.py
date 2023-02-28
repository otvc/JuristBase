import sys
sys.path.insert(0, '..')

import pickle
import pandas as pd
import numpy as np
import unittest

from base import  load_yaml_config, mlflow_model_path, load_torch_model
from JuristEngine.Models import PipelineTFIDFLinearScore

from transformers import pipeline
from huggingface_hub import Repository
from transformers.pipelines import PIPELINE_REGISTRY

class test_Pipelines(unittest.TestCase):
    
    def test_TFIDFLinearScore(self):
        codes = pd.read_csv('../data//CodesTypes.csv', index_col='Id')
        codes_tree = pd.read_csv('../data/CodesTree.csv', index_col = 'Id').drop(columns=['Unnamed: 0'])
        codes_tree.rename(columns = {'CodesId': 'codes_id'}, inplace = True)
        codes.index = codes.index - 3 # нужно сделать, что совпадали идентификаторы кодексов в codes_tree
        codes = codes.loc[0:]
        
        with open('../data/df_art_to_par_vect.pkl', 'rb') as f:
            vectorized_art = pickle.load(f)
        input = {
            'input_text': [' Что делать, если обманули мошенники, позвонили на сотовый телефон якобы сотрудник банка, \
                           сказали, что на меня оформили кредит в банке, спросили, каким банком я пользуюсь, я сказала, что Сбербанком, меня \
                           переключили на сотрудника сбербанка, я спросили хочу ли я перекрыть кредит, объяснили, что сделать, я оформила кредит в \
                            онлайн банке в Сбербанке, мне одобрили 77000, и потом сказали перевести какому то человеку, якобы он перекроет мой кредит, и \
                             я перевела всю сумму какому-то человеку, и поняла я все это через 2 часа, что я взяла кредит. Что мне делать, как мне быть в \
                             такой ситуации? ']
        }
        config = load_yaml_config('./config/test_Models.yaml')
        mlflow_yaml_path = config['test_TFIDFLinearScore']['model']['meta_model_path']
        model = load_torch_model(mlflow_model_path(mlflow_yaml_path))
        pipe = PipelineTFIDFLinearScore(model, codes, codes_tree, vectorized_art, 'paragraph')

        output = pipe(input, doc_batch_size = 32)
        print(output[0].shape)


        
        
if __name__ == '__main__':
    unittest.main()
    
