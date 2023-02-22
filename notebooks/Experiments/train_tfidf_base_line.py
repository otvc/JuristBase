import sys
sys.path.insert(0, '..')


import os
import warnings
from urllib.parse import urlparse

import pandas as pd

import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


from JuristEngine import Models 
from JuristEngine.Models import RetrieverDouble, TFIDFNet, LinearScore
from JuristEngine import Instruments
from ExperimentsBase import load_ttv_datasets, savefig_train_val

import mlflow
import mlflow.pytorch
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


#To config
def parse_experiments_args():
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    max_quest_features = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    max_doc_features = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    return lr, epochs, batch_size, max_quest_features, max_doc_features

def create_model(train_dataset, df_art_to_par, max_quest_features, max_doc_features):
    vectorizer_quest = TfidfVectorizer(max_features = max_quest_features)
    vectorizer_doc = TfidfVectorizer(max_features = max_doc_features)

    vectorizer_quest.fit(train_dataset.dataset.question.values)
    vectorizer_doc.fit(df_art_to_par.paragraph.values)
    quest_features = vectorizer_quest.transform(['0']).toarray().shape[-1]
    doc_features = vectorizer_doc.transform(['0']).toarray().shape[-1]
    
    tfidf_retriever = TFIDFNet(vectorizer_quest)
    doc_retriever = TFIDFNet(vectorizer_doc)
    linear_score = LinearScore(quest_features, doc_features, n_scores=2)
    base_retriever = RetrieverDouble(linear_score, tfidf_retriever, doc_retriever)
    return base_retriever
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    #To config
    start_time = ''
    graphics_dir = './data/plots/'
    model_dir = '/data/models/'
    path_train_val_plot = graphics_dir + 'losses_' + start_time + '.png'
    print(path_train_val_plot)
    base_path, train_jurist, test_jurist, val_jurist = './data/datasets/', 'train_article_match.pt', 'test_article_match.pt', 'val_article_match.pt'
    path_df_art_to_par = base_path+'df_art_to_par.csv'
    
    
    lr, epochs, batch_size, max_quest_features, max_doc_features =  parse_experiments_args()
    
    
    train_dataset, test_dataset, val_dataset = load_ttv_datasets(base_path + train_jurist, base_path + test_jurist, base_path + val_jurist)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)  
    test_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    df_art_to_par = pd.read_csv(path_df_art_to_par, index_col = 'article_id')
                                  
    model = create_model(train_dataset, df_art_to_par, max_quest_features, max_doc_features)
    
    with mlflow.start_run():
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        output = Instruments.train_tfidf_linear_score(model, train_dataloader, val_dataloader,optimizer, criterion, 
                                                      epochs = epochs, plot_loss = False)

        train_loss = output['train_loss']
        val_loss = output['val_loss']
        savefig_train_val(train_loss, val_loss, path_train_val_plot)
        mlflow.log_artifact(path_train_val_plot)


        with torch.no_grad():
            y_preds, y_gt = Instruments.test_loop(model, test_dataloader, Instruments.unpack_retriever_tfidf)
        y_probas, y_gt = torch.concat(y_preds), torch.concat(y_gt)
        test_roc_auc = roc_auc_score(y_gt.detach().numpy(), y_probas.detach().numpy().argmax(1))
        
        mlflow.log_metric('roc_auc', test_roc_auc)
        
        mlflow.pytorch.log_model(model, 'tf_idf_linear_score.pt', registered_model_name = 'tf_idf_linear_score.pt')
    
    