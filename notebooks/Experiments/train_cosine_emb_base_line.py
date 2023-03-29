import sys
sys.path.insert(0, '..')


import os
import warnings
import datetime
from urllib.parse import urlparse

import pandas as pd

import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


from JuristEngine import Models 
from JuristEngine.Models import RetrieverDouble, TFIDFNetEmb, CosineScore, Twins
from JuristEngine import Instruments
from ExperimentsBase import load_ttv_datasets, savefig_train_val, parse_experiments_args, create_doc_features, create_nms_index, find_docs_for_queries, MP

import mlflow
import mlflow.pytorch
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def create_model(train_dataset, df_art_to_par, max_quest_features, max_doc_features):
    vectorizer_quest = TfidfVectorizer(max_features = max_quest_features)
    vectorizer_doc = TfidfVectorizer(max_features = max_doc_features)

    vectorizer_quest.fit(train_dataset.dataset.question.values)
    vectorizer_doc.fit(df_art_to_par.paragraph.values)
    quest_features = vectorizer_quest.transform(['0']).toarray().shape[-1]
    doc_features = vectorizer_doc.transform(['0']).toarray().shape[-1]
    assert doc_features == quest_features
    
    tfidf_retriever = TFIDFNetEmb(vectorizer_quest)
    doc_retriever = TFIDFNetEmb(vectorizer_doc)
    base_retriever = Twins(tfidf_retriever, doc_retriever)

    return base_retriever, tfidf_retriever, doc_retriever

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    start_time = str(datetime.datetime.now())
    print(start_time)
    graphics_dir = './data/plots/'
    model_dir = '/data/models/'
    path_train_val_plot = graphics_dir + 'losses_' + start_time + '.png'
    
    base_path, train_jurist, test_jurist, val_jurist = './data/datasets/', 'train_article_match.pt', 'test_article_match.pt', 'val_article_match.pt'
    path_df_art_to_par = base_path + 'df_art_to_par.csv'

    lr, epochs, batch_size, max_quest_features, max_doc_features =  parse_experiments_args()
    MAX_K = 5
    
    train_dataset, test_dataset, val_dataset = load_ttv_datasets(base_path + train_jurist, base_path + test_jurist, base_path + val_jurist)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)  
    test_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    df_art_to_par = pd.read_csv(path_df_art_to_par, index_col = 'article_id')

    model, qest_retriever, doc_retriever = create_model(train_dataset, df_art_to_par, max_quest_features, max_doc_features)

    with mlflow.start_run():
        criterion = nn.CosineEmbeddingLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        output = Instruments.train_tfidf_cosine_score(model, train_dataloader, val_dataloader, optimizer, criterion, 
                                                      epochs = epochs, plot_loss = False)

        features = create_doc_features(df_art_to_par, doc_retriever)
        index = create_nms_index(features)

        df_ground_truth = test_dataset.dataset[test_dataset.dataset.score > 0]
        df_ground_truth_articles = df_ground_truth[['question_id', 'article_id']].groupby('question_id').agg(list)

        duplicated = df_ground_truth[['question', 'question_id']].duplicated(keep='last')
        duplicated_indeces = duplicated.index[duplicated]
        test_questions = df_ground_truth.loc[duplicated_indeces]
        predictions = find_docs_for_queries(test_questions, qest_retriever, index, df_art_to_par)
        print(df_ground_truth_articles['article_id'].values[:3])
        MP_metric = MP(predictions, df_ground_truth_articles['article_id'].values, MAX_K=MAX_K)
        print(MP_metric)

        train_loss = output['train_loss']
        val_loss = output['val_loss']
        
        savefig_train_val(train_loss, val_loss, path_train_val_plot)
        mlflow.log_artifact(path_train_val_plot)

        mlflow.log_metric(f'MP{MAX_K}', MP_metric)
        mlflow.pytorch.log_model(model, 'tf_idf_cosine_score.pt', registered_model_name = 'tf_idf_cosine_score.pt')
