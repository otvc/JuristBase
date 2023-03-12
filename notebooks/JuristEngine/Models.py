from typing import Union, Optional

import math

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import Pipeline, PreTrainedTokenizer, TFPreTrainedModel, PreTrainedModel
from transformers.pipelines import ArgumentHandler
from transformers.modelcard import ModelCard

class Retriever(nn.Module):
    
    '''
    
    `params`:
        `score_model`:nn.Module: net, which obtain tuple of 
        question and document embbeddings and return score for corresponding df.
        Where `questions` will have sizes (batch_size, emb_quest) or (batch_size, len(seq), emb_quest)
        Where `documents` will have sizes (batch_size, emb_doc) or (batch_size, len(seq), emb_doc)
    '''
    def __init__(self, score_model):
        super().__init__()
        self.score_model = score_model
        
    '''
    
    `params`:
        `x`:tuple[torch.Tensor]: tuple of text our questions and documents, where
         1. torch.Tensor with shapes (batch_size, len(question)) or (batch_size, len(question), emb_dim_1) which represent questions
         2. torch.Tensor with shapes (batch_size, len(question)) or (batch_size, len(document), emb_dim_2) which represent documents
         Where batch_id question correspond with batch_id document for which we want to obtain score
         Shapes depends from your retriever architecture
    '''
    def forward(self, x:tuple[torch.Tensor]):
        pass
        

class RetrieverDouble(Retriever):
    
    def __init__(self, score_model, model_quest, model_doc):
        super().__init__(score_model)
        self.model_quest = model_quest
        self.model_doc = model_doc
    
    '''
    On inference you can't use convert document, and use predetermined embeddings.
    If you will wan't use it, that change `inference` on `True` and send document in touple like
    torch.Tensors
    '''
    def forward(self, x:tuple[torch.Tensor], inference = False):
        x_q, x_d = x
        output_q = self.model_quest(x_q) # (batch_size, quest_emb_dim)
        output_d = self.model_doc(x_d) if not inference else x_d # (batch_size, doc_emb_dim)
        
        scores = self.score_model((output_q, output_d)) # (batch_size)
        
        return scores
    
class TFIDFNet(nn.Module):
    
    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer = vectorizer
        
    def forward(self, x):
        output = self.vectorizer.transform(x).toarray() #(batch_size, emb_dim)
        output = torch.Tensor(output)
        return output
    
class TFIDFNetEmb(nn.Module):
    
    def __init__(self, vectorizer, out_features = 128):
        super(TFIDFNetEmb, self).__init__()
        self.vectorizer = vectorizer
        
        in_features = vectorizer.transform(['0']).toarray().shape[-1]
        self.fc1 = nn.Linear(in_features, out_features=out_features)
        self.act1 = nn.Tanh()
        self.block = nn.Sequential(self.fc1, self.act1)
        
    def forward(self, x):
        tfidf_features = self.vectorizer.transform(x).toarray() #(batch_size, emb_dim)
        tfidf_features = torch.Tensor(tfidf_features)
        output = self.block(tfidf_features)
        
        return output
    
class Twins(nn.Module):
    
    def __init__(self, model_quest, model_doc):
        super(Twins, self).__init__()
        self.model_quest = model_quest
        self.model_doc = model_doc
    
    '''
    On inference you can't use convert document, and use predetermined embeddings.
    If you will wan't use it, that change `inference` on `True` and send document in touple like
    torch.Tensors
    '''
    def forward(self, x:tuple[torch.Tensor], inference = False):
        x_q, x_d = x
        output_q = self.model_quest(x_q) # (batch_size, quest_emb_dim)
        output_d = self.model_doc(x_d) if not inference else x_d # (batch_size, doc_emb_dim)
        return (output_q, output_d)
    
class LinearScore(nn.Module):
    def __init__(self, q_dim, d_dim, n_scores = 1, sigm_active = False):
        super().__init__()
        self.fc = nn.Linear(q_dim+d_dim, n_scores)
        self.sigm_active = sigm_active
    
    def forward(self, x):
        x = torch.cat(x, dim = 1)
        output = self.fc(x)
        if self.sigm_active:
            output = F.sigmoid(output)
        return output
    
class CosineScore(nn.Module):

    def __init__(self, dim = 1) -> None:
        super(CosineScore,  self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim)
    
    '''
    
    `params`:
        `x`:tuple : tuple of 2 tensors with question and documents, which should have similar shapes.
                    Embeddings should be contain on self.dim on both tensors.
    '''
    def forward(self, x:tuple):
        q_emb, d_emb = x
        output = self.cos_sim(q_emb, d_emb)
        return output

class PipelineTFIDFLinearScore:
    
    def __init__(self, model, 
                 codes, 
                 codes_tree, 
                 vectorized_art:pd.DataFrame, 
                 vector_column:str) -> None:
        self.model = model
        self.codes = codes
        self.codes_tree = codes_tree
        self.vectorized_art = vectorized_art
        self.vector_column = vector_column

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        preprocess_kwargs['doc_batch_size'] = kwargs['doc_batch_size'] if 'doc_batch_size' in kwargs else 32
        self.top_k  = kwargs['top_k'] if 'top_k' in kwargs else 4

        
        return preprocess_kwargs, {}, {}
    
    def create_query_articles_generator(self, text, batch_size):
        total_articles = self.vectorized_art.shape[0] # total articles count, for which we should estimate match
        total_batch_count = math.ceil(total_articles/batch_size) # total count of batch for particular bs
        def gen_query_article():
            for i in range(total_batch_count):
                vectors = self.vectorized_art.iloc[i*batch_size:(i+1)*batch_size][self.vector_column].values
                vectors = torch.Tensor(list(map(lambda x: x.toarray(), vectors))).reshape((vectors.shape[0], vectors[0].shape[-1]))
                yield ([text]*vectors.size(0), vectors)
                
        return gen_query_article
        
    def preprocess(self, inputs):
        model_input = inputs["input_text"] # (count_queries)
        batch_size = inputs['doc_batch_size'] #batch for use on inference with model
        text_generators = []
        for text in model_input:
            text_gen = self.create_query_articles_generator(text, batch_size)
            text_generators.append(text_gen)
        
        return {"model_input_gen": text_generators}
    
    def _forward(self, model_input):
        batch_gen = model_input["model_input_gen"]
        output = []
        for batch_gen_text in batch_gen:
            particular_output = []
            for batch in batch_gen_text():
                batch_out = self.model(batch, inference = True)
                particular_output.append(batch_out)
            output.append(torch.concat(particular_output))
        return output
    
    def get_full_article_name_by_iloc(self, iloc_indeces):
        article_indexes = self.vectorized_art.iloc[iloc_indeces].index
        if article_indexes.shape[0] == 0:
            return []
        needed_articles = self.codes_tree[self.codes_tree.index.isin(article_indexes)]
        needed_articles.rename(columns={'Name':'article_name'}, inplace = True)
        full_name_article = needed_articles.join(self.codes, on = 'codes_id', how = 'inner', rsuffix = 'r_')
        full_name_article['full_text'] = full_name_article['Name'] + '; ' + full_name_article['article_name']
        return full_name_article['full_text'].values
    
    def postprocess(self, model_outputs):
        output_full_names = [] # (count_queries, __) contain for query `iloc` ids of particular article  
        for part_output in model_outputs:
            part_output = F.softmax(part_output, dim = -1)
            df_probas = pd.DataFrame(part_output)
            iloc_art_ids = df_probas.sort_values(0, ascending=0).head(self.top_k+1).index
            full_names = self.get_full_article_name_by_iloc(iloc_art_ids)
            output_full_names.append(full_names)
        return output_full_names
 
    def __call__(self, input, **kwargs):
        sanitized_args, _, _ = self._sanitize_parameters(**kwargs)
        sanitized_args.update(input)
        preprocessed = self.preprocess(sanitized_args)
        output = self._forward(preprocessed)
        postprocessed_output = self.postprocess(output)
        return postprocessed_output

