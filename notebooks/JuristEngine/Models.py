import torch
from torch import nn
import torch.nn.functional as F


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer




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

    def forward(self, x:tuple[torch.Tensor]):
        x_q, x_d = x
        output_q = self.model_quest(x_q) # (batch_size, quest_emb_dim)
        output_d = self.model_doc(x_d) # (batch_size, doc_emb_dim)
        
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
   


        
        