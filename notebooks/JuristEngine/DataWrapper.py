import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class JuristDataset(Dataset):
    
    '''
    
    `params`:
        `df_databridge`: contain codes_id, answer_id, article_id
        `df_questions`: 
        `df_codes`: contain Id (id of codes, which needed to match article in df_articles),Name (codes name)
        `df_articles`: contain Parent(id of parent in article trees), CodesID (id codes), Name (text of article name which contain number of article, which needed for finding by ArticleNum from `df_quit_quest`)
    '''
    def __init__(self, 
                 df_databridge, 
                 df_questions, 
                 df_codes, 
                 df_codes_tree,
                 df_answers,
                 df_paragraphs, 
                 k_articles = 2,
                 k_codes = 3,
                 replace = False):
        self.databridge = df_databridge
        self.questions = df_questions
        self.codes = df_codes
        self.codes_tree = df_codes_tree
        self.paragraphs = df_paragraphs
        self.answers = df_answers
        
        self.k_articles = k_articles
        self.k_codes = k_codes
        self.replace = replace
        
        self.loaded_articles = dict.fromkeys(self.paragraphs.article_id.unique())
        self.loaded_codes = dict.fromkeys(self.codes_tree.codes_id.unique())
        
        self.dataset = self.create_dataset()
        
        
    
    def codes_is_load(self, codes_id:int):
        return codes_id in self.loaded_codes
    
    def article_is_load(self, article_id: int):
        return article_id in self.loaded_articles
  
    '''
    
    `params`:
        `correct_articles_id:dict`: it's map codes to article of this codes
    `return`:
        dataframe with columns codes_id, article_id
    '''
    def sample_bad_answers(self, correct_articles_id:dict[list] = []):
        bad_answers = []
        bad_codes_ids = []
        codes = np.random.choice(list(self.loaded_codes.keys()), self.k_codes, replace = self.replace)
        for cur_codes in codes:
            particular_codes = self.codes_tree[self.codes_tree.codes_id == cur_codes]
            arts_for_sample = list(set(particular_codes.index) - set(correct_articles_id.get(cur_codes, [])))
            choised_ids = np.random.choice(arts_for_sample, self.k_articles, replace = self.replace)
            bad_answers = np.concatenate([bad_answers, choised_ids])
            bad_codes_ids = np.concatenate([bad_codes_ids, [cur_codes]*self.k_articles])
        return pd.DataFrame({'codes_id':bad_codes_ids, 'article_id':bad_answers}).astype({'article_id':int})
    
    '''
    `params`:
        `article_ids`: list which contain ids of articles from `df_article.index`
    `return`:
        pd.DataFrame whith columns article_id, paragraph
    '''
    def get_article_paragraphs(self, article_ids):
        particular_paragraphs =  self.paragraphs[self.paragraphs.article_id.isin(article_ids)]
        article_paragraphs = particular_paragraphs.groupby('article_id').agg({'paragraph': ' '.join})
        return article_paragraphs[['article_id', 'paragraph']] 
    
    def create_dataset(self):
        df_art_to_par = self.paragraphs.groupby('article_id').agg({'paragraph': ' '.join})
        join_question_bridge = self.databridge
        #create question_id column for use it in loop in this function
        join_question_bridge['question_id'] = join_question_bridge.index
        #relation with paragraphs of article to obtain article paragraph
        join_paragraph_bridge = join_question_bridge[['article_id', 'codes_id', 'Text',  'question_id']].set_index('article_id').join(df_art_to_par, on = 'article_id')
        
        
        dataset = join_paragraph_bridge
        dataset.rename(columns = {'Text':  'question'}, inplace = True)
        dataset['article_id'] = dataset.index
        dataset['score'] = 1.0
        
        #create answers with score 0, because we have answers with score 1
        for quest_id in tqdm(dataset.question_id.unique()):
            #select data for particular question
            ds_part_quest = dataset[dataset.question_id == quest_id]
            #select codes and article id, to send in function for sampling bad index
            #because we shouldn't sample from ids, which score for the question is equal 1
            codes_arts = ds_part_quest.groupby('codes_id')['article_id'].apply(list).to_dict()
            bad_answers = self.sample_bad_answers(codes_arts)
            bad_answers['question_id'] = quest_id
            bad_answers['question'] = ds_part_quest.iloc[0].question
            bad_answers['score'] = 0.0
            bad_answers.set_index('article_id', inplace=True)
            bad_answers = bad_answers.join(df_art_to_par, on = 'article_id')
            dataset = pd.concat([dataset, bad_answers])
        
        dataset['article_id'] = dataset.index
        dataset = dataset[['question', 'paragraph', 'score', 'question_id', 'article_id']]
        dataset.dropna(inplace = True)
        dataset.index =  range(len(dataset))
        
        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        questions = self.dataset['question'].values[idx]
        paragraph = self.dataset['paragraph'].values[idx]
        scores = self.dataset['score'].values[idx]
        output = {'question': questions,'paragraph': paragraph, 'scores': scores}
        return output
