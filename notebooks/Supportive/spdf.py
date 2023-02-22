'''
Supportive functions for Pandas DataFrames
'''

import pandas as pd
import numpy as np
import razdel
from Supportive.snlp import spec_chars_hash

'''
Too slow because at this moment use cycles.

`params`:
    `df`: pd.DataFrame
    `index_level`: level_of_multiindex
    `columns`:list
`example`:
    import spdf
    
    mi = pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)])
    tdf = pd.DataFrame([1, 1, 2,], index = mi)
    tdf.index.get_level_values(0)
    print(spdf.drop_duplicates_by_index(tdf, columns = [0])):
    ------
         0
    1 2  1
    3 4  1
    5 6  2
'''
def drop_duplicates_by_index(df, columns:list, index_level = 0):
    indeces = df.index.get_level_values(index_level).unique()
    df = df[columns]
    df_unique = None
    for idx in indeces:
        sub_unique = df.loc[idx:idx].drop_duplicates()
        df_unique = pd.concat([df_unique, sub_unique], axis = 0)
    return df_unique  
    
'''
Extract target feature from row of dataframe and apply function for change target value
and not change anything else values
'''
def dec_df_apply(func_for_apply, target):
    def extr_feat_and_call(x):
        target_value = x[target]
        x[target] = func_for_apply(target_value)
        return x
    return extr_feat_and_call

def dec_tokenization_columns(columns):
    def tokenizer(x):
        for column in columns:
            values = [word.text for word in razdel.tokenize(x[column])]
            x[column] = ' '.join(filter(lambda w: w not in spec_chars_hash, values))
        return x
    return tokenizer
