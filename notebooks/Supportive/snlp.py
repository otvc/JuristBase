import pandas as pd
import numpy as np
import string
import re
import pymorphy2
import razdel


spec_chars = string.punctuation + '\n\xa0«»\t—…'

spec_chars_hash = dict.fromkeys(spec_chars)

spec_chars_regex = ['|' + re.escape(c) for c in spec_chars]
spec_chars_regex = '('+'|'.join(list(spec_chars_regex))[1:]+')'

'''
Clean df column's from spec symbols.
    params:
        `data`:
        `target`:
        `spec_symbols`:
'''
def df_text_clean(data, target, spec_symbols = spec_chars_regex):
    data[target] = data[target].str.replace(spec_symbols, '', regex=True)
    return data

def get_all_word_case(word):
    mparse = pymorphy2.MorphAnalyzer()
    p_word = mparse.parse(word)[0]
    
    cases = {'nomn','gent','datv','accs','ablt','loct','voct','gen2','acc2','loc2'}
    word_cases = set()
    for case in cases:
        word_cases.add(p_word.inflect({case}).word)
    
    return word_cases
    
'''
Function which normalize words in text
`params`:
    `text`:str :

`example`:
    text = 'здравствуйте!  расскажите мне пожалуйста детали и что в таком случае делать.  вам нужно явиться по вызову правоохранительных'
    norm_text = snlp.text_norm(text)
    print(norm_text)
    -----
    здравствуйте ! рассказать я пожалуйста деталь и что в такой случай делать . вы нужно явиться по вызов правоохранительный'
'''
def text_norm(text:str):
    text = text.lower()
    gen_tokenized_text = razdel.tokenize(text)
    
    morph = pymorphy2.MorphAnalyzer()
    func_convert = lambda word: word if word in spec_chars_hash else morph.parse(word.text)[0].normal_form
    
    norm_words = list(map(func_convert, gen_tokenized_text))
    norm_text = ' '.join(norm_words)
    
    return norm_text

def permutations(nums):
    permutations = []
    n = len(nums)
    def backtracking_comb(cur_len, used_n:list):
        if cur_len == n:
            permutations.append(list(used_n))
            return
        for i in nums:
            if i not in used_n:
                backtracking_comb(cur_len + 1, used_n + [i])
    backtracking_comb(0, [])
    return permutations

'''
Create permutation each word from list1 with each word in list2.
`node`: you can create function not only for 2 and for any k of list,
but i'm needed in permutation of two lists
'''
def get_two_list_permutations(list1:list[str], list2:list[str], sep = ' '):
    lists = set([tuple(list1), tuple(list2)])
    answers = []
    for flist in lists:
        for slist in lists - set([flist]):
            answers.extend([word1 + sep + word2 for word1 in flist for word2 in slist])
    return answers
    
