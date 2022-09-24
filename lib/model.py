from typing import Tuple, Union

# data preparation imports
import nltk
from nltk.corpus import stopwords
import pymorphy2
import pymorphy2_dicts_ru
from pandarallel import pandarallel
import pandas as pd
import re
import json

# embedding generator imports
from torch.utils.data import Dataset, DataLoader
import transformers as ts
import torch
import numpy as np
from scipy.special import softmax

import warnings

warnings.filterwarnings('ignore')

import os


class BertDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

    def __len__(self):
        return self.texts.shape[0]


class BertClassifier(torch.nn.Module):

    def __init__(self, pretrained_weights, model_save_path, q_frozen=0.9):
        super(BertClassifier, self).__init__()

        self.bert = ts.AutoModel.from_pretrained(pretrained_weights, local_files_only=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = model_save_path
        self.tokenizer = ts.AutoTokenizer.from_pretrained(pretrained_weights, local_files_only=True)

        self.out_features = self.bert._modules['pooler'].dense.out_features
        self.clf = torch.nn.Linear(self.out_features, 2)

        self.q_frozen = q_frozen
        n_frozen = int(len(list(self.bert.parameters())) * self.q_frozen)

        for param in list(self.bert.parameters())[:n_frozen]:
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        embeds = self.bert(**kwargs)[1]

        return self.clf(embeds)


class NLPModel:
    emoji_reg = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        re.UNICODE
    )

    regexps_list = [
        (r'[!\"#$%&\'*,;<=>?@^`{|}~/\n\n\t]+', ' '),  # service symbols
        (r'[-]{2,}', ' '),  # sequences of 2 or more dashes
        (r'[\\]{2,}', ' '),  # sequences of 2 or more slashes
        (r'[ ?(.|/|,) ]', ' '),  # seps surrounded with whitespaces
        (r'[ ]{2,}', ' '),  # sequences of 2 or more whitespaces
        (r'[ /\n\n\t,.!]+$', ''),  # symbols in the end of line
        (r'^[ /\n\n\t,.!]+', ''),  # symbols in the beginning of line
        (r'( )*(@|собака)( )*([\w\.-]|точка)+', 'почта'),  # mail
        (r'youtube|youtu.be|ютаб|ютьюб|ютуб|утуб|тытруб|vk|dscrd|telegram|tg|телеграмма|телеграм|телега|viber|skype'
         r'|скайп|дискорд|вконтакте',
         ' соцсеть '),  # social media
        (r'.ru|.com|.pro|.be|ru|com|pro|be|ссылка', ' сайт '),  # websites
        (r'билайн|beeline|мегафон|megafon|мтс|mts|теле2|tele2', ' оператор ')  # phone operators
    ]

    def __init__(self):

        # Preprocessing handles
        self.stopwords = json.load(open('data/stopwords.json'))
        self.word2num_dict = json.load(open('data/words2nums.json'))
        self.stem_obj = pymorphy2.MorphAnalyzer(result_type=None)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.bert_clf = BertClassifier('../../models_data/bert', '')

        tmp = torch.load('../../models_data/clf.pt', map_location=self.device)
        self.bert_clf.clf.weight = torch.nn.Parameter(tmp['weight'])
        self.bert_clf.clf.bias = torch.nn.Parameter(tmp['bias'])

        self.tokenizer = self.bert_clf.tokenizer

    def lemmatize_and_remove_sw(self, row):

        tokens = [self.stem_obj.parse(x)[0][2] for x in row.split()]

        row = ' '.join([text for text in tokens if text not in self.stopwords and text != ' '])

        return row

    def remove_emojis(self, data):
        return re.sub(self.emoji_reg, '', data)

    def apply_regexps(self, data):
        res = data

        for pattern, rep_string in self.regexps_list:
            pat = re.compile(pattern, re.I)
            res = re.sub(pat, rep_string, res)

        return res

    def word_num_replace(self, data):
        res = data

        for k, v in self.word2num_dict.items():
            pat = re.compile(k)
            res = re.sub(pat, str(v), res)

        return res

    def prepare_data(self, X):

        # Reading data
        X['sign'] = X['title'] + ' ' + X['description']

        X.drop(columns=['title', 'description', 'datetime_submitted'], inplace=True)

        # print('Filtering stage')

        # Filtering
        pandarallel.initialize(use_memory_fs=True)

        # print('prep call')
        X.sign = X.sign.parallel_apply(self.lemmatize_and_remove_sw)
        # print('emoji call')
        X.sign = X.sign.parallel_apply(self.remove_emojis)
        # print('regexp call')
        X.sign = X.sign.parallel_apply(self.apply_regexps)
        # print('w2n call')
        X.sign = X.sign.parallel_apply(self.word_num_replace)

        return X.sign


def get_prediction(model, test_dataloader=None):
    # print('Bert stage')

    pred_list = []

    with torch.no_grad():
        for data in test_dataloader:
            input_ids = data['input_ids'].to(model.device)
            attention_mask = data['attention_mask'].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = softmax(outputs.detach().cpu(), axis=1)[:, 1]

            pred_list += preds.tolist()

    return np.array(pred_list)


def task1(data):
    model = NLPModel()

    data = model.prepare_data(data)

    test_dataset = BertDataset(data, tokenizer=model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=100)

    return get_prediction(model.bert_clf, test_loader)


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
