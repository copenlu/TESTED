import pandas as pd
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict

from transformers import AutoTokenizer
import sys

sys.path.append('../..')
sys.path.append('../')
sys.path.append('../tools')


from weaklabeler.tools.transformer_tok import transformer_tok

class FewShotData(Dataset):
    def __init__(self, data: pd.DataFrame = None, labels: pd.DataFrame = None, \
                 tokenizer: AutoTokenizer = None, target_dict: Dict = {}):
        """Custom class for loading data

        Args:
            data (pd.DataFrame, optional): texts for loading. Defaults to None.
            labels (pd.DataFrame, optional): labels for the loader. Defaults to None.
            target_dict (Dict, optional): mapping in between authors and ids. Defaults to {}.
        """        
        self.data = data    
        self.labels = labels
        self.target_dict = target_dict
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.labels.values[index]
        data = self.data.values[index]
        
        label = self.target_dict[label]
                    
        tokenized_dataset = transformer_tok(data, self.tokenizer) 
        

        tokenized_dataset['attention_mask'] = tokenized_dataset['attention_mask'][0]
        tokenized_dataset['input_ids'] = tokenized_dataset['input_ids'][0]

        tokenized_dataset['labels'] = label


        return tokenized_dataset


if __name__ == '__main__':

    data = pd.read_csv('../Data/raw_labeled_stance_topics_dataset_5_topics.csv')

    train_x, val_x, train_y, val_y = train_test_split(data['text_after_hrash_clean'],data['author'] , \
                                    test_size=0.1, stratify = data['author'] )

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

    target_dict = {'face_masks':0, 'fauci':1, 'hydroxychloroquine': 2, 'school_closures':3,'stay_at_home_orders':4}

    train_dataset = FewShotData(data = train_x, labels = train_y, \
                                tokenizer = tokenizer, target_dict=target_dict )
    val_dataset = FewShotData(data = val_x, labels = val_y, \
                                tokenizer = tokenizer, target_dict=target_dict)

    train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = 8, shuffle=True)


    count = 0
    for i in train_dataloader:
        print(i)

        count += 1
        if count > 3:
            break

