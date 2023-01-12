import pandas as pd
import os
import torch
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


def contrastive_collate_fn(batch):
    
    input_ids = torch.stack([i['input_ids'].clone().detach() for i in batch])
    attention_mask = torch.stack([i['attention_mask'].clone().detach() for i in batch])
    labels = torch.stack([torch.tensor(i['labels']).clone().detach() for i in batch])

    contrastive_examples = {}

    for index, label in enumerate(labels):
        for index2, label2 in enumerate(labels):
            if index not in contrastive_examples:
                    contrastive_examples[index] = []

            if index == index2:
                continue

            if label != label2:
                contrastive_examples[index].append((index2, torch.tensor([-1]).to(labels.device)))
            else:
                contrastive_examples[index].append((index2, torch.tensor([1]).to(labels.device)))


    # Hacky handling for the case where there is only one example per batch
    if len(contrastive_examples[0]) == 1:
        for index in contrastive_examples:
            contrastive_examples[index] *= 2

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'contrastive_pairs': contrastive_examples}

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

        if self.labels is not None:
            label = self.labels.values[index]
            label = self.target_dict[label]
        else:
            label = 0

        data = self.data.values[index]

                    
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

