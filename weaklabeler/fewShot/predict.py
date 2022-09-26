from itertools import count
from random import sample
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

import argparse
import pandas as pd
from model import Transformer_classifier
import sys
from transformers import AutoTokenizer
from weaklabeler.fewShot.data import FewShotData

from tqdm import tqdm
tqdm.pandas()

from weaklabeler.tools.transformer_tok import transformer_tok
from weaklabeler.tools.utils import get_targets, get_available_cpus
from torch.utils.data import DataLoader

from typing import List
import os


def batcher(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def predict(texts:List , model:nn.Module, target_dict: Dict = {}, tokenizer: AutoTokenizer = None) -> str:
    """

    Custom Inference

    Args:
        texts (List): text for prediction
        model (nn.Module): model for inference
        taget_dict (_type_, optional):  Mapping between Ids to class. Defaults to {}.

    Returns:
        str: predicted class
    """

    predictions = []

    dataset = FewShotData(data = texts, labels = None, tokenizer = tokenizer, target_dict=target_dict )
    # print(target_dict)


    label_count = len(target_dict)
    
    available_workers = get_available_cpus()
    dataloader = DataLoader(dataset, batch_size = 128, shuffle=False, num_workers = available_workers)

    model.eval()
    with torch.inference_mode():

        for batch in dataloader:

            batch_input = {k: v.to('cuda') for k, v in batch.items()}
            # batch_labels = batch_input['labels'].view(-1)

            # logits = model(**batch_input)

            logits = model(**batch_input)
            probs = F.softmax(logits, dim=1)

            for index in range(len(probs)):
                prob = probs[index]

                # create an other class for the prediction if no class has high probability
                # print(prob.max().item(),target_dict[str(torch.argmax(prob).item())])
                # print("_________________________________________")
                if prob.max().item() < 0.5:
                    predictions.append("NONE")
                else:
                    predictions.append(target_dict[str(torch.argmax(prob).item())])


        # inputs = [transformer_tok(text, tokenizer) for text in text_batch]
        # inputs = [{'input_ids': inp['input_ids'].cuda(),'attention_mask':inp['attention_mask'].cuda() } \
        #             for inp in inputs]

        # for input in tqdm(inputs):
        #     input['labels'] = 1
        #     with torch.inference_mode():
        #         logits = model(**input)
        #         probs = F.softmax(logits, dim=1).squeeze(dim=0)

        #         predictions.append(target_dict[torch.argmax(probs).item()])

    return predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description="Transformer Inference"
    )

    parser.add_argument(
    '--model_path',
    help="The path of the model to load"
    )

    parser.add_argument(
    '--feat_extractor',
    help="The name of the feature extractor transformer"
    )

    parser.add_argument(
    '--text_col',
    help="Name of column containing training text"
    )

    parser.add_argument(
    '--test_path',
    help="The path of the test"
    )

    parser.add_argument(
    '--target_config_path',
    help="Path to target config file, with the mapping from target to id"
    )

    parser.add_argument("--save_col", help="Name of column to save predictions")
    

    args = parser.parse_args()


    model = torch.load(args.model_path,'cuda')
    model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.feat_extractor, use_fast=True)
    test = pd.read_csv(args.test_path, index_col=0, chunksize=128)
    target_dict = get_targets(args.target_config_path)


    print("Predicting...\n")
    # for chunk in test:
    #     chunk[args.save_col] = predict(test[args.text_col], model=model,target_dict=target_dict,tokenizer=tokenizer)

    header = True
    for chunk in tqdm(test):

        chunk[args.save_col] = predict(chunk[args.text_col], model=model,target_dict=target_dict,tokenizer=tokenizer)
        chunk.to_csv( 'results.csv', header=header, mode='a')


        header = False

    # test.to_csv('results.csv')