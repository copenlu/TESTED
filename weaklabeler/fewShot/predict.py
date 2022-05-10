import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

import argparse
import pandas as pd
from model import Transformer_classifier
import sys
from transformers import AutoTokenizer

from tqdm import tqdm
tqdm.pandas()

from weaklabeler.tools.transformer_tok import transformer_tok
from typing import List


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def predict(texts:List , model:nn.Module, taget_dict: Dict = {}, tokenizer: AutoTokenizer = None) -> str:
    """

    Custom Inference

    Args:
        texts (List): text for prediction
        model (nn.Module): model for inference
        taget_dict (_type_, optional):  Mapping between Ids to authors. Defaults to {}.

    Returns:
        str: predicted author
    """
    inputs = [transformer_tok(text, tokenizer) for text in texts]
    inputs = [{'input_ids': inp['input_ids'].cuda(),'attention_mask':inp['attention_mask'].cuda() } \
                for inp in inputs]

    predictions = []
    for input in tqdm(inputs):
        input['labels'] = 1
        with torch.inference_mode():
            logits = model(**input)
            probs = F.softmax(logits, dim=1).squeeze(dim=0)

            predictions.append(taget_dict[torch.argmax(probs).item()])

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
    '--test_path',
    help="The path of the test"
    )

    args = parser.parse_args()


    model = torch.load(args.model_path,'cpu')
    model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.feat_extractor, use_fast=True)
    test = pd.read_csv(args.test_path, index_col=0)

    print("Predicting...\n")
    test['pred'] = predict(list(test['TEXT'].values), model=model, tokenizer=tokenizer)

    test.to_csv('results.csv')