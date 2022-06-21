import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

import argparse
import pandas as pd
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from weaklabeler.fewShot.data import FewShotData

from tqdm import tqdm
tqdm.pandas()

from weaklabeler.tools.transformer_tok import transformer_tok
from weaklabeler.tools.utils import get_targets, get_available_cpus
from torch.utils.data import DataLoader

from typing import List

from stancedetection.data import iterators as data_iterators
from stancedetection.models.nn import LTNRoberta
from stancedetection.util.mappings import TASK_MAPPINGS
from stancedetection.util.model_utils import batch_to_device, get_learning_rate
from stancedetection.util.util import NpEncoder, configure_logging, set_seed

import logging
from collections import defaultdict
from functools import partial


logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TYPES = {
    # Auto Model for Sequence Tagging
    "auto": AutoModelForSequenceClassification,
    # Label Transfer Network
    "ltn_loss": LTNRoberta,
    "ltn_scores": LTNRoberta,
    "ltn": LTNRoberta,
    # Label Embedding
    "lel": LTNRoberta,
}

def predict(texts:List , model:nn.Module, target_dict: Dict = {}, tokenizer: AutoTokenizer = None) -> List:
    """

    Custom Inference

    Args:
        texts (List): text for prediction
        model (nn.Module): model for inference
        taget_dict (_type_, optional):  Mapping between Ids to class. Defaults to {}.

    Returns:
        List: predicted classes
    """

    predictions = []

    # train_dataset = FewShotData(data = texts, labels = None, tokenizer = tokenizer, target_dict=target_dict )
    print(target_dict)

    available_workers = get_available_cpus()
    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=False, num_workers = available_workers)

    with torch.inference_mode():

        for batch in tqdm(train_dataloader):

            batch_input = {k: v.to('cuda') for k, v in batch.items()}
            # batch_labels = batch_input['labels'].view(-1)

            # logits = model(**batch_input)

            logits = model(**batch_input)
            probs = F.softmax(logits, dim=1)

            predictions.append(probs.cpu().numpy())




@torch.no_grad()
def evaluate(
    model,
    dataset,
    batch_size,
    is_test=False,
    num_workers=0,
    max_steps=None,
    use_san=False,
    use_lel=False,
):
    collate_fn = partial(data_iterators.collate_fn, add_san_masks=use_san, add_lel_masks=use_lel)
    dataloader = DataLoader(
        dataset,
        shuffle=not is_test,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    data_iterator = tqdm(dataloader, position=0 if is_test else 2, leave=True, desc="Evaluating")

    predictions = defaultdict(list)
    model.eval()
    eval_loss = 0
    last_step = 1
    for step, batch in enumerate(data_iterator):
        batch = batch_to_device(batch, device=DEFAULT_DEVICE)
        if max_steps and step > max_steps:
            break

        loss, logits = model(**batch)
        eval_loss += loss.item()

        probs = logits.softmax(-1).detach().cpu().numpy()
        predictions["probs"] += probs.tolist()
        predictions["pred_stance"] += probs.argmax(-1).tolist()
        predictions["true_stance"] += batch["labels"].detach().cpu().tolist()
        last_step = step + 1

    return predictions, eval_loss / last_step


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description="Stance Inference"
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

    args = parser.parse_args()


    model = torch.load(args.model_path,'cpu')
    model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.feat_extractor, use_fast=True)
    test = pd.read_csv(args.test_path, index_col=0)
    target_dict = get_targets(args.target_config_path)


    print("Predicting...\n")
    test['topic_pred'] = predict(test[args.text_col], model=model,target_dict=target_dict,tokenizer=tokenizer)

    test.to_csv('results.csv')