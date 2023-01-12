from random import shuffle
import torch
from torch.utils.data import DataLoader

import tqdm as tqdm
from model import Transformer_classifier
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


import argparse
import torch
from typing import Union
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import aim
from aim import Run
from weaklabeler.fewShot.data import FewShotData, contrastive_collate_fn
from weaklabeler.fewShot.trainers import train_mlp, get_optimizer
from weaklabeler.tools.utils import get_targets, get_available_cpus
from weaklabeler.fewShot.eval import evaluate
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup

import pandas as pd
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description="Transformer Training")

    parser.add_argument(
    '--model_save_path',
    help="The name of the model to be saved"
    )

    parser.add_argument(
    '--experiment_name',
    help="The name of the experiment"
    )


    parser.add_argument(
    '--feat_extractor',
    help="The name of the feature extractor transformer"
    )

    parser.add_argument(
    '--training_path',type=str,
    help="The path to the training data"
    )
    parser.add_argument(
    '--valid_path',type=str,default=None,
    help="The path to the validation data"
    )

    parser.add_argument(
        '--batch_size',type=int,default=8,
        help="The batch size used during training"
        )

    parser.add_argument(
        '--learning_rate',type=float,
        help="The learning rate used during training"
        )

    parser.add_argument(
        '--epochs',type=int,default=3,
        help="The epochs of training"
        )

    parser.add_argument(
        '--shot_num',type=int,default=10,
        help="The number of total (not per class) example to consider per few_shot experiment"
        )

    parser.add_argument(
        '--val_step',type=int,default=2,
        help="The number of example to consider per few_shot experiment"
        )

    parser.add_argument('--text_col',
                        help="Name of column containing training text")

    parser.add_argument('--target_col',
                        help="Name of column containing training targets")
    
    parser.add_argument('--target_config_path',
                        help="Path to target config file, with the mapping from target to id")

    parser.add_argument('--filter_out',
                        help="The Filtering that should be applied using the coumn and name to \
                            the data before proceeding. sperate with / as (col/name)")

    parser.add_argument('--num_labels', type=int, default=None, help="Number of labels")
    parser.add_argument('--linear_probe', action='store_true', help="Use linear probing", default=False)
    parser.add_argument('--contrastive', action='store_true', help="Contrastive training", default=False)

    args = parser.parse_args()


    aim_run = Run(repo='.', experiment=args.experiment_name, run_hash=None)
    aim_run['model_metadata'] = vars(args)
    os.makedirs(args.model_save_path, exist_ok=True)


    data = pd.read_csv(args.training_path)

    if args.filter_out is not None:
        print(args.filter_out)
        filte_col, filter_name = args.filter_out.split("/")
        data = data[data[filte_col] != filter_name]

    print(len(data))
    ratio = 1.0 - (args.shot_num / len(data))

    if ratio < 0:
        raise ValueError("The number of examples to consider per few_shot experiment is larger than the number of examples in the dataset.")

    texts = data[args.text_col].tolist()

    train_x, val_x, train_y, val_y = train_test_split(data[args.text_col],data[args.target_col] , \
                                    test_size=ratio, stratify = data[args.target_col], shuffle=True)


    #statistics for length of sentences for train and val with distributio, median, mean, std, min, max
    import numpy as np




    print("Max length of sentence in train: ", max([len(x) for x in train_x.tolist()]))
    print("Min length of sentence in train: ", min([len(x) for x in train_x.tolist()]))
    print("Average length of sentence in train: ", sum([len(x) for x in train_x.tolist()])/len(train_x))
    print("Median length of sentence in train: ", np.median([len(x) for x in train_x.tolist()]))


    if args.valid_path is not None:
        valid_data = pd.read_csv(args.valid_path)
        if args.filter_out is not None:
            filte_col, filter_name = args.filter_out.split("/")
            valid_data = valid_data[valid_data[filte_col] == filter_name]

        val_x, val_y = valid_data['sentence'], valid_data[args.target_col]



    print("__"*20)
    print("Max length of sentence in val: ", max([len(x) for x in val_x.tolist()]))
    print("Min length of sentence in val: ", min([len(x) for x in val_x.tolist()]))
    print("Average length of sentence in val: ", sum([len(x) for x in val_x.tolist()])/len(val_x))
    print("Median length of sentence in val: ", np.median([len(x) for x in val_x.tolist()]))

    #plot the distribution of the length of sentences without outliers
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set_style("darkgrid")
    # sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    # plt.figure(figsize=(10,5))
    # sns.distplot([len(x) for x in train_x.tolist()], bins=100, kde=False, rug=False, label="train")
    # sns.distplot([len(x) for x in val_x.tolist()], bins=100, kde=False, rug=False, label="val")
    # plt.legend()
    # plt.title("Distribution of the length of sentences")
    # plt.xlabel("Length of sentence")
    # plt.ylabel("Frequency")
    # plt.show()


    print("Training data size: ", len(train_x))
    print("Validation data size: ", len(val_x))

    tokenizer = AutoTokenizer.from_pretrained(args.feat_extractor,model_max_length=512, use_fast=True)

    target_dict = get_targets(args.target_config_path)

    train_dataset = FewShotData(data = train_x, labels = train_y, \
                                tokenizer = tokenizer, target_dict=target_dict )
    val_dataset = FewShotData(data = val_x, labels = val_y, \
                                tokenizer = tokenizer, target_dict=target_dict)


    available_workers = get_available_cpus()
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True,\
         num_workers = available_workers, collate_fn=contrastive_collate_fn if args.contrastive else None)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=True,\
         num_workers = available_workers, collate_fn=contrastive_collate_fn if args.contrastive else None)


    model = Transformer_classifier(feat_extractor_name = args.feat_extractor, num_labels = args.num_labels, linear_probe = args.linear_probe)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    optimizer = get_optimizer(model, args.learning_rate)
    # optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate)

    model, optimizer = train_mlp(model = model, optimizer=optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader,\
         epochs=args.epochs,  val_step=args.val_step, aim_run=aim_run)

    print("Training Complete\n")

    print("Validating\n")
    val_loss, val_accuracy, preds, gt = evaluate(model.to('cuda'), val_dataloader)

    report_data = classification_report(torch.cat(preds, 0).cpu().numpy(), torch.cat(gt, 0).cpu().numpy(), output_dict=True)
    aim_run["classification_report"] = report_data
    print(report_data)


    torch.save(model, f"{os.path.join(args.model_save_path,args.experiment_name)}.pth")
    torch.save({'optimizer_state_dict': optimizer.state_dict()}, f"{os.path.join(args.model_save_path,args.experiment_name)}_optimizer.pth") 




"""
python weaklabeler/fewShot/train.py --experiment_name fewShot_epochs=5_n=128 --feat_extractor roberta-base \
--training_path weaklabeler/Data/few_shot_diverse_sample.csv --model_save_path weaklabeler/models/ \
--num_labels 4 --batch_size 16 --epochs 5 --learning_rate 0.0001 --shot_num 512 --val_step 3 \
--text_col sentence --target_col label_name --target_config_path weaklabeler/configs/few_shot_diverse_sample.json 
"""

"""
python weaklabeler/fewShot/train.py --experiment_name fewShot_epochs=5_n=8000_contrastive=True --feat_extractor roberta-base \
--training_path weaklabeler/Data/stance_data/few_shot_diverse_sample.csv --model_save_path weaklabeler/models/ \
--num_labels 4 --batch_size 16 --epochs 5 --learning_rate 0.00001 --shot_num 8000 --val_step 1 \
--text_col sentence --target_col label_name --target_config_path weaklabeler/configs/few_shot_diverse_sample.json --contrastive
"""