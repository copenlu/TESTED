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
from weaklabeler.fewShot.data import FewShotData
from weaklabeler.tools.utils import get_targets, get_available_cpus
from weaklabeler.fewShot.eval import evaluate
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup

import pandas as pd
import os

def train(model: nn.Module, optimizer: optim.AdamW, train_dataloader:DataLoader, \
    val_dataloader: DataLoader=None, epochs=10, patience:int = 2, val_step: int = 2, aim_run: Run = None) -> Union[Transformer_classifier, optim.AdamW]:
    """
    

    Args:
        model (nn.Module): The Transformer classifier
        optimizer (optim.AdamW): The optimiser for training
        train_loader (DataLoader): training loader
        epochs (int, optional): The amount of epochs to train. Defaults to 10.
        patience (int, optional): the patience for Early Stopping. Defaults to 2.
        val_step (int, optional): the validation rate.
        aim_run (Run, optional): Aim run for tracking. Defaults to None.

    Returns:
        Union[Transformer_classifier, optim.AdamW]: Model and the Optimizer
    """    
    

    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_accuracy = 0
    the_last_loss = 100
    trigger_times = 0

    train_steps_all = epochs * len(train_dataloader)
    schedule = get_linear_schedule_with_warmup( optimizer = optimizer, num_warmup_steps = 0,
                                num_training_steps = train_steps_all)

    for epoch_i in range(epochs):

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        model.train()
        # progress_bar = tqdm(range(train_steps_all))

        for step, batch in tqdm(enumerate(train_dataloader)):

            batch_input = {k: v.to(device) for k, v in batch.items()}
            batch_labels = batch_input['labels'].view(-1)

            logits = model(**batch_input)

            loss = loss_fn(logits, batch_labels)

            aim_run.track(loss.item() , name = "Batch_Loss", context = {'type':'train'})

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            schedule.step()

            optimizer.zero_grad()
            # progress_bar.update(1)

        avg_train_loss = total_loss / len(train_dataloader)
        aim_run.track(avg_train_loss , name = "Loss", context = {'type':'train'})

        if val_dataloader is not None and (epoch_i+1)%val_step == 0:
            print("Validation")

            val_loss, val_accuracy, _, _ = evaluate(model, val_dataloader)
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            time_elapsed = time.time() - t0_epoch

            results_val_str = f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
            print(results_val_str)

            aim_run.track(aim.Text(results_val_str+ "\n"), name = 'log_out')

            aim_run.track(val_loss , name = "Loss", context = {'type':'val'})
            aim_run.track(val_loss , name = "Metric", context = {'type':'val'})


            if abs(val_loss - the_last_loss) < 1e-5 or val_loss > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return model, optimizer
            
            the_last_loss = val_loss

          
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
    return model, optimizer



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
        '--batch_size',type=int,
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

    args = parser.parse_args()



    aim_run = Run(repo='.', experiment=args.experiment_name, run_hash=None)
    aim_run['model_metadata'] = vars(args)
    os.makedirs(args.model_save_path, exist_ok=True)


    data = pd.read_csv(args.training_path)

    ratio = 1.0 - (args.shot_num / len(data))

    if ratio < 0:
        raise ValueError("The number of examples to consider per few_shot experiment is larger than the number of examples in the dataset.")

    train_x, val_x, train_y, val_y = train_test_split(data[args.text_col],data[args.target_col] , \
                                    test_size=ratio, stratify = data[args.target_col])


    print(len(train_x), len(val_x))

    tokenizer = AutoTokenizer.from_pretrained(args.feat_extractor, use_fast=True)

    target_dict = get_targets(args.target_config_path)

    train_dataset = FewShotData(data = train_x, labels = train_y, \
                                tokenizer = tokenizer, target_dict=target_dict )
    val_dataset = FewShotData(data = val_x, labels = val_y, \
                                tokenizer = tokenizer, target_dict=target_dict)


    available_workers = get_available_cpus()
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers = available_workers)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=True, num_workers = available_workers)


    model = Transformer_classifier(args.feat_extractor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate)

    model, optimizer = train(model = model, optimizer=optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader,\
         epochs=args.epochs,  val_step=args.val_step, aim_run=aim_run)

    print("Training Complete\n")

    print("Validating\n")
    val_loss, val_accuracy, preds, gt = evaluate(model.to('cuda'), val_dataloader)
    print(classification_report(torch.cat(preds, 0).cpu().numpy(), torch.cat(gt, 0).cpu().numpy()))


    torch.save(model, f"{os.path.join(args.model_save_path,args.experiment_name)}.pth")
    torch.save({'optimizer_state_dict': optimizer.state_dict()}, f"{os.path.join(args.model_save_path,args.experiment_name)}_optimizer.pth") 



