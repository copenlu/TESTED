import torch
from typing import Union
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import aim
from aim import Run
from weaklabeler.fewShot.eval import evaluate
from tqdm import tqdm
import time

from model import Transformer_classifier
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup


def get_optimizer(model, lr:float = 10e-5):
    no_decay = ['bias', 'LayerNorm.weight']

    # print([p for n, p in model.named_parameters()])

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

def train_mlp(model: nn.Module, optimizer: optim.AdamW, train_dataloader:DataLoader, \
    val_dataloader: DataLoader=None, epochs=10, patience:int = 5, val_step: int = 2, aim_run: Run = None) -> Union[Transformer_classifier, optim.AdamW]:
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
    schedule = get_linear_schedule_with_warmup( optimizer = optimizer, num_warmup_steps = int(train_steps_all/10), num_training_steps = train_steps_all)

    for epoch_i in range(epochs):

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0
        total_contrastive_loss = 0
        model.train()
        # progress_bar = tqdm(range(train_steps_all))

        for step, batch in tqdm(enumerate(train_dataloader)):

            batch_input = {k: v.to(device) for k, v in batch.items() if k != 'contrastive_pairs'}
            if 'contrastive_pairs' in batch:
                batch_input['contrastive_pairs'] = batch['contrastive_pairs']
            batch_labels = batch_input['labels'].view(-1)

            logits, contrastive_loss = model(**batch_input)

            loss = loss_fn(logits, batch_labels)\
            
            if contrastive_loss is not None:
                loss += contrastive_loss    
                aim_run.track(contrastive_loss.item() , name = "Contrastive_Loss", context = {'type':'train'}, step=step)
                total_contrastive_loss += contrastive_loss.item()

            aim_run.track(loss.item() , name = "Batch_Loss", context = {'type':'train'}, step=step)

            total_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

            optimizer.step()
            schedule.step()

            optimizer.zero_grad()
            # progress_bar.update(1)

        avg_train_loss = total_loss / len(train_dataloader)
        aim_run.track(avg_train_loss , name = "Loss", context = {'type':'train'}, epoch=epoch_i)

        if contrastive_loss is not None:
            avg_contrastive_loss = total_contrastive_loss / len(train_dataloader)
            aim_run.track(avg_contrastive_loss , name = "Contrastive_Loss", context = {'type':'train'}, epoch=epoch_i)


        if val_dataloader is not None and (epoch_i+1)%val_step == 0:
            print("Validation")

            val_loss, val_accuracy, all_preds, all_gt = evaluate(model, val_dataloader)
            

            all_preds = [pred.cpu().numpy() for pred in torch.concat(all_preds)]
            all_gt = [gt.cpu().numpy() for gt in torch.concat(all_gt)]

            #calculate the f1 score
            f1 = classification_report(all_gt, all_preds, output_dict=True)['macro avg']['f1-score']


            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            time_elapsed = time.time() - t0_epoch

            results_val_str = f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {f1:^9.2f} | {time_elapsed:^9.2f} "
            
            print("Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Time")
            print(results_val_str)

            aim_run.track(aim.Text(results_val_str+ "\n"), name = 'log_out')

            aim_run.track(val_loss , name = "Loss", context = {'type':'val'}, epoch=epoch_i)
            aim_run.track(val_accuracy , name = "Metric", context = {'type':'val'},epoch=epoch_i)


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