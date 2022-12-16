import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import List, Union
from tqdm import tqdm
import numpy as np

def evaluate(model:nn.Module, val_dataloader: DataLoader) -> Union[Tensor, Tensor, List, List]:
    """
    Args:
        model (nn.Module): The model being tested
        val_dataloader (DataLoader): Torch Dataloader for 

    Returns:
        Union[Tensor, Tensor, List, List]: all outputs
    """
    
    val_accuracy, val_loss, all_preds, all_gt = [], [], [], []

    try:      
        loss_fn = nn.CrossEntropyLoss()
        device = 'cuda'
        model.eval()

        for batch in tqdm(val_dataloader):

            batch_input = {k: v.to(device) for k, v in batch.items() if k != 'contrastive_pairs'}

            if 'contrastive_pairs' in batch:
                batch_input['contrastive_pairs'] = batch['contrastive_pairs']
            batch_labels = batch_input['labels'].view(-1)

            model.zero_grad()


            with torch.no_grad():
                logits, contrastive_loss = model(**batch_input)

            loss = loss_fn(logits, batch_labels)
            if contrastive_loss is not None:
                loss += contrastive_loss
            val_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()

            all_preds.append(preds)
            all_gt.append(batch_labels)
            
            accuracy = (preds == batch_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

    except RuntimeError as e:
        print("Unable to calculate the validation loss with error: ",e)
        return val_loss, val_accuracy, all_preds, all_gt

    return val_loss, val_accuracy, all_preds, all_gt