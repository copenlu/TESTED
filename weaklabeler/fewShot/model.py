import torch
from torch import nn
from torch.functional import tensordot
from torch.nn import functional as F
from torch import Tensor

from transformers import AutoModel
from typing import List


class Transformer_classifier(nn.Module):
    def __init__(self, feat_extractor_name: str = '', num_labels: int = 5, hidden_list:List=[512],linear_probe:bool = True):
        """Transformer Classifier

        Args:
            feat_extractor_name (str, optional): Name of the feature extracator from HF hub or torch Hub.
            num_labels (int, optional): The number of labels to be predicted. Defaults to 4.
            hidden_list (List, optional): The hidden layers output. Defaults to [512].
        """        
        super(Transformer_classifier, self).__init__()
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feat_extractor = AutoModel.from_pretrained(feat_extractor_name)

        # This can be made into a flag. But usually should be fine.

        if linear_probe:
            for param in feat_extractor.base_model.parameters():
                param.requires_grad = False

        self.hidden_list = hidden_list
            
        self.feat_extractor = feat_extractor
        self.num_labels = num_labels
        self.embeding_shape = self.get_extractor_output_shape() 
                
        self.linear_1 = nn.Linear(self.embeding_shape, self.hidden_list[0], device=device)

        if len(hidden_list) > 1:
            self.linears = [self.linear_1] + \
            [nn.Linear(hidden_list[i-1], self.hidden_list[i], device=device) for i in range(1, len(hidden_list))]
        else: 
            self.linears = [self.linear_1]
            
        self.dropouts = [nn.Dropout(0.1) for i in range(len(self.linears))]

        self.final_logits = nn.Linear(hidden_list[-1], self.num_labels, device = device)
        

    def get_extractor_output_shape(self):
        last_layer = list(self.feat_extractor.named_children())[-1]
        shape = list(last_layer[1].modules())[1].out_features
        return shape

    def __call__(self, input_ids:Tensor, attention_mask:Tensor, labels: Tensor, **kwargs):

        embedding = self.feat_extractor(input_ids, attention_mask)
        last_hidden_states = embedding.last_hidden_state

        for i in range(len(self.linears)):
            last_hidden_states = self.linears[i](last_hidden_states)
            last_hidden_states = self.dropouts[i](last_hidden_states)
        
        res = self.final_logits(last_hidden_states[:,0,:].view(-1, self.hidden_list[-1]))

        return res.view(-1, self.num_labels)