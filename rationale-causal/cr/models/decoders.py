import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from cr.transformer_models.modeling_bert import BertForSequenceClassification




from cr.models.encoders import get_encoder_model_class

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args

        self.model = BertForSequenceClassification.from_pretrained(
            args.decoder_type,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_dir=self.args.cache_dir,
        )
    
    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output


