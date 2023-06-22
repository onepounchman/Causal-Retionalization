import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

from cr.transformer_models.modeling_bert import BertModel
from cr.transformer_models.modeling_distilbert import DistilBertModel
from cr.transformer_models.modeling_roberta import RobertaModel




def get_encoder_model_class(args):
    if args.encoder_type == 'bert-base-uncased':
        return BertModel
    elif args.encoder_type == 'distilbert-base-uncased':
        return DistilBertModel
    elif args.encoder_type == 'roberta-large':
        return RobertaModel

class TokenLevelEncoder(nn.Module):
    def __init__(self, args):
        super(TokenLevelEncoder, self).__init__()
        self.args = args
        encoder_model_class = get_encoder_model_class(args)
        self.model = encoder_model_class.from_pretrained(args.encoder_type, cache_dir=args.cache_dir)

    def forward(self, input_ids, attention_mask):
        token_reps = self.model(input_ids, attention_mask)[0]
        return token_reps

    
