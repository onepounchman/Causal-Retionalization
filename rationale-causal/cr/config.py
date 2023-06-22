from pathlib import Path
import torch

class Config:
    ROOT_DIR = Path('/home/ec2-user/SageMaker')
    PROJECT_NAME = 'rationale-causal'

    PROJECT_DIR = ROOT_DIR / PROJECT_NAME
    EXP_DIR = PROJECT_DIR / 'experiments'
    DATA_DIR = PROJECT_DIR / 'data'
    CACHE_DIR = PROJECT_DIR / 'cache'

    TOK_KWARGS = {
        'padding': 'max_length',
        'truncation': True,
        'return_offsets_mapping': True,
        'return_tensors': 'pt',
    }


    BEER_LABEL = {
        'neg': 0,
        'pos': 1,
    }
    HOTEL_LABEL = {
        'neg': 0,
        'pos': 1,
    }





