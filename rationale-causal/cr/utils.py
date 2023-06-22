import os
import yaml
import glob
from argparse import Namespace
import torch
from torch.optim import AdamW, Adam
from cr.dataloaders import SentimentDataLoader
from cr.models.sentiment_models import CausalSentimentTokenModel
from cr.config import Config
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply
import warnings


config = Config()


def update_args(args):
    """
    Update args to handle version differences when loading models.
    E.g., older models might miss some arguments.
    """
    if not hasattr(args, 'optimizer'):
        setattr(args, 'optimizer', 'adamw')
    if not hasattr(args, 'dataset_split'):
        setattr(args, 'dataset_split', 'all')
    if not hasattr(args, 'use_gold_rationale'):
        setattr(args, 'use_gold_rationale', False)
    if not hasattr(args, 'use_neg_rationale'):
        setattr(args, 'use_neg_rationale', False)
    if not hasattr(args, 'gamma'):
        setattr(args, 'gamma', 1.0)
    if not hasattr(args, 'dropout_rate'):
        setattr(args, 'dropout_rate', 0.1)
    if not hasattr(args, 'fix_input'):
        setattr(args, 'fix_input', None)
    if not hasattr(args, 'mask_scoring_func'):
        setattr(args, 'mask_scoring_func', 'linear')
    if not hasattr(args, 'flexible_prior'):
        setattr(args, 'flexible_prior', None)
    if not hasattr(args, 'budget_ratio'):
        setattr(args, 'budget_ratio', None)
    if not hasattr(args, 'cache_dir'):
        setattr(args, 'cache_dir', config.CACHE_DIR)
    return args


def args_factory(args):
    # model specific
    if args.encoder_type == 'bert-base-uncased':
        args.encoder_hidden_size = 768
    elif args.encoder_type == 'distilbert-base-uncased':
        args.encoder_hidden_size = 768
    elif args.encoder_type == 'roberta-large':
        args.encoder_hidden_size = 1024



    args.dir_name=f'mu={args.mu}-k={args.k}-lr={args.lr}-beta={args.beta}-epoch={args.num_epoch}-tau={args.tau}'
    args.file_name = 'model-seed={seed}-step={step}-acc={best_score:.2f}.pt'
    args.ckpt_dir = os.path.join(config.EXP_DIR, args.dataset_name,args.aspect,args.method,args.scale,args.dir_name)
    args.file_path = os.path.join(args.ckpt_dir, args.file_name)
    args.args_path = os.path.join(args.ckpt_dir, 'args.yaml')
    args.ckpt = not args.disable_ckpt
    args.use_cuda = not args.disable_cuda
    
    if args.debug:
        args.ckpt = False
        args.disable_ckpt = True

    args.best_score = float('-inf')
    args.best_logpns = float('-inf')
    args.total_seen = 0
    args.global_step = 0
    return args



def get_model_class(args):
    print(args.model_type)
    model_class = CausalSentimentTokenModel
    return model_class
  
def get_optimizer_class(args):
    if args.optimizer == 'adamw':
        optimizer_class = AdamW
    elif args.optimizer == 'adam':
        optimizer_class = Adam
    else:
        raise ValueError('Optimizer type not implemented.')
    return optimizer_class


def get_dataloader_class(args):
    if args.dataset_name in ('beer','hotel'):
        dataloader_class = SentimentDataLoader
    else:
        raise ValueError('Dataloader not implemented.')
    return dataloader_class


def save_args(args):
    with open(args.args_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f'Arg file saved at: {args.args_path}')

def save_ckpt(args, model, optimizer, latest=False,best_path='',seed=0):

    latest_name = f'mu={args.mu}-beta={args.beta}.ckpt'
    if not latest:
        args.best_ckpt_path = args.file_path.format(
              seed=seed,
              step=args.global_step,
              best_score=args.best_score * 100
          )

        checkpoint = {'ckpt_path': args.best_ckpt_path}
        print('not latest:', checkpoint)
    else:
        checkpoint = {'ckpt_path': os.path.join(args.ckpt_dir, latest_name)}
        print('latest:', checkpoint)
    checkpoint['args'] = vars(args)

    states = model.state_dict() 
    checkpoint['states'] = states
    checkpoint['optimizer_states'] = optimizer.state_dict()
    
    if not latest:
            if best_path != '':
                os.remove(best_path)
            best_path=checkpoint['ckpt_path']
    
    torch.save(checkpoint, checkpoint['ckpt_path'])
    return best_path

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
