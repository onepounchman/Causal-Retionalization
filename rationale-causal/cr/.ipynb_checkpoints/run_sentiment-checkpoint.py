import os
import sys
import argparse
from pathlib import Path
from time import time
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from cr.utils import (
    get_model_class,
    get_optimizer_class,
    get_dataloader_class,
    args_factory,
    save_args,
    save_ckpt
)
from cr.evaluate import (
    eval_after_train,
    evaluate,
    evaluate_for_stats,
    evaluate_for_pns,
    gold_rationale_capture_rate,
    cal_pns
)
from cr.logging_utils import log
from cr.visualize import visualize_rationale
from cr.config import Config
import warnings

from transformers import BertTokenizer


config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

def train_epoch(seed,epoch, model, dl, optimizer, args,best_path):
    
    model.train()
    num_batches = dl.train.dataset.num_batches
    args.epoch = epoch

    t = time()
    total_loss = 0
    total_kl_loss = 0  # does not apply to non-VIB models
    total_correct = []
    
    
    for batch_idx, batch in enumerate(dl.train):
        output = model(batch)
        loss = output['loss']
        pred_loss=output['pred_loss']
        logits = output['logits']
        ind = torch.cat((1-batch['labels'].view(-1,1),batch['labels'].view(-1,1)),1).view(-1,2)
        prob_raw = torch.sum(F.softmax(logits,dim=-1)*ind,1).clone().detach()
        

        token_z = output['token_z']
        kl_loss = output['kl_loss']
            
        if args.inspect_gpu:
            os.system('nvidia-smi')
            input()
        
        
        if args.dataparallel:
            if 'kl_loss' in output:
                total_kl_loss += output['kl_loss'].sum().item()
            total_loss += loss.sum().item()
            total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
            args.total_seen += logits.size(0)
            loss = loss.mean()
        else:
            if 'kl_loss' in output:
                total_kl_loss += output['kl_loss'].item()
            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
            args.total_seen += logits.size(0)
      
        
        if args.grad_accumulation_steps > 1:
            loss = loss / args.grad_accumulation_steps

        loss.backward()


        if (args.global_step + 1) % args.grad_accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
              if (args.global_step + 1) % args.print_every == 0:
                
                train_acc = sum(total_correct) / len(total_correct)
                train_loss = total_loss / len(total_correct)

    
                elapsed = time() - t
      
                log_dict = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'num_batches': num_batches,
                    'acc': train_acc,
                    'loss': train_loss,
                    'elapsed': elapsed,
                    'global_step': args.global_step,
                }

                
                log('train', args, log_dict)
               
                if args.wandb:
                    log_dict = {
                    'acc': train_acc,
                    'loss': train_loss
                }
 
                    wandb.log(log_dict)
                total_loss = 0
                total_correct = []
                t = time()
            
              if (args.global_step + 1) % args.eval_interval == 0:
                dev_acc = evaluate('dev', model, dl, args)  
                print('dev_acc',dev_acc)
                print('best_score',args.best_score)

                if dev_acc > args.best_score:
                    args.best_score = dev_acc
                if args.ckpt:
                    _ = save_ckpt(args, model, optimizer, latest=False,best_path=best_path,seed=seed)
                
                
                model.train()


        args.global_step += 1
        
    
    return best_path



def main(args):
    torch.cuda.set_device(args.device_id)
    training_start_time = time()
    args = args_factory(args)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    save_args(args)

    cached_features_file = os.path.join(args.cache_dir, 'cached_{}_{}_{}_{}_ml{}_bz{}'.format(
            args.dataset_name,
            args.scale,
            args.aspect,
            args.dataset_split,
            str(args.max_length),
            args.batch_size,
        ))


    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file:", cached_features_file)
        dl = torch.load(cached_features_file)
    else:
            dataloader_class = get_dataloader_class(args)
            dl = dataloader_class(args)
            print("Saving features into cached file:", cached_features_file)
            torch.save(dl, cached_features_file)


    seed = args.seed
    if args.wandb:
        name = f'{args.scale}-{args.aspect}-{args.method}'
        para_config={'method':args.method,'mu':args.mu,'k':args.k,'lr':args.lr,'beta':args.beta,'max-epoch':args.num_epoch,'tau':args.tau}
        wandb.init(project='causal-rationale_summary', name=name,config=para_config)
        
        
    args.best_score = float('-inf')
    args.global_step= 0
      

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_class = get_model_class(args)
    model = model_class(args=args)
          
    model = model.cuda()
    if args.dataparallel:
        model = nn.DataParallel(model)
      
    print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'All params      : {sum(p.numel() for p in model.parameters())}')
    optimizer_class = get_optimizer_class(args)
    optimizer = optimizer_class(model.parameters(), lr=args.lr)

    best_path = ''
    for epoch in range(args.num_epoch):
        print(f'seed:{seed},epoch:{epoch}')
        best_path=train_epoch(seed,epoch, model, dl, optimizer, args,best_path)

    eval_after_train(model,dl,args)
    save_args(args)
    if args.wandb:
        wandb.finish()
    print(f'seed{seed} run time elapsed: {(time() - training_start_time) / 60:.2f} min')
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--scale", type=str, default="normal", help="[small |normal]")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc]")
    parser.add_argument("--aspect", type=str, required=True, help="Look, Aroma,Palate for beer;Location for hotel")
    parser.add_argument("--dataset-split", type=str, default="all", help="[all | train | dev | test]")
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--encoder-type", type=str, default="bert-base-uncased")
    parser.add_argument("--decoder-type", type=str, default="bert-base-uncased")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=config.CACHE_DIR)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--attack_path", type=str, default=None)

    # cuda
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--inspect-gpu", action="store_true")
    parser.add_argument("--disable-cuda", action="store_true")

    # printing, logging, and checkpointing
    parser.add_argument("--print-every", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--disable-ckpt", action="store_true")

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)

    parser.add_argument("--k", type=int, default=3) # number of samples for PNS
    parser.add_argument("--mu", type=float, default=0.1) # weight for PNS
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pi", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    #parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--visualize", action="store_true")


    
    
    args = parser.parse_args()
    args.method = args.run_name.split('_')[1]
    
    
    if args.debug:
        args.print_every = 2
        args.batch_size = 3
        args.eval_interval = 20
        args.num_epoch = 100
        args.dataparallel = False
        args.overwrite_cache = False
        args.max_length = 200

    main(args)