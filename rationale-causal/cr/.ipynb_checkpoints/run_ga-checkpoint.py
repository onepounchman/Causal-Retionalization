"""
Main training script for Beer and Hotel.
"""
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
from sklearn import metrics
import wandb

from cr.utils import (
    get_model_class,
    get_optimizer_class,
    get_dataloader_class,
    args_factory,
    save_args,
    save_ckpt,
    load_ckpt,
    build_vib_path
)
from cr.logging_utils import log_ga
from cr.visualize import visualize_rationale_ga
from cr.stats import gold_rationale_capture_rate,cal_pns
from cr.config import Config

config = Config()


    
def train_epoch(seed,epoch, model, dl, optimizer, args,best_path):
    

    

    model.train()
    num_batches = dl.train.dataset.num_batches
    args.epoch = epoch

    t = time()
    total_loss = 0
    total_kl_loss = 0 
    total_correct = []
    
    for batch_idx, batch in enumerate(dl.train):
        

        output = model(batch)
        loss = output['loss']
        pred_loss=output['pred_loss']
        logits = output['logits']
        
        ind=torch.cat((1-batch['labels'].view(-1,1),batch['labels'].view(-1,1)),1).view(-1,2)
        prob_raw=torch.sum(F.softmax(logits,dim=-1)*ind,1).clone().detach()
      
        
        if args.method=='vib' or args.method=='causal':
            token_z=output['token_z']
            kl_loss=output['kl_loss']
          
        if args.method=='causalfc':
            log_prob_token=output['log_prob_token']    
 
          
        if args.method=='causalfc':
              loss=loss-args.mu*torch.mean(log_prob_token)

        else:
              loss=loss


            
            
            
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
        
        
        #epoch>=args.warm_epoch and 
        if args.method=='causal':
            # negative log PN
            k=args.k
            l=len(batch['labels']) # some batch doesn't have 64
            input_ids = batch['input_ids']
            labels=batch['labels']
            attention_mask=batch['attention_mask']
            all_indexs = []
            
            if args.flip=='all':
              with torch.no_grad():
                model.eval()
                z=model.forward_train_pred(batch)['token_z']
                model.train()
              
              # test mode, positive prediction
              output_posi = model.decoder(batch['input_ids'],z,
              batch['labels'])
                
              logits_posi = output_posi['logits']
              ce=nn.CrossEntropyLoss()
              celoss_posi=ce(F.softmax(logits_posi,dim=-1),batch['labels'])
              celoss_posi = args.mu*celoss_posi/ args.grad_accumulation_steps
              celoss_posi.backward()
                
                
                
              
              # test mode all negative predictions
              for i in range(l):
                num_ration=torch.sum(z[i,:]).item()
                
                ind=torch.nonzero(z[i,:])
                curr_mask=z[i,:].repeat(num_ration,1)
                for j in range(len(ind)):
                  curr_mask[j,ind[j]]=0
                
                batch_curr={'input_ids':input_ids[i,:].repeat(num_ration,1),
                           'labels':labels[i].view(1,-1).repeat(num_ration,1),
                           'attention_mask': curr_mask}
                                
                output_curr = model.decoder(batch_curr['input_ids'],batch_curr['attention_mask'],
                                          batch_curr['labels'])
                logits_curr = output_curr['logits']

                ind_curr=torch.cat((1-labels[i].view(1,-1),labels[i].view(1,-1)),1).view(-1,2).repeat(num_ration,1)
                prob_sub=torch.sum(F.softmax(logits_curr,dim=-1)*ind_curr,1)
                
  
                labels_curr=torch.abs(1-batch_curr['labels'])
                celoss=ce(F.softmax(logits_curr,dim=-1),torch.squeeze(labels_curr,1))
                celoss = -args.mu * celoss/ l*args.grad_accumulation_steps
                celoss.backward()
                

                  
        
        if (args.global_step + 1) % args.grad_accumulation_steps == 0:
 
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
             if (args.global_step + 1) % args.print_every == 0:
                if args.method=='causal' or args.method=='vib':

                    model.eval()
                    logpns=cal_pns(model,batch,args)

                    model.train()
                    print(f'number of ration:{torch.mean(torch.sum(token_z,dim=1))}')

                train_acc = sum(total_correct) / len(total_correct)
                train_loss = total_loss / len(total_correct)

    
                elapsed = time() - t
      
                if args.method=='causal' or args.method=='vib':
                    log_dict = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'num_batches': num_batches,
                    'acc': train_acc,
                    #'logpns':logpns,
                    #'posi_prob':posi,
                    #'nega_prob':nega,
                    'loss': train_loss,
                    'elapsed': elapsed,
                    'global_step': args.global_step,
                    #'kl_loss': kl_loss,
                }
                else:
                  log_dict = {
                      'epoch': epoch,
                      'batch_idx': batch_idx,
                      'num_batches': num_batches,
                      'acc': train_acc,
                      'loss': train_loss,
                      'elapsed': elapsed,
                      'global_step': args.global_step,
                      #'kl_loss': kl_loss,
                  }
                  
                log_ga('train', args, log_dict)
                if args.wandb:
                    log_dict = {
                    'acc': train_acc,
                    #'logpns':logpns,
                    'loss': train_loss,
                    #'posi_prob':posi,
                    #'nega_prob':nega
                    #'kl_loss': kl_loss,
                }
                    wandb.log(log_dict)
                total_loss = 0
                #total_kl_loss = 0
                total_correct = []
                t = time()
            
             if (args.global_step + 1) % args.eval_interval == 0:
                
                
                dev_acc = evaluate('dev', model, dl, args)
                  
                print('dev_acc',dev_acc)
                print('best_score',args.best_score)
                if dev_acc > args.best_score:
                  # if > best score, latest=FALSE
                    args.best_score = dev_acc
                    if args.ckpt:
                        best_path=save_ckpt(args, model, optimizer, latest=False,best_path=best_path,seed=seed)
                if args.ckpt:
                    _ = save_ckpt(args, model, optimizer, latest=True,best_path=best_path,seed=seed)
                
                
                model.train()


        args.global_step += 1
          
  
    return best_path



def eval_after_train(model,dl,args):
    model.eval()
    total_correct = []
    total_loss=0
    pred=[]
    y=[]
    for batch_idx, batch in enumerate(dl['test']):
        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']
        prob=F.softmax(logits,dim=-1)[:,1].tolist()
        pred+=prob
        y+=batch['labels'].tolist()
        total_loss+=loss
        
    y=np.array(y)
    pred=np.array(pred)
    fpr,tpr,thre=metrics.roc_curve(y,pred,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    
    logpns=evaluate_for_pns('test', model, dl, args)
    
    log_dict = {
        'epoch': args.epoch,
        'batch_idx': batch_idx,
        'test_auc': auc,
        'test_loss': total_loss,
        'global_step': args.global_step,
    }
    #log_ga('test', args, log_dict)
    if args.wandb:
        log_dict = {
        'test_auc': auc,
        'test_loss': loss,
        'logpns': logpns

    }
        wandb.log(log_dict)


    

def evaluate(mode, model, dl, args):
  # return acc is for binary classification on dev
    model.eval()

    total_loss = 0
    total_correct = []
    y=[]
    pred=[]
    for batch_idx, batch in enumerate(dl[mode]):
        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']
        prob=F.softmax(logits,dim=-1)[:,1].tolist()
        pred+=prob
        y+=batch['labels'].tolist()
        total_loss+=loss
    
    y=np.array(y)
    pred=np.array(pred)
    fpr,tpr,thre=metrics.roc_curve(y,pred,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    
    logpns=evaluate_for_pns(mode, model, dl, args)
    log_dict = {
        'epoch': args.epoch,
        'batch_idx': batch_idx,
        'eval_auc': auc,
        'eval_loss': total_loss,
        'global_step': args.global_step,
    }
    log_ga(mode, args, log_dict)
    if args.wandb:
        log_dict = {
        'eval_auc': auc,
        'eval_loss': loss,
        'logpns':logpns

    }
        wandb.log(log_dict)
    return auc#,logpns


def evaluate_for_stats(mode, model, dl, args):
    model.eval()
    tps = []
    fps = []
    fns = []
    for batch_idx, batch in enumerate(dl[mode]):
        #batch is a dictionary with batch size 1

        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']
        
        batch_tps, batch_fps, batch_fns = gold_rationale_capture_rate(args, batch, output, dl.tokenizer)
        tps += batch_tps
        fps += batch_fps
        fns += batch_fns
    
    f1s = [2 * tp / (2 * tp + fp + fn+0.00000001) for tp, fp, fn in zip(tps, fps, fns)]
    precisions = [tp / (tp + fp +0.00000001) for tp, fp in zip(tps, fps)]
    recalls = [tp / (tp +fn+0.00000001) for tp, fn in zip(tps, fns)]
    macro_f1 = sum(f1s) / len(f1s)
    precision=sum(precisions)/len(precisions)
    recall=sum(recalls)/len(recalls)
    
    # calc micro f1
    micro_f1 = 2 * sum(tps) / (2 * sum(tps) + sum(fps) + sum(fns))
    return macro_f1, micro_f1, precision, recall


def evaluate_for_vis(mode, model, dl, args):
    model.eval()
    highlights=[]
    probs=[]
    ids=[]
    for batch_idx, batch in enumerate(dl[mode]):
         if len(ids)<100:
           if batch['labels']==1:
              if args.dataparallel:
                  output = model.module.forward_eval(batch)
              else:
                  output = model.forward_eval(batch)
              loss = output['loss']
              logits = output['logits']
              pred_highlight,prob=visualize_rationale_ga(args, batch, output, dl.tokenizer)

              highlights.append(pred_highlight)
              probs.append(prob)
              ids.append(batch_idx)
    
    selected_ids=ids
    selected_highlights=highlights
    
    text_path=f'{args.best_ckpt_path[:-3]}.txt'.replace('model',args.method+'-vis')
    print(text_path)
    if os.path.exists(text_path):
        os.remove(text_path)
 
    
    with open(text_path ,"a") as f:
      for i in range(len(selected_highlights)):
              print('[ID]:',selected_ids[i],file=f)
              print('[Prob]:',probs[i],file=f)
              print('[Pred rationale]:', selected_highlights[i],file=f)
              print('---',file=f)

                
def evaluate_for_pns(mode, model, dl, args):
    model.eval()
    counts_all=[]
    pns_all=[]
    posi_prob=0
    nega_prob=0
    count=0
    with torch.no_grad():
      for batch_idx, batch in enumerate(dl[mode]):
          if mode=='dev':
              if batch_idx<3:  
                pns=cal_pns(model,batch,args)
                pns_all.append(pns)
                #posi_prob+=posi
                #nega_prob+=nega
          else:
              if batch_idx<5:  
                  pns=cal_pns(model,batch,args)
                  if pns>0:
                    count+=1
                  pns_all.append(pns)
    pns_ave=sum(pns_all)/(count+0.0001)
    return pns_ave#,posi_prob,nega_prob

        

def main(args):
  
    with open('ga_code.pkl','rb') as f:
        code_to_ix=pickle.load(f)
    
    args.n_token=len(code_to_ix)
    
    training_start_time = time()
    args = args_factory(args)
    if args.method=='causalfc':
      print(f'args:model-{args.method}-mu={args.mu}-lr={args.lr}-beta-{args.beta}')
    elif args.method=='causal':
      print(f'args:model-{args.method}-flip={args.flip}-mu={args.mu}-k={args.k}-lr={args.lr}-beta-{args.beta}')
    elif args.method=='vib':
      print(f'args:model-{args.method}-lr={args.lr}-beta-{args.beta}')
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    save_args(args)


    ## save examples ## 
    # create dataloader and save it
    cached_features_file = os.path.join(args.cache_dir, 'cached_{}_{}_ml{}_bz{}'.format(
        args.dataset_name,
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
    
    ## set random seed
    
    seed = args.seed
    for i in range(args.nseeds):
      if args.wandb:
        name=f'{args.method}-{i}'
        if args.method=='causalRNP' or args.method=='causalVIB':
          para_config={'method':args.method,'mu':args.mu,'k':args.k,'lr':args.lr,'beta':args.beta,'max-epoch':args.num_epoch}
        elif args.method=='fc':
          para_config={'lr':args.lr,'max-epoch':args.num_epoch}
        else:
          para_config={'lr':args.lr,'beta':args.beta,'max-epoch':args.num_epoch}
      #wandb.init(project='rationale-robust',entity='causal',config=para_config)
        wandb.init(project='causal-rationale_ga', name=name,config=para_config)
      args.best_score = float('-inf')

      args.global_step= 0
      seed = seed + i
      #seed=0
      
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)


      print('model type:', args.model_type)
      model_class = get_model_class(args)
      model = model_class(args=args)
      
      
      if args.warm_epoch>0:
          args=build_vib_path(args,seed)
          model.load_state_dict(torch.load(args.vib_path))
          print('load yes')
      if args.dataparallel:
          model = nn.DataParallel(model)
      
      
      model = model.cuda()
      print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
      print(f'All params      : {sum(p.numel() for p in model.parameters())}')
      optimizer_class = get_optimizer_class(args)
      optimizer = optimizer_class(model.parameters(), lr=args.lr)

      fc_model=False
      best_path=''
      for epoch in range(args.num_epoch-args.warm_epoch):
          print(f'seed:{seed},epoch:{epoch}')
          best_path=train_epoch(seed,epoch, model, dl, optimizer, args,best_path)
      
      eval_after_train(model,dl,args)
      save_args(args)
      wandb.finish()
      print(f'seed{seed} run time elapsed: {(time() - training_start_time) / 60:.2f} min')

    print(f'Full run time elapsed: {(time() - training_start_time) / 60:.2f} min')
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc]")
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
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--inspect-gpu", action="store_true")
    parser.add_argument("--disable-cuda", action="store_true")

    # printing, logging, and checkpointing
    parser.add_argument("--print-every", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--disable-ckpt", action="store_true")

    # training
    parser.add_argument("--warm_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nseeds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)


    parser.add_argument("--desired_length", type=float, default=30.0)
    parser.add_argument("--k", type=int, default=3) # number of samples for PNS
    parser.add_argument("--mu", type=float, default=0.1) # weight for PNS
    parser.add_argument("--lambda1", type=float, default=0.0) #for concise loss
    parser.add_argument("--lambda2", type=float, default=0.0) #for continuity loss
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pi", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gamma2", type=float, default=1.0)
    parser.add_argument("--use-gold-rationale", action="store_true")
    parser.add_argument("--use-neg-rationale", action="store_true")
    parser.add_argument("--fix-input", type=str, default=None)
    

    args = parser.parse_args()
    args.method = args.run_name.split('_')[1]
    
    
    args.emsize = 192  # embedding dimension
    args.d_hid = 192  # dimension of the feedforward network model in nn.TransformerEncoder
    args.nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    args.nhead = 6  # number of heads in nn.MultiheadAttention
    args.dropout = 0.1 # dropout probability
    
    
    if args.debug:
        args.print_every = 2
        args.batch_size = 3
        args.eval_interval = 20
        args.num_epoch = 100
        args.dataparallel = False
        args.overwrite_cache = False
        args.max_length = 200

    main(args)