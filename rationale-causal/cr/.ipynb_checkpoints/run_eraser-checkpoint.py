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

import wandb

from rrtl.utils import (
    get_model_class,
    get_optimizer_class,
    get_dataloader_class,
    args_factory,
    save_args,
    save_ckpt,
    load_ckpt,
    build_vib_path
)
from rrtl.logging_utils import log
from rrtl.visualize import visualize_rationale,visualize_prob
from rrtl.stats import gold_rationale_capture_rate,cal_pns,first_capture
from rrtl.config import Config
import warnings
config = Config()


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
      # check backward
      #with torch.autograd.set_detect_anomaly(True):
        

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
                

                  
            elif args.flip=='random':
              k=args.k
              
              for i in range(l):
                    
                    indices=torch.nonzero(output['token_z'][i]>0.5)
                    
                    #change to a list (.tolist() return a list or a number)
                    indexs=torch.squeeze(indices).tolist()
                    
                    if indexs==[]:
                      indexs=[0,0,0,0,0]
                    if type(indexs)==int:
                      indexs=[indexs]
                    if len(indexs)<k:
                      indexs=indexs+[indexs[0]]*(k-len(indexs))
                               
                    random.shuffle(indexs)  
                    all_indexs.append(indexs[0:k])

              orga_indexs=[all_indexs[i][j]  for j in range(k)for i in range(len(all_indexs))]
              
              start=0
              end=l
              for i in range(k):
                
                  output = model.forward_change(batch,'train',orga_indexs[start:end])
                  logits = output['logits'] 
                  prob_sub=torch.sum(F.softmax(logits,dim=-1)*ind,1)
                  #0 first way of doing this
                  logpns=torch.mean(torch.log(prob_raw-prob_sub+1))
                  logpns = -args.mu * logpns/(k* args.grad_accumulation_steps)
                  logpns.backward()
                  start+=l
                  end+=l
                  
        
        
        if (args.global_step + 1) % args.grad_accumulation_steps == 0:
 
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
             if (args.global_step + 1) % args.print_every == 0:
                if args.method=='causal' or args.method=='vib':
                    a=1
                    #model.eval()
                    #logpns,posi,nega=cal_pns(model,batch)

                    #model.train()
                    #print(f'number of ration:{torch.mean(torch.sum(token_z,dim=1))}')
                if args.method=='causalfc':
                  kl_loss=output['kl_loss']
                  pred_loss=output['pred_loss']
                  print(f'log_prob_token:{torch.sum(log_prob_token)}')
                  print(f'log_prob_token_loss:{torch.mean(log_prob_token)}')
                  print(f'pred_loss:{pred_loss}')
                  print(f'kl_loss:{kl_loss}')
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
                  
                log('train', args, log_dict)
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
                #if args.method=='causal'or args.method=='vib':
                #  print('dev_logpns',dev_logpns)
                #print('best_score',args.best_score)
                if dev_acc > args.best_score:
                  # if > best score, latest=FALSE
                    args.best_score = dev_acc
                    if args.ckpt:
                        best_path=save_ckpt(args, model, optimizer, latest=False,best_path=best_path,seed=seed)
                if args.ckpt:
                    _ = save_ckpt(args, model, optimizer, latest=True,best_path=best_path,seed=seed)
                
                
                model.train()


        args.global_step += 1
        

    #if args.method=='vib' and epoch==4:
    #      epoch_path ='vib-seed={seed}-save_epoch={curr_epoch}.pt'.format(
    #              seed=seed,
    #              curr_epoch=epoch,
    #          )
    #      epoch_path= os.path.join(args.ckpt_dir,epoch_path)
    #      states = model.state_dict() 
    #      torch.save(states, epoch_path)
    #      sys.exit()
          
  
    return best_path



def evaluate(mode, model, dl, args):
  # return acc is for binary classification on dev
    model.eval()

    total_loss = 0
    total_correct = []
    for batch_idx, batch in enumerate(dl[mode]):
        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']
        #token_z = output['token_z']


        if args.dataparallel:
            total_loss += loss.sum().item()
        else:
            total_loss += loss.item()
        total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
    acc = sum(total_correct) / len(total_correct)
    loss = total_loss / len(total_correct)
    
    if args.method=='causal' or args.method=='vib':

      #logpns,posi,nega=evaluate_for_pns(mode, model, dl, args)
      log_dict = {
          'epoch': args.epoch,
          'batch_idx': batch_idx,
         # 'logpns':logpns,
        #  'posi_prob':posi,
        #  'nega_prob':nega,
          'eval_acc': acc,
          'eval_loss': loss,
          'global_step': args.global_step,
      }

      log(mode, args, log_dict)
      if args.wandb:
          log_dict = {
        #  'eval_logpns':logpns,
          'eval_acc': acc,
          'eval_loss': loss,
         # 'eval_posi_prob':posi,
        #  'eval_nega_prob':nega,

      }
          wandb.log(log_dict)
    else:
      log_dict = {
          'epoch': args.epoch,
          'batch_idx': batch_idx,
          'eval_acc': acc,
          'eval_loss': loss,
          'global_step': args.global_step,
      }

      log(mode, args, log_dict)
      if args.wandb:
          log_dict = {
          'eval_acc': acc,
          'eval_loss': loss,

      }
          wandb.log(log_dict)
    return acc


def evaluate_for_stats(mode, model, dl, args):
    model.eval()
    tps = []
    fps = []
    fns = []
    caps = []
    cap_rate=0
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

        cap=first_capture(args, batch, output, dl.tokenizer)
        caps.append(cap)
        
    
    # f1 score division by 0 means classifier is useless
    # calc macro f1
    f1s = [2 * tp / (2 * tp + fp + fn+0.00000001) for tp, fp, fn in zip(tps, fps, fns)]
    precisions = [tp / (tp + fp +0.00000001) for tp, fp in zip(tps, fps)]
    recalls = [tp / (tp +fn+0.00000001) for tp, fn in zip(tps, fns)]
    # in some cases there are tp fp fn all 0, which makes division by 0 error
    #f1s = [print(2 * tp + fp + fn) for tp, fp, fn in list(zip(tps, fps, fns))]
    #print(list(zip(tps, fps)))
    macro_f1 = sum(f1s) / len(f1s)
    precision=sum(precisions)/len(precisions)
    recall=sum(recalls)/len(recalls)
    
    # calc micro f1
    micro_f1 = 2 * sum(tps) / (2 * sum(tps) + sum(fps) + sum(fns))
    
    cap_rate=sum(caps)/len(caps)
    return macro_f1, micro_f1, precision, recall,cap_rate


def evaluate_for_vis(mode, model, dl, args):
    model.eval()
    for batch_idx, batch in enumerate(dl[mode]):
        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']
        visualize_rationale(args, batch, output, dl.tokenizer)
        
#dsign for validation
def evaluate_for_pns(mode, model, dl, args):
    model.eval()
    counts_all=[]
    pns_all=[]
    posi_prob=0
    nega_prob=0
    with torch.no_grad():
      for batch_idx, batch in enumerate(dl[mode]):
          if mode=='dev':
              if batch_idx<3:  
                pns=cal_pns(model,batch)
                pns_all.append(pns)
                #posi_prob+=posi
                #nega_prob+=nega
          else:
              pns=cal_pns(model,batch)
              pns_all.append(pns)
    pns_ave=mean(pns_all)
    return pns_ave#,posi_prob,nega_prob

        

def main(args):
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
    

    ## set random seed
    seed = args.seed
    for i in range(args.nseeds):
      if args.wandb:
        name=f'{args.scale}-{args.aspect}-{args.method}-{i}'
        if args.method=='causal':   
          para_config={'flip':args.flip,'warm':args.warm_epoch,'mu':args.mu,'k':args.k,'lr':args.lr,'beta':args.beta,'max-epoch':args.num_epoch}
        if args.method=='vib':
          para_config={'lr':args.lr,'beta':args.beta,'max-epoch':args.num_epoch,'warm':args.warm_epoch}
      #wandb.init(project='rationale-robust',entity='causal',config=para_config)
        wandb.init(project='causal-rationale_full', name=name,config=para_config)
        
        
      args.best_score = float('-inf')
      args.global_step= 0
      
      #seed = seed + i +1
      #seed = seed + i 
      #seed=0
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)



      model_class = get_model_class(args)
      model = model_class(args=args)
      
      
      if (args.method=='causal' or args.method=='vib') and args.warm_epoch>0:
          args=build_vib_path(args,seed)
          print('vib_path:',args.vib_path)
 
          model.load_state_dict(torch.load(args.vib_path))
          #model.load_state_dict(torch.load('/home/cdsw/internship_project/code/rationale-robust/experiments/beer/Look/vib/small/lr=5e-05-beta=1.0-pi=0.1-epoch=20-warm=5/model-seed=42-step=2599-acc=95.77.pt')['states'])
          #model.load_state_dict(torch.load('/home/cdsw/internship_project/code/rationale-robust/experiments/beer/Look/causal/small/lr=5e-05-flip=all-mu=0.1-k=1-beta-1.0-pi=0.1-epoch=20-warm=5/model-seed=42-step=3399-acc=96.03.pt')['states'])
          #model.load_state_dict(torch.load('/home/cdsw/internship_project/code/rationale-robust/experiments/beer/Look/causal/small/lr=5e-05-flip=all-mu=0.1-k=1-beta-1.0-pi=0.1-epoch=20-warm=5/model-seed=42-step=3399-acc=96.03.pt')['states'])
          print('load yes')
          

        
      model = model.cuda()
      if args.dataparallel:
          model = nn.DataParallel(model)
      
      # calculate and vis whether model ouput sharp prob
      #visualize_prob(model,dl,'dev')
      #sys.exit()
    
      print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
      print(f'All params      : {sum(p.numel() for p in model.parameters())}')
      optimizer_class = get_optimizer_class(args)
      optimizer = optimizer_class(model.parameters(), lr=args.lr)

      #fc_model=False
      #if args.method=='causalfc':
      #    load_path='/home/cdsw/internship_project/code/rationale-robust/experiments/beer/Look/fc/small/model-lr=5e-05=beta=0.0-step=2599-acc=97.66.pt'
      #    fc_model,_,_ = load_ckpt(load_path)
      #    fc_model.cuda()

      best_path=''
      for epoch in range(args.num_epoch-args.warm_epoch):
          print(f'seed:{seed},epoch:{epoch}')
          best_path=train_epoch(seed,epoch, model, dl, optimizer, args,best_path)

      save_args(args)
      wandb.finish()
      print(f'seed{seed} run time elapsed: {(time() - training_start_time) / 60:.2f} min')
    print(f'Full run time elapsed: {(time() - training_start_time) / 60:.2f} min')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--scale", type=str, default="normal", help="[small |normal]")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc]")
    parser.add_argument("--aspect", type=str, required=True, help="Look, Aroma,Palate for beer;Cleanliness,Location,Service for hotel")
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
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)

    # VIB model
    parser.add_argument("--flip", type=str, default="same", help="[random|all")
    parser.add_argument("--desired_length", type=float, default=30.0)
    parser.add_argument("--k", type=int, default=3) # number of samples for PNS
    parser.add_argument("--mu", type=float, default=0.1) # weight for PNS
    #parser.add_argument("--alpha", type=float, default=0.0) #for concise loss
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
    
    # SPECTRA model
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--budget_ratio", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--solver_iter", type=int, default=100)

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