import os
import sys
import argparse
from pathlib import Path
from time import time
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from cr.config import Config
from cr.logging_utils import log,log_cap
from cr.visualize import visualize_rationale
import wandb
config = Config()

def eval_after_train(model,dl,args):
    model.eval()
    total_correct = []
    for batch_idx, batch in enumerate(dl['test']):
        output = model.forward_eval(batch)
        logits = output['logits']
        if args.visualize:
            print('yes')
            visualize_rationale(args, batch, batch_idx,output)
        total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
    acc = sum(total_correct) / len(total_correct)
    macro_f1, micro_f1, precision, recall=evaluate_for_stats('test', model, dl, args)
    if args.scale=='noise':
          cap_rate=evaluate_for_cap('dev',model,dl,args)
        
          log_dict = {
          'test_acc': acc,
          'test_precision': precision,
          'test_recall': recall,
          'test_F1':macro_f1,
          'cap_rate':cap_rate  
      }
          log_cap(args, log_dict)
    
    else:
          log_dict = {
            'test_acc': acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_F1':macro_f1
        }

          log('test', args, log_dict)
    
    if args.wandb:
      wandb.log(log_dict)
    

    
    
    
def evaluate(mode, model, dl, args):
    model.eval()

    total_loss = 0
    total_correct = []
    for batch_idx, batch in enumerate(dl[mode]):

        output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']

        total_loss += loss.item()
        total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
    acc = sum(total_correct) / len(total_correct)
    loss = total_loss / len(total_correct)
    
    macro_f1, micro_f1, precision, recall=evaluate_for_stats('test', model, dl, args)
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
          'test_precision': precision,
          'test_recall': recall,
          'test_F1':macro_f1
      }
          wandb.log(log_dict)

    return acc


def evaluate_for_stats(mode, model, dl, args):
    model.eval()
    tps = []
    fps = []
    fns = []
    for batch_idx, batch in enumerate(dl[mode]):

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
    micro_f1 = 2 * sum(tps) / (2 * sum(tps) + sum(fps) + sum(fns))
    
    return macro_f1, micro_f1, precision, recall
  


def gold_rationale_capture_rate(args, batch, output, tokenizer):
        tps = []
        fps = []
        fns = []
        
        ids = batch['input_ids'].tolist()
        id_to_labelname = {v: k for k, v in config.BEER_LABEL.items()}
        pred_labels = torch.argmax(output['logits'], dim=1).tolist()
        labels = batch['labels'].tolist()
        gold_rationales = batch['rationales'].long().tolist()
        
        for ids, rationale, gold_rationale, label, pred_label in zip(batch['input_ids'].tolist(), output['token_z'], gold_rationales, labels, pred_labels):
            tokens = tokenizer.convert_ids_to_tokens(ids)
            rationale = rationale.long().tolist()
            pred = set()
            gold = set()
            for token_pos, (token, r, gr) in enumerate(zip(tokens, rationale, gold_rationale)):
                if token in ('[CLS]', '[SEP]', '[PAD]'):
                    continue
                if r == 1:
                    pred.add(token_pos)
                if gr == 1:
                    gold.add(token_pos)

            tp = len(gold & pred)
            fp = len(pred) - tp
            fn = len(gold) - tp
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
        return tps, fps, fns

def evaluate_for_pns(mode, model, dl, args):
    model.eval()
    counts_all=[]
    pns_all=[]
    posi_prob=0
    nega_prob=0
    count=0
    with torch.no_grad():
      for batch_idx, batch in enumerate(dl[mode]):
          pns=cal_pns(model,batch,args)
          count+=1
          pns_all.append(pns)
    pns_ave=sum(pns_all)/count
    return pns_ave

        
def cal_pns(model,batch,args):

    output = model.forward_eval(batch)
    input_ids = batch['input_ids']
    labels = batch['labels']
    index=torch.cat((1-batch['labels'].view(-1,1),batch['labels'].view(-1,1)),1).view(-1,2)
    rationale=output['token_z']
    loc=torch.nonzero(rationale)
    counts=loc[:,0].unique(return_counts=True)[1].tolist()
          
    T=1
    loc=loc.tolist()
    logits=output['logits']/T
    prob_raw=torch.sum(F.softmax(logits,dim=-1)*index,1)
    
    pns = 0
    count = 0
    for idx in loc:
        ith = idx[0]
        p = idx[1]
        rationale_copy = rationale.clone()
        rationale_copy[ith,p] = 0
        logits = (model.decoder(input_ids[ith].view(1,-1), rationale_copy[ith,:].view(1,-1), labels[ith]))['logits']

        prob_sub = torch.sum(F.softmax(logits/T,dim=-1)*index[ith],1)
        z = prob_raw[ith]-prob_sub
        if z>0:
            count+=1
        z = torch.where(z>0,z,torch.zeros_like(z))
        pns += z.item()
    pns = pns/(count+0.000001)
    return pns


def evaluate_for_cap(mode,model,dl,args):
    num_val_aroma = 2928
    num_val_palate = 2294
    model.eval()
    caps = []
    for batch_idx, batch in enumerate(dl[mode]):
        output = model.forward_eval(batch)
        cap = capture_first(args, batch, output, dl.tokenizer)
        caps.append(cap)
    if args.aspect == 'Aroma':
        cap_rate = sum(caps)/num_val_aroma
    elif args.aspect == 'Palate':
        cap_rate = sum(caps)/num_val_palate
    return cap_rate

def capture_first(args, batch, output, tokenizer):
    cap = 0
    ids = batch['input_ids'].tolist()
    rationales = output['token_z'].long().tolist()
    for i in range(len(ids)):
            i_ids = ids[i]
            i_rationales = rationales[i]
            i_tokens = tokenizer.convert_ids_to_tokens(i_ids)
            if i_rationales[1] == 1:
                    cap += 1                               
    return cap

