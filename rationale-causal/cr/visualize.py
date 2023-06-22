import torch
import time
from cr.config import Config
import sys
import torch.nn.functional as F
import seaborn as sns
import os
from transformers import BertTokenizerFast
config = Config()


def visualize_rationale(args, batch, batch_idx,output):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
        ids = batch['input_ids'].tolist()
        if args.dataset_name == 'beer':
          id_to_labelname = {v: k for k, v in config.BEER_LABEL.items()}
        if args.dataset_name == 'hotel':
          id_to_labelname = {v: k for k, v in config.HOTEL_LABEL.items()}
        pred_labels = torch.argmax(output['logits'], dim=1).tolist()
        labels = batch['labels'].tolist()
        gold_rationales = batch['rationales'].long().tolist()

        text_path = f'{args.best_ckpt_path[:-3]}.txt'.replace('model',args.method+'-vis')
        
        if batch_idx == 0:
            print(text_path)
            if os.path.exists(text_path):
                os.remove(text_path)
        with open(text_path ,"a") as f:
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
              f1=2 * tp / (2 * tp + fp + fn+0.00000001)
            
              highlight = [(token, r, gr) for token, r, gr in zip(tokens, rationale, gold_rationale) if token != '[PAD]']
              pred_highlight = [f'*{token}*' if r == 1 and token not in ('[CLS]', '[SEP]') else f'{token}' for token, r, gr in highlight]
              gold_highlight = [f'*{token}*' if gr == 1 else f'{token}' for token, r, gr in highlight]
              texts=[f'{token}' if token not in ('[CLS]', '[SEP]') else f'' for token, r, gr in highlight]
              print('[ID]:',batch_idx,file=f)
              print('[text]:',' '.join(texts),file=f)
              print('[Gold]:', id_to_labelname[label],file=f)
              print('[Pred]:', id_to_labelname[pred_label],file=f)
              print('[F1]:', f1,file=f)
              print('[Gold rationale]:', ' '.join(gold_highlight),file=f)
              print('[Pred rationale]:', ' '.join(pred_highlight),file=f)
              print('---',file=f)