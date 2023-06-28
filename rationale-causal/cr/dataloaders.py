import os
import csv
import re
import string
import random
import json
import pickle
import sys
from pprint import pprint

from tqdm import tqdm
import pandas as pd
from nltk.corpus import words
from nltk.tokenize import sent_tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from cr.config import Config

config = Config()
PRINTABLE = set(string.printable)

def get_special_token_map(encoder_type):
    if encoder_type.startswith('roberta'):
        special_token_map = {
            'bos_token': '<s>',
            'eos_token': '</s>',
            'sep_token': '</s>',
            'cls_token': '<s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>',
        }
    elif encoder_type.startswith('bert') or encoder_type.startswith('distilbert'):
        special_token_map = {
            'sep_token': '[SEP]',
            'cls_token': '[CLS]',
            'unk_token': '[UNK]',
            'pad_token': '[PAD]',
            'mask_token': '[MASK]',
        }
    return special_token_map


def extract_token_rationales(text, highlight_text, highlight_mode='pos'):
    # get char-level spans for rationales and words
    if highlight_mode == 'pos':
        left_sym = '*'
        right_sym = '*'
    elif highlight_mode == 'neg':
        left_sym = '['
        right_sym = ']'
    pattern_string = f'[\{left_sym}]+.*?[\{right_sym}]+'

    highlight_matches = [m for m in re.finditer(pattern_string, highlight_text)]
    highlight_spans = [m.span() for m in highlight_matches]
    if not highlight_spans:
        return []

    highlight_words = [highlight_text[s:e].strip(string.punctuation) for s, e in highlight_spans]
    pattern = '.*'.join([f'({word})' for word in highlight_words])
    matches = [m for m in re.finditer(pattern, text)]
    if len(matches) != 1:
        selector = [
            True if highlight_text[s - 1:s] == left_sym or highlight_text[e:e + 1] == right_sym else False 
            for s, e in [m.span() for m in re.finditer(pattern, highlight_text)]
        ]
        matches = [m for m, is_select in zip(matches, selector) if is_select == True]

    if not matches:
        return []
    match = matches[0]
    spans = [list(match.span(i)) for i in range(1, len(match.groups()) + 1)]
    assert ' '.join(highlight_words) == ' '.join([text[s:e] for s, e in spans])
    return spans


def get_rationale_vector_from_spans(offsets, span_set):
    spans = list(span_set)
    span_starts = set([s for s, e in spans])
    span_ends = set([e for s, e in spans])
    start_to_end_map = {s: e for s, e in spans}
    rationale = [0] * len(offsets)
    current_start = float('inf')
    current_end = float('-inf')

    for i, offset in enumerate(offsets):
        offset_start, offset_end = offset
        if offset_start in span_starts:
            rationale[i] = 1
            current_start = offset_start
            current_end = start_to_end_map[offset_start]
        elif offset_end in span_ends:
            rationale[i] = 1
            current_start = float('inf')
            current_end = float('-inf')
        elif offset_start > current_start and offset_end < current_end:
            rationale[i] = 1
    return rationale


def get_fixed_masks(tokenized, tokenizer):
    s_pos = []
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0].tolist())
    for pos, token in enumerate(tokens):
        if token in ('[CLS]', '[SEP]'):
            s_pos.append(pos)
        if token == '[PAD]':
            continue
    if len(s_pos) != 3:
        return None, None, None
    cls_pos, sep1_pos, sep2_pos = s_pos
    p_start_end = [cls_pos + 1, sep1_pos - 1]
    h_start_end = [sep1_pos + 1, sep2_pos - 1]
    return s_pos, p_start_end, h_start_end


class BaseDataLoader:
    def __init__(self, args):
        self.args = args
        self.tok_kwargs = config.TOK_KWARGS
        self.tok_kwargs['max_length'] = self.args.max_length
        if self.args.encoder_type.startswith('bert') or self.args.encoder_type.startswith('distilbert'):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=self.args.cache_dir)
        elif self.args.encoder_type.startswith('roberta'):
            self.tokenizer = RobertaTokenizerFast.from_pretrained(self.args.encoder_type, cache_dir=self.args.cache_dir)
        
        self.dataset_name_to_dataset_class = {
            'beer': SentimentDataset,
            'hotel': SentimentDataset
        }
        self._dataloaders = {}
        self.special_token_map = get_special_token_map(self.args.encoder_type)

    def _load_processed_data(self, mode):
        raise NotImplementedError

    def _build_dataloader(self, data, mode):
        dataset = self.dataset_name_to_dataset_class[self.args.dataset_name](
            self.args,
            data,
            self.tokenizer,
            self.tok_kwargs
        )
        collate_fn = dataset.collater
        batch_size = self.args.batch_size
        shuffle = True if mode == 'train' else False
        
        self._dataloaders[mode] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        print(f'[{mode}] dataloader built => {len(dataset)} examples')
    
    def build(self, mode):
        data = self._load_processed_data(mode)
        self._build_dataloader(data, mode)

    def build_all(self):
        for mode in ['train', 'dev', 'test']:
            self.build(mode)
    
    def __getitem__(self, mode):
        return self._dataloaders[mode]

    @property
    def train(self):
        return self._dataloaders['train']

    @property
    def dev(self):
        return self._dataloaders['dev']
    
    @property
    def test(self):
        return self._dataloaders['test']


class BaseDataset(Dataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.tok_kwargs = tok_kwargs

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    @property
    def num_batches(self):
        return len(self.data) // self.args.batch_size


class SentimentDataLoader(BaseDataLoader):
    def __init__(self, args):
        super(SentimentDataLoader, self).__init__(args)
        if args.dataset_split == 'all':
            self.build_all()
        else:
            self.build(args.dataset_split)

    def _load_raw_data(self, mode):
            datapoints = []
            aspect = self.args.aspect
            scale = self.args.scale
            print('aspect:',aspect)
            print('mode:',mode)
            print('scale:',scale)
            
            if self.args.dataset_name == 'hotel':
                if mode in ('train', 'dev'):
                    path = config.DATA_DIR / f'sentiment/data/oracle/hotel_Location.{mode}'
                elif mode == 'test':
                    path = config.DATA_DIR / f'sentiment/data/target/hotel_Location.train'
                    
            if self.args.dataset_name == 'beer':
                if scale =='small':
                        if mode in ('train', 'dev'):
                            path = config.DATA_DIR / f'sentiment/data/source/beer_{aspect}.{mode}_120'
                        elif mode == 'test':
                            path = config.DATA_DIR / f'sentiment/data/target/beer_{aspect}.train'
                        else:
                            raise ValueError('mode not supported.')
                if scale =='noise':
                        if mode in ('train', 'dev'):
                            path = config.DATA_DIR / f'sentiment/data/source/beer_{aspect}.{mode}_noise'
                        elif mode == 'test':
                            path = config.DATA_DIR / f'sentiment/data/target/beer_{aspect}.train'
                        else:
                            raise ValueError('mode not supported.')
            
            df = pd.read_csv(path, delimiter='\t')
            for index, row in df.iterrows():

                label = row['label']

                if label >= 0.6:
                    label = 1  # pos
                elif label <= 0.4:
                    label = 0  # neg
                else:
                    continue
                text = row['text']
                if 'rationale' in row:
                    rationale = [int(r) for r in row['rationale'].split()]
                else:
                    rationale = [-1] * len(row['text'].split())


                datapoints.append({
                    'label': label,
                    'text': text,
                    'rationale': rationale,
                    'env': 'None'
                })
            if self.args.debug:
              datapoints = datapoints[:200]
            return datapoints

    def _load_processed_data(self, mode):
        processed_datapoints = []
        datapoints = self._load_raw_data(mode)
        for datapoint in tqdm(datapoints, total=len(datapoints)):
            
            label = datapoint['label']
            input_tokens = ['[CLS]'] + datapoint['text'].split()
            rationale = datapoint['rationale']
            input_ids = []
            attention_mask = []
            rationale_ = []
            for input_token, r in zip(input_tokens, rationale):
                tokenized = self.tokenizer.encode_plus(input_token, add_special_tokens=False)
                input_ids += tokenized['input_ids']
                attention_mask += tokenized['attention_mask']
                rationale_ += [r] * len(tokenized['input_ids'])

                
            if  len(input_ids) >= self.args.max_length:
                input_ids = input_ids[:self.args.max_length - 1] + [102]
                attention_mask = attention_mask[:self.args.max_length - 1] + [1]
                rationale = rationale_[:self.args.max_length - 1] + [0]
            else:
                input_ids = input_ids + [102] #102 is [SEP]
                attention_mask = attention_mask + [1]
                rationale = rationale_ + [0]
                
            input_ids = self.pad(input_ids)
            attention_mask = self.pad(attention_mask)
            rationale = self.pad(rationale)

            assert len(input_ids) == self.args.max_length

            processed_datapoints.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'rationale': rationale,
                'env': 0
            })
            
        return processed_datapoints

    def pad(self, seq):
        return seq + (self.args.max_length - len(seq)) * [0]


class SentimentDataset(BaseDataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        super(SentimentDataset, self).__init__(args, data, tokenizer, tok_kwargs)

    def collater(self, batch):
        device = 'cuda' if self.args.use_cuda else 'cpu'
        return {
                'input_ids': torch.tensor([datapoint['input_ids'] for datapoint in batch]).long().to(device),
                'attention_mask': torch.tensor([datapoint['attention_mask'] for datapoint in batch]).long().to(device),
                'labels': torch.tensor([datapoint['label'] for datapoint in batch]).long().to(device),
                'rationales': torch.tensor([datapoint['rationale'] for datapoint in batch]).long().to(device),
            }
      
    


