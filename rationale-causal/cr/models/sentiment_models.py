import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli,LogitRelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
import numpy as np
from transformers import BertModel

from cr.models.encoders import TokenLevelEncoder

from cr.models.decoders import Decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CausalSentimentTokenModel(nn.Module):
    def __init__(self, args):
        super(CausalSentimentTokenModel, self).__init__()
        self.args = args
        self.encoder = TokenLevelEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args)
    
    def forward(self, batch,mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        if 'rationales' in batch:
            rationales=batch['rationales']
        else:
            rationales=None

        token_reps = self.encoder(input_ids, attention_mask)
        batch_size, num_tokens, _ = token_reps.size()
        token_logits = self.rep_to_logit_layer(token_reps)
        token_logits = self.drop(token_logits)
        token_logits = token_logits.squeeze(2)

        token_z = self.sample_z(token_logits, attention_mask, mode)
        output = self.decoder(input_ids, token_z, labels)

        logits = output['logits']
        ind=torch.cat((1-batch['labels'].view(-1,1),batch['labels'].view(-1,1)),1).view(-1,2)
        prob_raw=torch.sum(F.softmax(logits,dim=-1)*ind,1).clone().detach()
      
        kl_loss = self.calc_kl_loss(token_logits, attention_mask)
        loss = output.loss+self.args.beta * kl_loss
        
        if mode!='eval':
            change_loss=self.counterfact(input_ids,attention_mask,labels,token_z,prob_raw)
        else:
            change_loss=0

        loss = loss+self.args.mu*change_loss
        return {
            'loss': loss,
            'change_loss':change_loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'token_z': token_z,
            'rationales': rationales,
            'kl_loss': kl_loss,
            'token_logits':token_logits
        }
    
    
    def counterfact(self, input_ids, attention_mask,labels,token_z,prob_raw):
        length = token_z.size()[1]
        num = len(input_ids)
        lengths = attention_mask.sum(dim=1)
        percent = self.args.k
        active_top_k = (lengths * percent/100).float().ceil().long()
        for i in range(num):
              change_loss = 0
              # find gumble samples larger than 0.8
              indices = torch.nonzero(token_z[i,:]>0.8)
              indexs = torch.squeeze(indices).tolist()
              num_ration = active_top_k[i].item()
              if indexs == []:
                 continue
              elif type(indexs) == int:
                indexs = [indexs]
                samples = indexs
              elif len(indexs) < active_top_k[i].item():
                samples = indexs
              else:
                samples = (np.random.choice(indexs,size=num_ration,replace=False, p=[(1/len(indexs))*1 for _ in range(len(indexs))])).tolist()
              
              real_num = len(samples)
              curr_mask = token_z[i].expand(real_num,length)
              inter_var = torch.where(curr_mask>0,1,0).to(device).float()
              
              for j in range(real_num):
                  inter_var[j,samples[j]]=0
              curr_mask = torch.mul(curr_mask,inter_var)
  
                  
              batch_curr = {'input_ids':input_ids[i,:].expand(real_num,length),
                           'labels':labels[i].view(1,-1).expand(real_num,1),
                           'attention_mask': curr_mask}
              

              output_curr = self.decoder(batch_curr['input_ids'],batch_curr['attention_mask'],
                                          batch_curr['labels'])
          
              logits_curr = output_curr['logits']
              prob_sub = F.softmax(logits_curr,dim=-1)[:,labels[i].item()]
              z = prob_raw[i]-prob_sub+1
              z = torch.where(z>1,z,torch.tensor(1.0).to(device))
              logpns = torch.mean(torch.log(z))
              change_loss = change_loss-logpns 

        return change_loss
    
            
    def sample_z(self, logits, mask, mode):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()
            
          
        elif mode == 'eval':
            z = torch.sigmoid(logits)
        z = z.masked_fill(~mask.bool(), 0.0)
        

        if mode == 'eval':
            pi = self.args.pi 
            active_top_k = (lengths * pi).ceil().long()

            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard
        return z
      
    def forward_selections(self,input_ids,one_change,labels):
        output = self.decoder(input_ids,one_change, labels)
        return output.loss
    
    
    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')
          
    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)
        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)
        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / (mask.sum(dim=1))
        return kl_loss.mean()


