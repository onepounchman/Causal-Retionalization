import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli,LogitRelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence

from transformers import BertModel,BertConfig 

from cr.models.encoders import (
    GAEncoder,
)

from cr.models.decoders import (
    GADecoder,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GARNPSentimentTokenModel(nn.Module):
    def __init__(self, args):
        super(GARNPSentimentTokenModel, self).__init__()
        self.args = args
        self.encoder = GAEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(192, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = GADecoder(args)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        #print(len(labels))
        if 'rationales' in batch:
          rationales=batch['rationales']
        else:
          rationales= None

        token_reps = self.encoder(input_ids, attention_mask)
        batch_size, num_tokens, _ = token_reps.size()
        token_logits = self.rep_to_logit_layer(token_reps)
        token_logits = self.drop(token_logits)
        token_logits = token_logits.squeeze(2)
        token_z = self.sample_z(token_logits, attention_mask, mode)
        
        output = self.decoder(input_ids, token_z, labels)
        

        # concise_loss
        #concise_loss=torch.mean(F.relu(torch.sum(token_z,dim=0)-self.args.desired_length))
        #concise_loss=torch.mean(F.relu(torch.sum(token_z,dim=0)).float())
        #desired_length=torch.ceil(self.args.pi*torch.sum(input_ids!=0,dim=-1))
        concise_loss=torch.mean(F.relu(torch.sum(token_z,dim=-1)).float())       


        # continuity loss
        loss = output.loss+self.args.lambda1*concise_loss
        
        
        return {
            'loss': loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'token_z': token_z,
            'rationales': rationales
        }
    
    def sample_z(self, logits, mask, mode):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()
        elif mode == 'eval':
            z = torch.sigmoid(logits)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        if mode == 'eval':

            active_top_k = (lengths * self.args.pi).ceil().long()

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard
        return z
      
    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')

          
class GACausalSentimentTokenModel(nn.Module):
    def __init__(self, args):
        super(GACausalSentimentTokenModel, self).__init__()
        self.args = args
        self.encoder = GAEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(192, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = GADecoder(args)
    
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
    
        if self.args.method=='causalVIB':
          kl_loss = self.calc_kl_loss(token_logits, attention_mask)
          loss = output.loss+self.args.beta * kl_loss
        
        elif self.args.method=='causalRNP':
          desired_length=torch.ceil(self.args.pi*torch.sum(input_ids!=0,dim=-1))
          concise_loss=torch.mean(F.relu(torch.sum(token_z,dim=-1)-desired_length)) 
          loss = output.loss+self.args.beta*concise_loss
        
        if mode =='eval':
            change_loss=0
        else:
            change_loss=self.counterfact(input_ids,attention_mask,labels,token_z,prob_raw)

        loss=loss+self.args.mu*change_loss
        return {
            'loss': loss,
            'change_loss':change_loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'token_z': token_z,
            'rationales': rationales,
            'kl_loss': kl_loss,
            'token_logits':token_logits
            #'selections': selections
        }
    
    
    def counterfact(self, input_ids, attention_mask,labels,token_z,prob_raw):
        length=token_z.size()[1]
        num=len(input_ids)
        # size of token_z 64*120
        lengths=attention_mask.sum(dim=1)
        percent=self.args.k
        active_top_k = (lengths * percent/100).float().ceil().long()
        #for each data point
        for i in range(num):
              change_loss=0
              # find gumble samples larger than 0.8
              indices=torch.nonzero(token_z[i,:]>0.8)
              indexs=torch.squeeze(indices).tolist()
              num_ration=active_top_k[i].item()
              if indexs==[]:
                 continue
              elif type(indexs)==int:
                indexs=[indexs]
                samples=indexs
              # sample top 5% from those selection rationales
              elif len(indexs)<active_top_k[i].item():
                samples=indexs
              else:
                samples=(np.random.choice(indexs,size=num_ration,replace=False, p=[(1/len(indexs))*1 for _ in range(len(indexs))])).tolist()
              
              real_num=len(samples)
              curr_mask=token_z[i].expand(real_num,length)
              
              ## previous
              #inter_var=torch.where(curr_mask>0,1,0).to(device).float()
              inter_var=torch.where(curr_mask>0,1,0).to(device).float()
              
              #slightly faster
              for j in range(real_num):
                  inter_var[j,samples[j]]=0
              curr_mask=torch.mul(curr_mask,inter_var)
  
                  
              batch_curr={'input_ids':input_ids[i,:].expand(real_num,length),
                           'labels':labels[i].view(1,-1).expand(real_num,1),
                           'attention_mask': curr_mask}
              

              output_curr = self.decoder(batch_curr['input_ids'],batch_curr['attention_mask'],
                                          batch_curr['labels'])
          
              logits_curr = output_curr['logits']
              

              prob_sub=F.softmax(logits_curr,dim=-1)[:,labels[i].item()]
              z=prob_raw[i]-prob_sub+1
              z=torch.where(z>1,z,torch.tensor(1.0).to(device))
              logpns=torch.mean(torch.log(z))
              change_loss = change_loss-logpns 

        return change_loss
    
            
    def sample_z(self, logits, mask, mode):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()
            
          
        elif mode == 'eval' or mode=='train_pred' or mode=='double':
            z = torch.sigmoid(logits)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)
        

        if mode == 'eval' or mode=='train_pred' or mode=='double':
            
            if mode=='eval':
              pi=self.args.pi
              #pi=self.args.eval_pi
            elif mode=='double':
               pi=0.2
            else:
              pi=self.args.pi
              
            active_top_k = (lengths * pi).ceil().long()
            #active_top_k = (lengths * 0.05).ceil().long()
            #active_top_k = (lengths * 0.1).ceil().long()

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                  subidx = z_hard_inds[i, :active_top_k[i]]
                  z_hard[i, subidx] = 1.0
            z = z_hard
        return z
    
    def counter_selections_same(self,token_z,index):
        token_z[:,index]=torch.abs(1-token_z[:,index])  
        return token_z
      
    def counter_selections_diff(self,token_z,index):
        for i in range(len(index)):
          token_z[i,index[i]]=torch.abs(1-token_z[i,index[i]])  
          #token_z[i,index[i]]=torch.zeros((1)).cuda()
          return token_z
      
    def counter_selections_all(self,z):
        n=z.shape[0]
        p=z.shape[1]
        
        all_z=[z.clone() for j in range(p)]
        for j in range(p):
          for i in range(n):
            all_z[j][i,j]=torch.abs(1-z[i,j].detach())
          
        return all_z
      
      
    def forward_selections(self,input_ids,one_change,labels):
        output = self.decoder(input_ids,one_change, labels)
        return output.loss
    
    
    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')
    
    def forward_train_pred(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='train_pred')
          
    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)
        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)
        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / (mask.sum(dim=1))
        return kl_loss.mean()


      
class GAVIBSentimentTokenModel(nn.Module):
    def __init__(self, args):
        super(GAVIBSentimentTokenModel, self).__init__()
        self.args = args
        self.encoder = GAEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(192, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = GADecoder(args)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        #if 'rationales' in batch:
        #  rationales=batch['rationales']
        #else:
        #  rationales=N
        token_reps = self.encoder(input_ids, attention_mask)
        batch_size, num_tokens, _ = token_reps.size()
        token_logits = self.rep_to_logit_layer(token_reps)
        token_logits = self.drop(token_logits)
        token_logits = token_logits.squeeze(2)

        token_z = self.sample_z(token_logits, attention_mask, mode)

 
        output = self.decoder(input_ids, token_z, labels)
      

        
        kl_loss = self.calc_kl_loss(token_logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'kl_loss': kl_loss,
            'token_z': token_z,
          #  'rationales': rationales
          #  'prob': prob,
           # 'log_prob_token':log_prob_token
           # 'token_logits':token_logits
        }
    
                
    def sample_z(self, logits, mask, mode):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            tau=torch.tensor(self.args.tau)
            #relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            relaxed_bernoulli_dist = RelaxedBernoulli(tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()

            #print(z)
            #print(z.size())
            #sys.exit()
            
          
            
        elif mode == 'eval' or mode=='train_pred'or mode=='double':
            z = torch.sigmoid(logits)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)
        
        #z_hard=torch.where(z>0.5,1,0)
        #z=(z_hard-z).detach()+z
        

        
        if mode == 'eval' or mode=='train_pred' or mode=='double':
            if mode=='eval':
              pi=self.args.pi
              #pi=self.args.eval_pi
            elif mode=='train_pred':
              p
            elif mode=='double':
              pi=0.2
            else:
              pi=0.05
            active_top_k = (lengths * pi).ceil().long()
            #active_top_k = (lengths * pi).ceil().long()
            #active_top_k = (lengths * 0.1).ceil().long()

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                  subidx = z_hard_inds[i, :active_top_k[i]]
                  z_hard[i, subidx] = 1.0
            z = z_hard

        return z
      

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)
        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)
        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / (mask.sum(dim=1))
        return kl_loss.mean()

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')
    
    def forward_train_pred(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='train_pred')
    #def forward_eval_rationale(self, batch,token_z):


class GASLMSentimentTokenModel(nn.Module):
    def __init__(self, args):
        super(GASLMSentimentTokenModel, self).__init__()
        self.args = args
        self.encoder = GAEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(192, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = GADecoder(args)
    
    def forward(self, batch, mode='train'):
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
      
        
        # samples has size [batch x length] torch.Size([16, 300])

 
        inter = self.encoder(input_ids, token_z)[:,0]
        drop_output= self.drop(inter)
        output = self.decoder(input_ids, token_z, labels)
        

        
        kl_loss = self.calc_kl_loss(token_logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss

        
        return {
            'loss': loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'kl_loss': kl_loss,
            'token_z': token_z,
          #  'prob': prob,
           # 'log_prob_token':log_prob_token
           # 'token_logits':token_logits
        }

                
    
    def sample_z(self, logits, mask, mode):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            tau=torch.tensor(self.args.tau)
            #relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            relaxed_bernoulli_dist = RelaxedBernoulli(tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()
            
          
            
        elif mode == 'eval' or mode=='train_pred'or mode=='double':
            z = torch.sigmoid(logits)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)
        
        
        if mode == 'eval' or mode=='train_pred' or mode=='double':
            if mode=='eval':
              pi=self.args.pi
              #pi=self.args.eval_pi
            elif mode=='double':
              pi=0.2
            elif mode=='train_pred':
              pi=self.args.k/100
            else:
              pi=0.05
            active_top_k = (lengths * pi).ceil().long()
            #active_top_k = (lengths * pi).ceil().long()
            #active_top_k = (lengths * 0.1).ceil().long()

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                  subidx = z_hard_inds[i, :active_top_k[i]]
                  z_hard[i, subidx] = 1.0
            z = z_hard

        return z
      

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)
        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)
        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / (mask.sum(dim=1))
        return kl_loss.mean()

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')
    
    def forward_train_pred(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='train_pred')
    

class GAFullContextSentimentModel(nn.Module):
    def __init__(self, args):
        super(GAFullContextSentimentModel, self).__init__()
        self.args = args
        self.decoder = GADecoder(args)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.decoder(input_ids, attention_mask, labels)

        loss = output.loss
        return {
            'loss': loss,
            'logits': output.logits,
          'pred_loss': output.loss
        }

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')


