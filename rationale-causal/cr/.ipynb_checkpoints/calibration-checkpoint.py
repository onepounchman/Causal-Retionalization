import torch
from torch import nn, optim
from torch.nn import functional as F
import argparse
import sys
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from cr.run_eraser import (
    evaluate,
    evaluate_for_stats,
    evaluate_for_vis,
    evaluate_for_pns
)
from cr.utils import (
    load_ckpt,
    update_args,
    get_dataloader_class,
)

from rrtl.visualize import visualize_rationale,visualize_prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, batch):
        output = self.model.forward_eval(batch)
        logits = output['logits']
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, dl):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl['dev']):
            #for input, label in valid_loader:
                #input = input.cuda()
                output = self.model.forward_eval(batch)
                logits = output['logits']
                label = batch['labels']
                logits_list.append(logits)
                labels_list.append(label)
                ind=torch.cat((1-batch['labels'].view(-1,1),batch['labels'].view(-1,1)),1).view(-1,2)
                prob_raw=torch.sum(F.softmax(logits,dim=-1)*ind,1).clone().detach()
                #print(prob_raw)
                #sys.exit()
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

      
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece



# make evaluation batch size become 1
def eval_args_factory(ckpt_args, args):
    """
    Adjust args temporarily for running evaluation.
    """
    # here we need to make evaluation of rationale on test
    ckpt_args = update_args(ckpt_args)
    ckpt_args.no_shuffle = True
    ckpt_args.dataparallel = False
    ckpt_args.batch_size = 64
    ckpt_args.global_step = -1
    ckpt_args.epoch = -1
    ckpt_args.wandb = False
    ckpt_args.use_gold_rationale = False
    ckpt_args.use_neg_rationale = False
    ckpt_args.dataset_name = args.dataset_name
    ckpt_args.dataset_split = args.dataset_split
    ckpt_args.debug = args.debug
    #ckpt_args.attack_path = args.attack_path
    return ckpt_args


def load_for_temp(args,batch_size):
    model, ckpt_args, _ = load_ckpt(args.load_path)
    ckpt_args = eval_args_factory(ckpt_args, args)
    model = model.cuda()
    ckpt_args.batch_size=batch_size
    ckpt_args.pi=args.pi
    dataloader_class = get_dataloader_class(ckpt_args)
    dl = dataloader_class(ckpt_args)
    return model, dl, ckpt_args


def run_temp(args):
    
    
    model, dl_test, ckpt_args = load_for_temp(args,1)
    #orig_model = model # create an uncalibrated model somehow
    #valid_loader = dl# Create a DataLoader from the SAME VALIDATION SET used to train orig_model
    #scaled_model = ModelWithTemperature(orig_model)
    #scaled_model.set_temperature(valid_loader)
    

    #_, dl_test, _ = load_for_temp(args,1)
    #evaluate_for_pns('test', model, dl_test, args,0.1)
    evaluate_for_pns('test', model, dl_test, args,0.5)
    evaluate_for_pns('test', model, dl_test, args,1)
    evaluate_for_pns('test', model, dl_test, args,1.5)
    #evaluate_for_pns('test', model, dl_test, args,scaled_model.temperature.item())

def evaluate_for_pns(mode, model, dl, args,temp):
    model.eval()
    counts_all=[]
    pns_all=[]
    with torch.no_grad():
      for batch_idx, batch in enumerate(dl[mode]):         
              pns,counts=cal_pns(model,batch,temp)
              pns_all.append(pns)
              counts_all.append(counts)
    #print(pns_all)
    #print(sum(counts_all))
    pns_ave=sum(pns_all)/200
    print(f'temperature={temp}  pns={pns_ave:.3f}')
    return pns_ave

def vis_for_prob(args):
    model, dl_test, ckpt_args = load_for_temp(args,200)
    args.dataset_split='test'
    model.eval()
    prob_raw=None
    temp=args.temp
    with torch.no_grad():
            for batch_idx, batch in enumerate(dl_test['test']):   
                      output=model.forward_eval(batch)
                      input_ids = batch['input_ids']
                      labels = batch['labels']
                      index=torch.cat((1-batch['labels'].view(-1,1),batch['labels'].view(-1,1)),1).view(-1,2)
                      rationale=output['token_z']
                      logits=output['logits']/temp
                      prob_raw=torch.sum(F.softmax(logits,dim=-1)*index,1)
    prob_raw_np=prob_raw.cpu().detach().numpy()
    bins=np.linspace(0,1,20)
    plt.hist(prob_raw_np,bins,alpha=0.5,label=f'{args.method},T={temp}')
    plt.legend(loc='upper right')
    plt.savefig('prob_vib.pdf')
    return prob_raw
  
  
def cal_pns(model,batch,temp):
          # here index is 64*2 with 0 and 1 indicates label
          # loc is a list contain all rationales, first indiacte
          # ith sample, j indicate which dimension
          output=model.forward_eval(batch)
          input_ids = batch['input_ids']
          labels = batch['labels']
          index=torch.cat((1-batch['labels'].view(-1,1),batch['labels'].view(-1,1)),1).view(-1,2)
          rationale=output['token_z']
          loc=torch.nonzero(rationale)
          counts=loc[:,0].unique(return_counts=True)[1].tolist()
          
          loc=loc.tolist()
          logits=output['logits']/temp
          prob_raw=torch.sum(F.softmax(logits,dim=-1)*index,1)
          
          ##print(prob_raw)
          pns=[]
          probs=[]
          count=0
          diff=[]
          for idx in loc:
              ith=idx[0]
              p=idx[1]
              rationale_copy=rationale.clone()
              rationale_copy[ith,p]=0
              if args.method!= 'SLM':
                  logits=(model.decoder(input_ids[ith].view(1,-1), rationale_copy[ith,:].view(1,-1), labels[ith]))['logits']
              else:
                after_drop=model.drop(model.encoder(input_ids[ith].view(1,-1), rationale_copy[ith,:].view(1,-1))[:,0,:])
                logits=model.classifier(after_drop)
              logits=logits/temp
              prob_sub=torch.sum(F.softmax(logits,dim=-1)*index[ith],1)
              diff=prob_raw[ith]-prob_sub
              if diff>0:
               count+=1
               pns.append(torch.sum(torch.where(diff>0,diff,torch.tensor(0.0).to(device))).item())
          #print(pns)
          pns_sum=sum(pns)
          if count!=0:
            pns_ave=pns_sum/count
          else:
            pns_ave=0
          return pns_ave,count



if __name__ == '__main__':
    """
    python -m rrtl.run_eraser_eval --dataset-name beer --load-path /path/to/your/checkpoint/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset to evaluate on. [beer | hotel | ...]")
    parser.add_argument("--dataset-split", type=str, default="dev", help="[train | dev | test]")
    parser.add_argument("--eval-mode", type=str, default="eval", help="[eval | stats | vis]")
    parser.add_argument("--pi", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--method", type=str, default="causal")
    parser.add_argument("--temp", type=float, default=1)
    args = parser.parse_args()

    run_temp(args)
    #vis_for_prob(args)
      
