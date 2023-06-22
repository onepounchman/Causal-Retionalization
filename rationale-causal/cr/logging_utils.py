import torch

from cr.config import Config

config = Config()


def log(mode, args, log_dict):
    if mode == 'train':
        print((
            f"[train] Epoch: {log_dict['epoch']} | "
            f"batch: {log_dict['batch_idx']} / {log_dict['num_batches']} (global step: {log_dict['global_step']}) | "
            f"train acc: {log_dict['acc'] * 100:.2f} | "
            f"train loss: {log_dict['loss']:.4f} | "
            f"Elapsed {log_dict['elapsed']:.2f}s"
        ))
    elif mode =='dev':
        print((
            f"[{mode}] Epoch: {log_dict['epoch']} | "
            f"global step: {log_dict['global_step']} | "
            f"{mode} acc: {log_dict['eval_acc'] * 100:.2f} | "
            f"{mode} loss: {log_dict['eval_loss']:.4f}| "
        ))
            
    elif mode =='test':
        print((
              f"{mode} acc: {log_dict['test_acc'] * 100:.2f} | "
              f"{mode} precision: {log_dict['test_precision'] * 100:.4f} | "
              f"{mode} recall: {log_dict['test_recall'] * 100:.4f} | "
              f"{mode} F1: {log_dict['test_F1'] * 100:.4f} | "
        ))

def log_cap(args, log_dict):
        print((
              f"{mode} acc: {log_dict['test_acc'] * 100:.2f} | "
              f"{mode} precision: {log_dict['test_precision'] * 100:.4f} | "
              f"{mode} recall: {log_dict['test_recall'] * 100:.4f} | "
              f"{mode} F1: {log_dict['test_F1'] * 100:.4f} | "
              f"{mode} cap_rate: {log_dict['cap_rate'] * 100:.4f} | "
        ))





