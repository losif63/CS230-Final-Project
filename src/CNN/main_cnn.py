'''
Created on Nov 20, 2025

@author: Sebastian Prepelita, based on file from Prerana Rane
'''
import os
import torch
import numpy as np
import random

from src.CNN import config

from src.CNN import dataset

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    print("Main CNN starting...")
    
    set_seed(config.Config.SEED)    
    config.Config.print_config()
    
    train_loader, val_loader, test_loader = dataset.create_dataloaders(config.Config)
    
    
    print("Main CNN ending...")