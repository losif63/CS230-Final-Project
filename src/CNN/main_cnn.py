'''
Created on Nov 20, 2025

@author: Sebastian Prepelita, based on file from Prerana Rane
'''
import time, datetime
import torch
import numpy as np
import random

import sys

from src.CNN import config

from src.CNN import dataset
from src.CNN import cnn_model

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

################# To move in model.py:



if __name__ == '__main__':
    print("Main CNN starting...")
    
    set_seed(config.Config.SEED)    
    #config.Config.print_config()
    
    print(f"Creating CNN model:")
    
    cnnModel = cnn_model.create_model(config.Config)
    print(f"#Trainable params: {cnnModel.trainable_params}")
    print("CNN model completion done...")
    print(f"Total number of parameters: #{cnnModel.getTotalTrainableParams():,} params.")
    sys.exit()
    
    start = time.perf_counter()
    train_loader, val_loader, test_loader = dataset.create_dataloaders(config.Config)
    end = time.perf_counter()
    print(f"   Data loading took {datetime.timedelta(seconds=end-start)}")
    
    
    print("Main CNN ending...")