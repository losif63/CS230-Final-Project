'''
Created on Nov 20, 2025

@author: Sebastian Prepelita, based on file from Prerana Rane
'''
import time, datetime
import torch
import numpy as np
import os
import random
 
import sys

from src.CNN import config

from src.CNN import dataset
from src.CNN import cnn_model
from src.CNN import train
from src.CNN import plotting
from src.CNN import metrics

import torchsummary


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
    #config.Config.print_config()
    
    print(f"Creating CNN model:")
    cnnModel = cnn_model.create_model(config.Config)
    
    # torchsummary.summary(cnnModel, input_size = (6, 2400))
    
    start = time.perf_counter()
    train_loader, val_loader, test_loader = dataset.create_dataloaders(config.Config)
    end = time.perf_counter()
    print(f"   Data loading took {datetime.timedelta(seconds=end-start)}")
    
    print(f"Creating ADAM optimizer...")
    optimizer = torch.optim.Adam(
        cnnModel.parameters(),
        lr=config.Config.LEARNING_RATE,
        weight_decay=config.Config.WEIGHT_DECAY
    )
    print("DONE... Starting training...")
    
    history = train.train_model(
        model=cnnModel,
        train_loader=train_loader,
        val_loader=val_loader,
        lossFunction=metrics.pose_6dof_loss,
        optimizer=optimizer,
        num_epochs=config.Config.NUM_EPOCHS,
        device=config.Config.DEVICE,
        save_dir=str(config.Config.CHECKPOINT_DIR)
    )
    
    plotting.plot_training_history(history, save_path='training_results.png')
    
    print("Starting testing on TEST dataset:")
    checkpoint_path = os.path.join(config.Config.CHECKPOINT_DIR, 'best_model.pth')
    train.load_checkpoint(cnnModel, checkpoint_path, config.Config.DEVICE)
    
    test_metrics = train.evaluate(cnnModel, test_loader, metrics.pose_6dof_loss, config.Config.DEVICE)
    plotting.print_metrics(test_metrics, title="Test Set Results")
    
    sys.exit()
    
    
    
    
    print("Main CNN ending...")