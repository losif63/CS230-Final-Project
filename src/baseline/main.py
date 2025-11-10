import os
import torch
import numpy as np
import random

from config import Config
from dataset import create_dataloaders
from model import create_model
from metrics import pose_6dof_loss
from baseline.train import train_model, evaluate, load_checkpoint
from plotting import plot_training_history, print_metrics


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(Config.SEED)    
    Config.print_config()
    
    train_loader, val_loader, test_loader = create_dataloaders(Config)
    
    model = create_model(Config)    
    criterion = pose_6dof_loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=Config.NUM_EPOCHS,
        device=Config.DEVICE,
        save_dir=str(Config.CHECKPOINT_DIR)
    )
    
    plot_training_history(history, save_path='training_results.png')
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    load_checkpoint(model, checkpoint_path, Config.DEVICE)
    
    test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE)
    print_metrics(test_metrics, title="Test Set Results")

if __name__ == "__main__":
    main()
