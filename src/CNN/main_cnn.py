r'''
Created on Nov 20, 2025

    Minimum window size: about 300 samples

        1m travel distance --> time = 1m/340 --> N_samples = t/dT=t*fs --> N_samples = 1/340*48000 = 141.172 samples
        2m travel distance --> ... --> N_samples = 2m/340 * 48000 = 282.344 samples

    To run this from non Eclipse, forst go to root:

        cd C:\Users\prepelit\Desktop\CS230-Final-Project-sebastian
        C:\Python311_PyTorch\python.exe -m src.CNN.main_cnn


@author: Sebastian Prepelita, based on file from Prerana Rane
'''
import time, datetime
import torch
import numpy as np
import os
import random
import logging
from enum import Enum
from typing import Optional

import sys

from src.CNN import config

from src.CNN import dataset
from src.CNN import cnn_model
from src.CNN import train
from src.CNN import plotting
from src.CNN import metrics
from src.CNN import history_io

import torchsummary


class LogMode(Enum):
    """Enum for logging output modes."""
    FILE_ONLY = "file"
    TERMINAL_ONLY = "terminal"
    BOTH = "both"


def setup_logger(
    log_filename: str,
    mode: LogMode = LogMode.BOTH,
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with configurable output mode.

    Parameters
    ----------
    log_filename : str
        Name of the log file (without extension, .log will be added automatically).
    mode : LogMode
        Where to send log output:
        - LogMode.FILE_ONLY: Only write to file
        - LogMode.TERMINAL_ONLY: Only write to terminal
        - LogMode.BOTH: Write to both file and terminal
    log_dir : str, optional
        Directory where log file should be saved.
        Defaults to training_results directory.
    level : int
        Logging level (default: logging.INFO)

    Returns
    -------
    logging.Logger
        Configured logger instance

    Example
    -------
    >>> logger = setup_logger("training_log", mode=LogMode.BOTH)
    >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger('CNN_Training')
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Set up formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add file handler if needed
    if mode in [LogMode.FILE_ONLY, LogMode.BOTH]:
        if log_dir is None:
            log_dir = str(config.Config.TRAINING_RESUTLS_DIR)
        os.makedirs(log_dir, exist_ok=True)

        if not log_filename.endswith('.log'):
            log_filename = log_filename + '.log'

        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add terminal handler if needed
    if mode in [LogMode.TERMINAL_ONLY, LogMode.BOTH]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_available_devices(return_CPU=False, logger: Optional[logging.Logger] = None):
    """
    Retrieve all available CUDA devices from PyTorch.

    Parameters
    ----------
    return_CPU : bool
        If True, only return CPU device (default: False)
    logger : logging.Logger, optional
        Logger instance for logging device information

    Returns
    -------
    list[str]
        List of available device strings. Always includes 'cpu' as the first element,
        followed by 'cuda:0', 'cuda:1', etc. for each available CUDA device.

    Example
    -------
    >>> devices = get_available_devices()
    >>> print(devices)
    ['cpu', 'cuda:0', 'cuda:1']
    >>> # Select a device
    >>> device = torch.device(devices[1])  # Use first GPU
    """
    if return_CPU:
        devices = ['cpu']
    else:
        devices = []

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        header = f" CUDA is available with {num_gpus} GPU(s)"
        if logger:
            logger.info(header)

        for i in range(num_gpus):
            device_name = f"cuda:{i}"
            devices.append(device_name)
            # Get GPU properties
            # props = torch.cuda.get_device_properties(i)
            # gpu_info = (
            #     f"GPU {i}: {props.name}\n"
            #     f"  Device String: {device_name}\n"
            #     f"  Compute Capability: {props.major}.{props.minor}\n"
            #     f"  Total Memory: {props.total_memory / 1024**3:.2f} GB\n"
            #     f"  Multiprocessors: {props.multi_processor_count}"
            # )
            # if logger:
            #     logger.info(gpu_info)
    else:
        no_cuda_msg = f" !!! CUDA is not available - CPU only !!!"
        if logger:
            logger.info(no_cuda_msg)

    devices_msg = f"     Available devices: {devices}"
    if logger:
        logger.debug(devices_msg)

    return devices


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def getPlottingFileName(training_results_dir, cnnModel):
    os.makedirs(training_results_dir, exist_ok=True)
    return training_results_dir / f'{cnnModel.getModelName()}__training_results.png'

def train_test_model(
    model_params: cnn_model.CNNModelParams,
    specific_torch_device = None,
    logger: Optional[logging.Logger] = None
):
    """
    Train and test a CNN model.

    Parameters
    ----------
    model_params : CNNModelParams
        Model parameters
    specific_torch_device : torch.device, optional
        Specific device to use for training
    logger : logging.Logger, optional
        Logger instance for logging progress
    """
    if specific_torch_device is not None:
        config.Config.DEVICE = specific_torch_device
    # Set the seed for reproducibility:
    set_seed(config.Config.SEED)
    training_results_dir = config.Config.TRAINING_RESUTLS_DIR

    msg = f"  (1) Creating CNN model '{model_params.model_name}'"
    if logger:
        logger.info(msg)

    # Create the model from parameters (automatically validates)
    cnnModel = cnn_model.create_model(
        model_params=model_params,
        config=config.Config
    )

    # torchsummary.summary(cnnModel, input_size = (6, 2400))
    plotting_file = getPlottingFileName(training_results_dir, cnnModel)
    start = time.perf_counter()
    train_loader, val_loader, test_loader = dataset.create_dataloaders(config.Config, logger=logger)
    end = time.perf_counter()

    data_loading_msg = f"   Data loading took {datetime.timedelta(seconds=end-start)}"
    if logger:
        logger.info(data_loading_msg)

    optimizer_msg = "  (2) Creating ADAM optimizer"
    if logger:
        logger.info(optimizer_msg)

    optimizer = torch.optim.Adam(
        cnnModel.parameters(),
        lr=config.Config.LEARNING_RATE,
        weight_decay=config.Config.WEIGHT_DECAY
    )
    ############################
    # Training the model:
    ############################
    training_start_msg = "  (3) Starting training..."
    if logger:
        logger.info(training_start_msg)

    start = time.perf_counter()
    history = train.train_model(
        model=cnnModel,
        train_loader=train_loader,
        val_loader=val_loader,
        lossFunction=metrics.pose_6dof_loss,
        optimizer=optimizer,
        num_epochs=config.Config.NUM_EPOCHS,
        device=config.Config.DEVICE,
        save_dir=str(config.Config.CHECKPOINT_DIR),
        logger=logger
    )

    training_done_msg = f"      DONE TRAINING - took {datetime.timedelta(seconds=time.perf_counter()-start)}"
    if logger:
        logger.info(training_done_msg)
        logger.info(f"  (4) Plotting to file {plotting_file}")

    ############################
    # Plotting:
    ############################
    plotting.plot_training_history(history, save_path=str(plotting_file))

    testing_msg = "  (5) Starting testing on TEST dataset"
    if logger:
        logger.info(testing_msg)

    start = time.perf_counter()
    # ############################################
    # Testing on test set (+ writing .hdf5 file):
    # ############################################
    test_avg_metrics, test_all_metrics_per_sample = train.evaluate_per_samples(
        cnnModel, test_loader, metrics.pose_6dof_loss, config.Config.DEVICE,
        evalTest=True, logger=logger
    )
    testing_msg = "  (6) Writing test results to filem and reloading..."
    if logger:
        logger.info(testing_msg)
    # Save with model parameters and metadata
    checkpoint_path = os.path.join(config.Config.CHECKPOINT_DIR, f'{cnnModel.getModelName()}__latest_trained_model.pth')
    metadata = {
        'split': 'test',
        'batch_size': config.Config.BATCH_SIZE,
        'latest_checkpoint': str(checkpoint_path),
        'sessions': config.Config.TEST_SESSIONS
    }
    train_result_fn = f'{cnnModel.getModelName()}__test_results'
    filepath = history_io.save_evaluation_results(
        history_data=test_all_metrics_per_sample,
        avg_metrics=test_avg_metrics,
        filename=train_result_fn,
        model_params=model_params,
        metadata=metadata,
        logger=logger
    )

    # Test reading/writing:
    loaded_data, loaded_avg_metrics, loaded_metadata, loaded_model_params = history_io.load_evaluation_results(
        filename=train_result_fn,
        logger=logger
    )
    testing_msg = "  (7) Retrieving and plotting test results..."
    if logger:
        logger.info(testing_msg)
    assert loaded_model_params == model_params
    plotting.print_metrics(test_avg_metrics, title="Test Set Results [per sample]", logger=logger)

    # Plot test results
    test_results_plot_path = config.Config.TESTING_RESUTLS_DIR / f'{cnnModel.getModelName()}__test_results_plot.png'
    plotting.plot_test_results(
        avg_metrics=test_avg_metrics,
        history_per_sample=test_all_metrics_per_sample,
        model_params=model_params,
        save_path=str(test_results_plot_path),
        logger=logger
    )


if __name__ == '__main__':
    print("Main CNN starting...")

    # ========================================================================
    # Set up logger - Configure output mode here
    # ========================================================================
    # LogMode.FILE_ONLY: Only write to log file
    # LogMode.TERMINAL_ONLY: Only write to terminal (like print)
    # LogMode.BOTH: Write to both log file and terminal (default)

    logger = setup_logger(
        log_filename="cnn_training",
        mode=LogMode.BOTH,  # Change to FILE_ONLY or TERMINAL_ONLY as needed
        level=logging.DEBUG
    )

    logger.info("Main CNN starting...")

    # Get available devices
    available_GPU_devices = get_available_devices(False, logger=logger)

    # You can select a specific device like this:
    # selected_device = available_devices[1]  # Select first GPU
    # config.Config.DEVICE = torch.device(selected_device)

    # ========================================================================
    # Option 1: Load models from JSON configuration file
    # ========================================================================
    model_params_list = history_io.load_model_configs_from_json('model_configs.json', logger=logger)

    for idx, model_params in enumerate(model_params_list):
        msg_start = f"\n  {'='*80}\n" + f"  Training Model {idx + 1}/{len(model_params_list)}: {model_params.model_name}" +  "  Creating CNN model '{model_params.model_name}'\n" + f"  {'='*80}"
        logger.info(msg_start)

        try:
            train_test_model(model_params, specific_torch_device = available_GPU_devices[0], logger=logger)
        except Exception as e:
            error_msg = f"Error training model '{model_params.model_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise e

    # ========================================================================
    # Option 2: Train a single model defined here (current default)
    # ========================================================================
    # This is useful for quick testing or running a single model
    # model_params = cnn_model.CNNModelParams(
    #     model_name="first_test",
    #     n_channels=6,
    #     samples_per_frame=2400,
    #     cnn_num_filter_list=[4, 4, 4],  # [64, 128, 256, 256], #same as output channels
    #     cnn_filter_size_list=[10, 8, 4],  # [128, 128, 64, 8],
    #     cnn_stride_list=[1, 2, 2],  # [1, 1, 1, 1],
    #     cnn_padding_list=[0, 0, 0],  # [0, 0, 0, 0],
    #     max_pool_filter_size_list=[2, 4, 6],  # [0, 2, 4, 6], # use 0 to skip
    #     max_pool_stride_size_list=[2, 4, 6],  # [2, 2, 4, 6], # use 0 to skip
    #     FC_hidden_dims=[10],  # [512, 256, 128],
    #     output_dim=7,
    #     dropout=0.3
    # )
    # train_test_model(model_params, specific_torch_device = available_GPU_devices[0], logger=logger)

    logger.info("Main CNN ending...")
