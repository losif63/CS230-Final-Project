r'''
Created on Nov 20, 2025

    Main file for training of CNN models, CS230 project, fall 2025. 

    Minimum window size: about 300 samples

        1m travel distance --> time = 1m/340 --> N_samples = t/dT=t*fs --> N_samples = 1/340*48000 = 141.172 samples
        2m travel distance --> ... --> N_samples = 2m/340 * 48000 = 282.344 samples

    To run this from non Eclipse, forst go to root:

        cd C:\Users\prepelit\Desktop\CS230-Final-Project-sebastian
        C:\Python311_PyTorch\python.exe -m src.CNN.main_cnn


    # On cluster:
    source /mnt/audio/prepelit/ML_training/python_pythorch_venv/bin/activate
    cd /mnt/audio/prepelit/ML_training/CS230-Final-Project-sebastian
    python -m src.CNN.main_cnn
    deactivate

@author: Sebastian Prepelita, based on baseline file from Prerana Rane
'''
import time
import datetime
import torch
import numpy as np
import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import torch.nn as nn

from src.CNN import cnn_model
from src.CNN import dataset
from src.CNN import train
from src.CNN import metrics
from src.CNN import plotting
from src.CNN import history_io
from src.CNN import config


@contextmanager
def suppress_stdout():
    """
    Context manager to suppress print statements (stdout) while preserving logger output.

    Use this when LogMode.FILE_ONLY is set to prevent print() statements from appearing
    in SLURM .out files while keeping logger output intact.

    Example
    -------
    >>> with suppress_stdout():
    >>>     print("This will not appear")
    >>>     logger.info("This will still appear in the log file")
    """
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


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

def train_single_model_wrapper(args_tuple):
    """
    Wrapper function for parallel training. Sets up a separate logger for each process.

    Parameters
    ----------
    args_tuple : tuple
        Tuple of (model_params, device_str, model_index, total_models, log_mode)

    Returns
    -------
    dict
        Dictionary with training results and status
    """
    start = time.perf_counter()
    model_params, device_str, model_index, total_models, log_mode = args_tuple

    # Create a unique logger for this model with the specified log_mode
    log_filename = f"cnn_training__{model_params.model_name}"
    model_logger = setup_logger(
        log_filename=log_filename,
        mode=log_mode,
        level=logging.DEBUG
    )

    try:
        model_logger.info(f"\n{'='*80}" + f"\nTraining Model {model_index + 1}/{total_models}: {model_params.model_name}"
                        + f"\n \tAssigned to device: {device_str}"
                          + f"\n{'='*80}\n")

        # Train the model
        train_test_model(
            model_params=model_params,
            specific_torch_device=device_str,
            logger=model_logger
        )

        model_logger.info(f"\n{'='*80}" + f"\nSuccessfully completed training for: {model_params.model_name}")
        model_logger.info(f"   TOTAL TIME OK -={datetime.timedelta(seconds=time.perf_counter()-start)}=-\n " + f"{'='*80}\n")

        return {
            'model_name': model_params.model_name,
            'status': 'success',
            'device': device_str
        }

    except Exception as e:
        error_msg = f"Error training model '{model_params.model_name}': {str(e)}"
        model_logger.error(error_msg, exc_info=True)
        return {
            'model_name': model_params.model_name,
            'status': 'failed',
            'error': str(e),
            'device': device_str
        }


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
        config=config.Config,
        logger=logger,
    )
    # torchsummary.summary(cnnModel, input_size = (6, 2400))

    plotting_file = getPlottingFileName(training_results_dir, cnnModel)
    start = time.perf_counter()
    train_loader, val_loader, test_loader = dataset.create_dataloaders(config.Config, logger=logger)
    end = time.perf_counter()

    data_loading_msg = f"   Data loading took {datetime.timedelta(seconds=end-start)}"
    if logger:
        logger.info(data_loading_msg)

    # Assign config values to model_params if they are None (so they get saved to .h5 file)
    log_msgs = ["  (2) Creating ADAM optimizer"]

    if model_params.learning_rate is None:
        model_params.learning_rate = config.Config.LEARNING_RATE
        log_msgs.append(f"      Learning rate (ADAM): {model_params.learning_rate} (from config)")
    else:
        log_msgs.append(f"      Learning rate (ADAM): {model_params.learning_rate} (from model_params)")

    if model_params.weight_decay is None:
        model_params.weight_decay = config.Config.WEIGHT_DECAY
        log_msgs.append(f"      Weight decay (ADAM): {model_params.weight_decay} (from config)")
    else:
        log_msgs.append(f"      Weight decay (ADAM): {model_params.weight_decay} (from model_params)")

    if model_params.num_epochs is None:
        model_params.num_epochs = config.Config.NUM_EPOCHS
        log_msgs.append(f"      Num epochs (training): {model_params.num_epochs} (from config)")
    else:
        log_msgs.append(f"      Num epochs (training): {model_params.num_epochs} (from model_params)")

    if logger:
        logger.info("\n".join(log_msgs))

    optimizer = torch.optim.Adam(
        cnnModel.parameters(),
        lr=model_params.learning_rate,
        weight_decay=model_params.weight_decay
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
        num_epochs=model_params.num_epochs,
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

def launch_parallel_training(main_log_filename = 'MAIN_PARALLEL_TRAINING', model_list_file: str = 'model_configs.json', log_mode = LogMode.BOTH):# Change to FILE_ONLY or TERMINAL_ONLY as needed
    """
    Launch parallel training of multiple models on available GPUs.

    Parameters
    ----------
    model_params_list : list[cnn_model.CNNModelParams]
        List of model parameters to train
    """
    start = time.perf_counter()

    logger = setup_logger(
        log_filename=main_log_filename,
        mode=log_mode,
        level=logging.DEBUG
    )
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This is required for CUDA on Linux/Unix systems. Windows uses 'spawn' by default.
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError as e:
        logger.error(f"Failed to set multiprocessing start method to 'spawn': {str(e)}")
        logger.error("This may cause issues with CUDA on Linux systems")

    model_params_list = history_io.load_model_configs_from_json(model_list_file, logger=logger)
    # Get available devices
    available_GPU_devices = get_available_devices(False, logger=logger)

    if len(available_GPU_devices) == 0:
        error_msg = "No available GPU devices found. Please check"
    # Calculate maximum parallel models based on GPUs and trainings per GPU
    trainings_per_gpu = config.Config.TRAININGS_PER_GPU
    assignment_strategy = config.Config.GPU_ASSIGNMENT_STRATEGY
    max_parallel_models = len(available_GPU_devices) * trainings_per_gpu

    logger.info(f"\n{'='*80}\n Starting PARALLEL training:")
    logger.info(f"  - Total models to train: {len(model_params_list)}")
    logger.info(f"  - Available GPUs: {len(available_GPU_devices)}")
    logger.info(f"  - Trainings per GPU: {trainings_per_gpu}")
    logger.info(f"  - GPU assignment strategy: {assignment_strategy}")
    logger.info(f"  - Max parallel models: {max_parallel_models}")
    logger.info(f"{'='*80}\n")

    # Prepare arguments for parallel training with selected strategy
    training_args = []

    if assignment_strategy == "round_robin":
        # Round-robin: Distribute models evenly across GPUs; All GPUs stay occupied until all models complete
        # Good for heterogeneous model sizes
        # Example with 3 GPUs: Model 0→GPU0, Model 1→GPU1, Model 2→GPU2, Model 3→GPU0, ...
        for idx, model_params in enumerate(model_params_list):
            gpu_index = idx % len(available_GPU_devices)
            device_str = available_GPU_devices[gpu_index]
            training_args.append((model_params, device_str, idx, len(model_params_list), log_mode))

    elif assignment_strategy == "sequential":
        # Sequential: Assign models in blocks to GPUs; Less balanced if model training times vary significantly
        # Example with 3 GPUs and 2 trainings/GPU: Models 0-1→GPU0, Models 2-3→GPU1, Models 4-5→GPU2
        for idx, model_params in enumerate(model_params_list):
            gpu_index = idx // trainings_per_gpu
            # Wrap around if we have more models than total capacity
            gpu_index = gpu_index % len(available_GPU_devices)
            device_str = available_GPU_devices[gpu_index]
            training_args.append((model_params, device_str, idx, len(model_params_list), log_mode))

    else:
        error_msg = f"Invalid GPU_ASSIGNMENT_STRATEGY: '{assignment_strategy}'. Must be 'round_robin' or 'sequential'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log GPU assignment plan
    logger.info("GPU Assignment Plan:")
    for gpu_idx, device_str in enumerate(available_GPU_devices):
        assigned_models = [f"{args[0].model_name}" for args in training_args if args[1] == device_str]
        logger.info(f"  {device_str}: {len(assigned_models)} models - {assigned_models}")
    logger.info("")

    # Train models in parallel using ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=max_parallel_models) as executor:
        # Submit all training jobs
        future_to_model = {
            executor.submit(train_single_model_wrapper, args): args[0].model_name
            for args in training_args
        }

        # Process results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results.append(result)

                if result['status'] == 'success':
                    logger.info(f"✓ Completed: {result['model_name']} on {result['device']}")
                else:
                    logger.error(f"✗ Failed: {result['model_name']} - {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"✗ Exception during training of {model_name}: {str(e)}", exc_info=True)
                results.append({
                    'model_name': model_name,
                    'status': 'exception',
                    'error': str(e)
                })

    # Summary
    logger.info(f"\n{'='*80}\n Training Summary:\n {'='*80}")
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    logger.info(f"  Total models: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")

    logger.info(f"{'='*80}\n" + f"   TOTAL TIME OK -={datetime.timedelta(seconds=time.perf_counter()-start)}=-\n " + f"{'='*80}\n")

    if failed > 0:
        logger.info("\nFailed models:")
        for result in results:
            if result['status'] != 'success':
                logger.info(f"  - {result['model_name']}: {result.get('error', 'Unknown error')}")

    logger.info(f"{'='*80}\n")

def launch_single_simple_model_training(log_mode = LogMode.BOTH):
    logger = setup_logger(
         log_filename="cnn_training",
         mode=log_mode,  # Change to FILE_ONLY or TERMINAL_ONLY as needed
         level=logging.DEBUG
     )
    start = time.perf_counter()
    logger.info("Main CNN (simple single test model) starting...")
    # Get available devices
    available_GPU_devices = get_available_devices(False, logger=logger)

    simple_model_params = cnn_model.CNNModelParams(
         model_name="first_test",
         n_channels=6,
         samples_per_frame=2400,
         cnn_num_filter_list=[4, 4, 4],  # [64, 128, 256, 256], #same as output channels
         cnn_filter_size_list=[10, 8, 4],  # [128, 128, 64, 8],
         cnn_stride_list=[1, 2, 2],  # [1, 1, 1, 1],
         cnn_padding_list=[0, 0, 0],  # [0, 0, 0, 0],
         max_pool_filter_size_list=[2, 4, 6],  # [0, 2, 4, 6], # use 0 to skip
         max_pool_stride_size_list=[2, 4, 6],  # [2, 2, 4, 6], # use 0 to skip
         FC_hidden_dims=[10],  # [512, 256, 128],
         output_dim=7,
         num_epochs = 10,
         learning_rate = 1e-4,
         weight_decay = 1e-5,
         dropout=0.3
     )
    train_test_model(simple_model_params, specific_torch_device = available_GPU_devices[0], logger=logger)

    logger.info(f"Main CNN ending OK -={datetime.timedelta(seconds=time.perf_counter()-start)}=-")

def launch_sequential_training(log_mode = LogMode.BOTH):
    """
    Function loops through all the models in the model_params_list from the .json file and trains them sequentially
    (one after the other). Equivalent to the parallel training when there is only one GPU available.
    """
    # LogMode.FILE_ONLY: Only write to log file
    # LogMode.TERMINAL_ONLY: Only write to terminal (like print)
    # LogMode.BOTH: Write to both log file and terminal (default)
    logger = setup_logger(
        log_filename="cnn_SEQUENTIAL_training",
        mode=log_mode,  # Change to FILE_ONLY or TERMINAL_ONLY as needed
        level=logging.DEBUG
    )
    logger.info("Main SEQUENTIAL CNN starting...")
    # Get available devices
    available_GPU_devices = get_available_devices(False, logger=logger)

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
    logger.info("Main SEQUENTIAL CNN ending...")

if __name__ == '__main__':
    # ========================================================================
    # Option 1: Train a single defined model (see inside the function)
    # This is useful for quick testing or running a single model
    # ========================================================================
    LOG_MODE = LogMode.FILE_ONLY  # Change to BOTH, FILE_ONLY or TERMINAL_ONLY as needed

    # Apply suppress_stdout when FILE_ONLY mode to keep SLURM .out files clean
    context_manager = suppress_stdout() if LOG_MODE == LogMode.FILE_ONLY else contextmanager(lambda: (yield))()

    with context_manager:
        launch_single_simple_model_training(log_mode=LOG_MODE)

    # ===========================================================================
    # Option 2: Multiple models (json), Sequential Training (one model at a time)
    # ===========================================================================
    # with context_manager:
    #     launch_sequential_training(log_mode=LOG_MODE)

    # =======================================================================================================
    # Option 3: Multiple models (json), Parallel Training (train multiple models per GPU, on multiple GPUs)
    # =======================================================================================================
    # with context_manager:
    #     launch_parallel_training(
    #         main_log_filename = 'MAIN_PARALLEL_TRAINING',
    #         model_list_file = 'model_configs.json',
    #         log_mode = LOG_MODE
    #     )
