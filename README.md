# CS230 Final Project

* Jaduk Suh
* Prerana Rane
* Sebastian Preprelita


# MLP Baseline - src/baseline
* config.py              # Configuration settings
* data_utils.py          # Data loading utilities
* dataset.py             # Dataset implementation
* model.py               # MLP Baseline model
* metrics.py             # Loss functions and evaluation metrics
* train.py               # Training and evaluation functions
* plotting.py       # Plotting utilities
* main.py                # Main training script
* requirements.txt       # Python dependencies


To run:

pip install -r requirements.txt

update data path in config.py

python main.py

# CNN - src/CNN
* `cnn_2d_model`           &nbsp;&nbsp;&nbsp;&nbsp; $${\color{green}# 2D CNN model classes and orchestration}$$
* `cnn_model`              # 1D CNN model classes and orchestration
* `config.py`              # General configuration settings. See also model_configs\*.json
* `data_utils.py`          # Data loading utilities
* `dataset.py`             # Dataset implementation. Note loaded dataset is pinned in CPU memory.
  * TRAIN_CACHE_FN, VAL_CACHE_FN, TEST_CACHE_FN in config.py are the name of the cached .hdf5 files for train, validation, test sets, respectively (to speed up dataloading, a cached dataset is created in an .hdf5 file)
* `error_analysis.py`      # Some plots and metrics for error analysis of the 1D CNN on the test dataset.
* `histy_io.py`            # Utility functions for saving and loading evaluation CNN training history data.
* `main_cnn.py`            # Main CNN training script. Handles three different types of training. See to run below.
* `metrics.py`             # Loss functions and evaluation metrics
* `model_configs.json`     # .json file with different model configurations.
* `plotting.py`            # Plotting utilities
* `t_test.py`              # Module that does a t-test analysis over multiple models based on type of input.
* `train.py`               # Training and evaluation functions. One spectogram plotting function.

* requirements.txt       # Python dependencies

To run:
```
pip install -r requirements.txt
```
update data path and other major configs in config.py

`cnn_main.py` has 3 possible types of runs:
* single model: trains a single model
  * call `launch_single_simple_model_training()` function
* multiple model training, sequential: trains a set of models from an input .json file, one after the other on a single device (E.g., GPU-0)
  * call `launch_sequential_training()`
  * NUM_WORKERS_SEQUENTIAL in config.py controls the number of multiprocessors in the dataset loader. For our workstreams, we don't need a lot since most computing is done on the GPU. 
* multiple model training, parallel: trains a set of models from an input .json file in paralle. Here, the script pulls all available devices (e.g., GPU-0, GPU-1, ..., GPU-8) and trains one or more (see TRAININGS_PER_GPU in config.py) models per available device in PARALLEL. Use this if you have multiple GPUs. 
  * call `launch_parallel_training()`
  * GPU_ASSIGNMENT_STRATEGY in config.py controls how the model assignment from the .json file is done: "round_robin" or "sequential" (model 0 --> GPU0, model 1--> GPU 1 etc.). This can help load-balance the GPU usage and/or when the training will finish. Note this depends on how you've constructed the input .json file: most commonly, you'd construct variations sequentially, in which case a "round_robin" training would make sense for load_balancing. Please read config.py for more details.
  * Please reach a nice balance (use nvidia-smi or see when you're running out of memory) between TRAININGS_PER_GPU (how many models are trained in parallel per each GPU device) and batch size (BATCH_SIZE in config.py)
  * NUM_WORKERS_PARALLEL in config.py controls the number of multiprocessors in the dataset loader. For our workstreams, we don't need a lot since most computing is done on the GPU. 

To run the training, comment/uncomment the right function for you (single model, multi-sequentia, multi-parallel) and just run:
```
cd CNN
python cnn_main.py
```

## To change input type (audio vs spectogram vs 2d input)
Modify the following inputs to the three functions in `cnn_main.py`:
1. For raw audio: set `apply_spectograms_params_=None` and `apply_mag_and_gd=False`
2. For 2D inputs (log-magnitude and group delay): set `apply_spectograms_params_=None` and `apply_mag_and_gd=True`
3. For 2D spectogram : set `apply_spectograms_params_= {"N_window": 240, "hop_size": 2, "apply_positional_encoding": True, "compute_on_gpu": True}` (or similar parameters) and `apply_mag_and_gd=False`

# CRNN - src/crnn
run main.py
# Sequential - src/sequential
