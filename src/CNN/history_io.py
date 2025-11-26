"""
Utility functions for saving and loading evaluation history data to/from HDF5 files.

HDF5 file structure:
- /history_data/     (group) - Per-sample metric arrays with compression
- /average_data/     (group) - Averaged metrics (no compression needed)
- /metadata/         (group) - Metadata as datasets (no compression needed)
- /model_params/     (group) - Model hyperparameters (no compression needed)

Created on Nov 26, 2025
@author: Sebastian Prepelita
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import os
import json
import logging

from src.CNN import config
from src.CNN.cnn_model import CNNModelParams


def load_model_configs_from_json(
    json_filepath: str,
    cnn_src_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> List[CNNModelParams]:
    """
    Load model configurations from a JSON file and return a list of CNNModelParams objects.

    This function reads a JSON file containing model configurations and creates
    CNNModelParams objects for each model. The function is designed to be extensible:
    if new parameters are added to CNNModelParams, they only need to be added to the
    JSON file - no changes to this function are required.

    Parameters
    ----------
    json_filepath : str
        Path to the JSON file containing model configurations.
        Can be relative (to cnn_src_dir) or absolute.

    cnn_src_dir : Path, optional
        Directory where the JSON file is located.
        Defaults to the CNN source directory (same directory as this file).

    Returns
    -------
    List[CNNModelParams]
        List of CNNModelParams objects in the order they appear in the JSON file.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the JSON file is malformed or missing required fields.
    KeyError
        If the JSON structure doesn't contain 'models' key.

    JSON File Format
    ----------------
    The JSON file should have the following structure:

    ```json
    {
      "models": [
        {
          "model_name": "example_model",
          "n_channels": 6,
          "samples_per_frame": 2400,
          "cnn_num_filter_list": [4, 4, 4],
          "cnn_filter_size_list": [10, 8, 4],
          "cnn_stride_list": [1, 2, 2],
          "cnn_padding_list": [0, 0, 0],
          "max_pool_filter_size_list": [2, 4, 6],
          "max_pool_stride_size_list": [2, 4, 6],
          "FC_hidden_dims": [10],
          "output_dim": 7,
          "dropout": 0.3
        },
        ... more models ...
      ]
    }
    ```

    Example
    -------
    >>> from src.CNN import history_io
    >>>
    >>> # Load models from default location (src/CNN/model_configs.json)
    >>> models = history_io.load_model_configs_from_json('model_configs.json')
    >>> print(f"Loaded {len(models)} models")
    >>> for model_params in models:
    >>>     print(f"  - {model_params.model_name}")
    >>>
    >>> # Load from absolute path
    >>> models = history_io.load_model_configs_from_json('/path/to/my_models.json')
    >>>
    >>> # Load from custom directory
    >>> from pathlib import Path
    >>> models = history_io.load_model_configs_from_json(
    ...     'configs.json',
    ...     cnn_src_dir=Path('/custom/directory')
    ... )

    Notes
    -----
    - The function uses CNNModelParams.from_dict() for object creation, making it
      automatically compatible with any new fields added to CNNModelParams.
    - Models are validated using CNNModelParams.checkInternalConsistency() when
      they are later used in create_model().
    - If a model configuration is missing required fields, from_dict() will raise
      a clear error indicating which fields are missing.
    """
    # Default to the directory where this file is located (src/CNN/)
    if cnn_src_dir is None:
        cnn_src_dir = Path(__file__).parent

    # Convert to Path object for easier manipulation
    json_path = Path(json_filepath)

    # If path is not absolute, make it relative to cnn_src_dir
    if not json_path.is_absolute():
        json_path = cnn_src_dir / json_path

    # Check if file exists
    if not json_path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {json_path}\n"
            f"Searched in: {cnn_src_dir}"
        )

    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON file: {json_path}\n"
            f"Error: {e}"
        )

    # Validate structure
    if 'models' not in data:
        raise KeyError(
            f"JSON file must contain a 'models' key with a list of model configurations.\n"
            f"Found keys: {list(data.keys())}"
        )

    if not isinstance(data['models'], list):
        raise ValueError(
            f"'models' key must contain a list, got {type(data['models']).__name__}"
        )

    # Create CNNModelParams objects
    model_params_list = []

    for idx, model_dict in enumerate(data['models']):
        try:
            # Use from_dict() - this automatically handles any new fields
            # added to CNNModelParams without needing changes to this function
            model_params = CNNModelParams.from_dict(model_dict)
            model_params_list.append(model_params)

        except TypeError as e:
            raise ValueError(
                f"Failed to create model {idx + 1} from configuration.\n"
                f"Model name: {model_dict.get('model_name', 'UNKNOWN')}\n"
                f"Error: {e}\n"
                f"This may indicate missing required fields in the JSON configuration."
            )

    model_names = "\n".join([f"  {idx + 1}. {mp.model_name}" for idx, mp in enumerate(model_params_list)])
    load_msg = (
        f" JSON loader: Loaded {len(model_params_list)} model configurations from: {json_path}"
    )
    if logger:
        logger.info(load_msg)
    return model_params_list


def save_evaluation_results(
    history_data: Dict[str, np.ndarray],
    avg_metrics: Dict[str, float],
    filename: str,
    model_params: Optional[CNNModelParams] = None,
    training_results_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Save evaluation results to an HDF5 file with organized groups.

    File structure:
        /history_data/          - Per-sample metrics (compressed)
            position_mse
            position_mae
            rotation_mse
            rotation_mae
            angular_error_deg
        /average_data/          - Averaged metrics (not compressed)
            position_mse
            position_mae
            rotation_mse
            rotation_mae
            angular_error_deg
            loss
        /metadata/              - Metadata (not compressed)
            epoch
            split
            num_samples
            ... (user-defined)
        /model_params/          - Model hyperparameters (not compressed)
            model_name
            n_channels
            samples_per_frame
            ... (all CNNModelParams fields)

    Parameters:
    -----------
    history_data : Dict[str, np.ndarray]
        Dictionary containing per-sample metric arrays.
        Expected keys: position_mse, position_mae, rotation_mse, rotation_mae, angular_error_deg

    avg_metrics : Dict[str, float]
        Dictionary containing averaged metrics.
        Expected keys: position_mse, position_mae, rotation_mse, rotation_mae, angular_error_deg, loss

    filename : str
        Name of the HDF5 file (with or without .h5/.hdf5 extension).

    model_params : CNNModelParams, optional
        Model parameters to save. If provided, creates /model_params group.

    training_results_dir : Path, optional
        Directory where the file will be saved.
        Defaults to config.Config.TRAINING_RESUTLS_DIR.

    metadata : Dict[str, Any], optional
        Additional metadata (e.g., epoch, split, checkpoint path).

    Returns:
    --------
    str
        Full path to the saved file.

    Example:
    --------
    >>> from src.CNN.cnn_model import CNNModelParams
    >>>
    >>> # Create model params
    >>> model_params = CNNModelParams(
    ...     model_name="test_model",
    ...     n_channels=6,
    ...     samples_per_frame=2400,
    ...     cnn_num_filter_list=[4, 4, 4],
    ...     cnn_filter_size_list=[10, 8, 4],
    ...     cnn_stride_list=[1, 2, 2],
    ...     cnn_padding_list=[0, 0, 0],
    ...     max_pool_filter_size_list=[2, 4, 6],
    ...     max_pool_stride_size_list=[2, 4, 6],
    ...     FC_hidden_dims=[10],
    ...     output_dim=7,
    ...     dropout=0.3
    ... )
    >>>
    >>> # Evaluate model
    >>> avg_metrics, all_metrics = train.evaluate_per_samples(...)
    >>>
    >>> # Save results
    >>> metadata = {'epoch': 10, 'split': 'test', 'checkpoint': 'path/to/checkpoint.pth'}
    >>> save_evaluation_results(all_metrics, avg_metrics, 'test_results', model_params, metadata=metadata)
    """
    if training_results_dir is None:
        training_results_dir = config.Config.TESTING_RESUTLS_DIR

    os.makedirs(training_results_dir, exist_ok=True)

    if not filename.endswith(('.h5', '.hdf5')):
        filename = filename + '.h5'

    filepath = Path(training_results_dir) / filename

    with h5py.File(filepath, 'w') as f:
        # Create history_data group with compression
        history_grp = f.create_group('history_data')
        for key, value in history_data.items():
            if isinstance(value, np.ndarray):
                history_grp.create_dataset(key, data=value, compression='gzip', compression_opts=4)
            else:
                history_grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)

        # Create average_data group (no compression for small data)
        avg_grp = f.create_group('average_data')
        for key, value in avg_metrics.items():
            avg_grp.create_dataset(key, data=value)

        # Create metadata group (no compression for small data)
        metadata_grp = f.create_group('metadata')
        metadata_grp.create_dataset('num_samples', data=len(next(iter(history_data.values()))))

        if metadata is not None:
            for key, value in metadata.items():
                if isinstance(value, (str, bytes)):
                    metadata_grp.create_dataset(key, data=value)
                elif isinstance(value, (int, float, bool)):
                    metadata_grp.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    # Store lists as JSON strings
                    metadata_grp.create_dataset(key, data=json.dumps(value))
                else:
                    # Convert to string for other types
                    metadata_grp.create_dataset(key, data=str(value))

        # Create model_params group if provided (no compression for small data)
        if model_params is not None:
            model_grp = f.create_group('model_params')
            params_dict = model_params.to_dict()

            for key, value in params_dict.items():
                if isinstance(value, (str, bytes)):
                    model_grp.create_dataset(key, data=value)
                elif isinstance(value, (int, float, bool)):
                    model_grp.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    # Store lists as JSON strings for complex types
                    model_grp.create_dataset(key, data=json.dumps(value))
                else:
                    model_grp.create_dataset(key, data=str(value))

    save_msg = (
        f"  Saved evaluation results to: {filepath}\n"
        f"    History data: {list(history_data.keys())}\n"
        f"    Average data: {list(avg_metrics.keys())}\n"
        f"    Num samples: {len(next(iter(history_data.values())))}"
    )

    if model_params:
        save_msg += f"\n  Model: {model_params.model_name}"

    if logger:
        logger.debug(save_msg)

    return str(filepath)


def load_evaluation_results(
    filename: str,
    training_results_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any], Optional[CNNModelParams]]:
    """
    Load evaluation results from an HDF5 file.

    Parameters:
    -----------
    filename : str
        Name of the HDF5 file (with or without .h5/.hdf5 extension).

    training_results_dir : Path, optional
        Directory where the file is located.
        Defaults to config.Config.TRAINING_RESUTLS_DIR.

    Returns:
    --------
    Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any], Optional[CNNModelParams]]
        - Dictionary containing per-sample metric arrays
        - Dictionary containing averaged metrics
        - Dictionary containing metadata
        - CNNModelParams object (if model_params group exists, otherwise None)

    Example:
    --------
    >>> history_data, avg_metrics, metadata, model_params = load_evaluation_results('test_results.h5')
    >>> print(f"Loaded {metadata['num_samples']} samples")
    >>> print(f"Model: {model_params.model_name}")
    >>> print(f"Average position MAE: {avg_metrics['position_mae']:.4f}")
    >>> print(f"Mean position MAE: {history_data['position_mae'].mean():.4f}")
    """
    if training_results_dir is None:
        training_results_dir = config.Config.TESTING_RESUTLS_DIR

    if not filename.endswith(('.h5', '.hdf5')):
        filename = filename + '.h5'

    filepath = Path(training_results_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Evaluation results file not found: {filepath}")

    history_data = {}
    avg_metrics = {}
    metadata = {}
    model_params = None

    with h5py.File(filepath, 'r') as f:
        # Load history_data group
        if 'history_data' in f:
            history_grp = f['history_data']
            for key in history_grp.keys():
                dataset = history_grp[key]
                history_data[key] = dataset[:]

        # Load average_data group
        if 'average_data' in f:
            avg_grp = f['average_data']
            for key in avg_grp.keys():
                dataset = avg_grp[key]
                avg_metrics[key] = float(dataset[()])

        # Load metadata group
        if 'metadata' in f:
            metadata_grp = f['metadata']
            for key in metadata_grp.keys():
                dataset = metadata_grp[key]
                value = dataset[()]

                # Decode bytes to string if needed
                if isinstance(value, bytes):
                    value = value.decode('utf-8')

                # Try to parse JSON for lists
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass  # Keep as string if not valid JSON

                metadata[key] = value

        # Load model_params group if it exists
        if 'model_params' in f:
            model_grp = f['model_params']
            params_dict = {}

            for key in model_grp.keys():
                dataset = model_grp[key]
                value = dataset[()]

                # Decode bytes to string if needed
                if isinstance(value, bytes):
                    value = value.decode('utf-8')

                # Try to parse JSON for lists
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass  # Keep as string if not valid JSON

                params_dict[key] = value

            # Create CNNModelParams from dict
            model_params = CNNModelParams.from_dict(params_dict)

    load_msg = (
        f"  Loaded evaluation results from: {filepath}\n"
        f"    History data: {list(history_data.keys())}\n"
        f"    Average data: {list(avg_metrics.keys())}\n"
        f"    Num samples: {metadata.get('num_samples', 'unknown')}"
    )

    if model_params:
        load_msg += f"\n  Model: {model_params.model_name}"

    if logger:
        logger.debug(load_msg)
    return history_data, avg_metrics, metadata, model_params
