"""
Created on Nov 28, 2025
    
    Module for CS230 fall 2025 project.
    
@author: Sebastian Prepelita
CNNModel2D - 2D CNN for spectrogram/Fourier-based audio-to-pose mapping
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import logging
import src.baseline.config


@dataclass
class CNNModel2DParams:
    """
    Dataclass to store 2D CNN model parameters.

    This class encapsulates all the hyperparameters needed to create a 2D CNN model,
    making it easy to save, load, and pass around model configurations.

    For 2D CNN models, all kernel sizes, strides, and padding must be specified as
    (height, width) tuples to support non-square kernels.

    Training parameters for ADAM (learning_rate, weight_decay, num_epochs) are optional.
    If not provided, values from config.Config will be used during training.
    """

    model_name: str
    n_channels: int
    samples_per_frame: int
    input_height: int
    cnn_num_filter_list: List[int]
    cnn_filter_size_list: List[Tuple[int, int]]
    cnn_stride_list: List[Tuple[int, int]]
    cnn_padding_list: List[Tuple[int, int]]
    max_pool_filter_size_list: List[Tuple[int, int]]
    max_pool_stride_size_list: List[Tuple[int, int]]
    FC_hidden_dims: List[int]
    output_dim: int
    dropout: float
    learning_rate: Optional[float] = None  # for ADAM
    weight_decay: Optional[float] = None  # for ADAM
    num_epochs: Optional[int] = None  # for training

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a CNNModel2DParams instance from a dictionary.

        Only uses fields that are defined in the dataclass, ignoring extra fields
        like 'trainable_params', 'comment', 'hypothesis', 'Estimated total size', etc.

        Automatically converts lists to tuples for 2D parameter fields (since JSON/HDF5
        serialization converts tuples to lists).
        """
        # Get valid field names from the dataclass
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}

        # Filter the input dictionary to only include valid fields
        filtered_data = {
            key: value for key, value in data.items() if key in valid_fields
        }

        # Convert lists to tuples for 2D parameters (tuples get converted to lists during JSON/HDF5 serialization)
        tuple_fields = [
            "cnn_filter_size_list",
            "cnn_stride_list",
            "cnn_padding_list",
            "max_pool_filter_size_list",
            "max_pool_stride_size_list",
        ]

        for field in tuple_fields:
            if field in filtered_data and isinstance(filtered_data[field], list):
                # Convert each element from list to tuple (e.g., [1, 10] -> (1, 10))
                filtered_data[field] = [
                    tuple(item) if isinstance(item, list) else item
                    for item in filtered_data[field]
                ]

        return cls(**filtered_data)

    def checkInternalConsistency(self):
        """
        Validate the internal consistency of 2D CNN model parameters.

        Raises:
            ValueError: If any parameter validation fails.
        """
        # Check n_channels is a positive integer
        if not isinstance(self.n_channels, int) or self.n_channels <= 0:
            raise ValueError(
                f"n_channels must be a positive integer, got {self.n_channels}"
            )

        # Check samples_per_frame is a positive integer
        if not isinstance(self.samples_per_frame, int) or self.samples_per_frame <= 0:
            raise ValueError(
                f"samples_per_frame must be a positive integer, got {self.samples_per_frame}"
            )

        # Check input_height is a positive integer
        if not isinstance(self.input_height, int) or self.input_height <= 0:
            raise ValueError(
                f"input_height must be a positive integer, got {self.input_height}"
            )

        # Check that all CNN-related lists are the same length
        cnn_lists = [
            ("cnn_num_filter_list", self.cnn_num_filter_list),
            ("cnn_filter_size_list", self.cnn_filter_size_list),
            ("cnn_stride_list", self.cnn_stride_list),
            ("cnn_padding_list", self.cnn_padding_list),
            ("max_pool_filter_size_list", self.max_pool_filter_size_list),
            ("max_pool_stride_size_list", self.max_pool_stride_size_list),
        ]

        # Verify all are lists
        for name, lst in cnn_lists:
            if not isinstance(lst, list):
                raise ValueError(f"{name} must be a list, got {type(lst).__name__}")

        # Get the expected length from the first list
        expected_length = len(self.cnn_num_filter_list)

        # Check all lists have the same length
        for name, lst in cnn_lists:
            if len(lst) != expected_length:
                raise ValueError(
                    f"All CNN-related lists must have the same length. "
                    f"cnn_num_filter_list has length {expected_length}, "
                    f"but {name} has length {len(lst)}"
                )

        # Check cnn_num_filter_list contains positive integers
        for i, val in enumerate(self.cnn_num_filter_list):
            if not isinstance(val, int) or val <= 0:
                raise ValueError(
                    f"cnn_num_filter_list[{i}] must be a positive integer, got {val}"
                )

        # Check all elements in kernel/stride/padding lists are tuples of 2 non-negative integers
        tuple_lists = [
            ("cnn_filter_size_list", self.cnn_filter_size_list),
            ("cnn_stride_list", self.cnn_stride_list),
            ("cnn_padding_list", self.cnn_padding_list),
            ("max_pool_filter_size_list", self.max_pool_filter_size_list),
            ("max_pool_stride_size_list", self.max_pool_stride_size_list),
        ]

        for name, lst in tuple_lists:
            for i, val in enumerate(lst):
                if not isinstance(val, tuple):
                    raise ValueError(
                        f"{name}[{i}] must be a tuple of 2 ints, got {type(val).__name__}"
                    )
                if len(val) != 2:
                    raise ValueError(
                        f"{name}[{i}] must be a tuple of 2 ints, got tuple of length {len(val)}"
                    )
                for j, v in enumerate(val):
                    if not isinstance(v, int) or v < 0:
                        raise ValueError(
                            f"{name}[{i}][{j}] must be a non-negative integer, got {v}"
                        )

        # Check FC_hidden_dims is a list of positive integers
        if not isinstance(self.FC_hidden_dims, list):
            raise ValueError(
                f"FC_hidden_dims must be a list, got {type(self.FC_hidden_dims).__name__}"
            )

        for i, dim in enumerate(self.FC_hidden_dims):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"FC_hidden_dims[{i}] must be a positive integer, got {dim}"
                )

        # Check output_dim is in the allowed set {3, 4, 7}
        if self.output_dim not in {3, 4, 7}:
            raise ValueError(
                f"This problem/project requires output_dim to be 3 (position only), 4 (rotation only), or 7 (full 6DOF), got {self.output_dim}"
            )

        # Check dropout is a float between 0.0 and 1.0
        if not isinstance(self.dropout, (float, int)):
            raise ValueError(
                f"dropout must be a float, got {type(self.dropout).__name__}"
            )

        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {self.dropout}")

    def __eq__(self, other):
        """
        Compare two CNNModel2DParams instances for equality.

        Parameters
        ----------
        other : CNNModel2DParams
            Another CNNModel2DParams instance to compare with.

        Returns
        -------
        bool
            True if all parameters are equal, False otherwise.
        """
        if not isinstance(other, CNNModel2DParams):
            return False

        return (
            self.model_name == other.model_name
            and self.n_channels == other.n_channels
            and self.samples_per_frame == other.samples_per_frame
            and self.input_height == other.input_height
            and self.cnn_num_filter_list == other.cnn_num_filter_list
            and self.cnn_filter_size_list == other.cnn_filter_size_list
            and self.cnn_stride_list == other.cnn_stride_list
            and self.cnn_padding_list == other.cnn_padding_list
            and self.max_pool_filter_size_list == other.max_pool_filter_size_list
            and self.max_pool_stride_size_list == other.max_pool_stride_size_list
            and self.FC_hidden_dims == other.FC_hidden_dims
            and self.output_dim == other.output_dim
            and self.dropout == other.dropout
            and self.learning_rate == other.learning_rate
            and self.weight_decay == other.weight_decay
            and self.num_epochs == other.num_epochs
        )

    def __repr__(self):
        """Return a formatted string representation of the model parameters."""
        lines = [
            f"CNNModel2DParams(",
            f"  model_name={self.model_name!r},",
            f"  n_channels={self.n_channels},",
            f"  samples_per_frame={self.samples_per_frame},",
            f"  input_height={self.input_height},",
            f"  cnn_num_filter_list={self.cnn_num_filter_list},",
            f"  cnn_filter_size_list={self.cnn_filter_size_list},",
            f"  cnn_stride_list={self.cnn_stride_list},",
            f"  cnn_padding_list={self.cnn_padding_list},",
            f"  max_pool_filter_size_list={self.max_pool_filter_size_list},",
            f"  max_pool_stride_size_list={self.max_pool_stride_size_list},",
            f"  FC_hidden_dims={self.FC_hidden_dims},",
            f"  output_dim={self.output_dim},",
            f"  dropout={self.dropout},",
            f"  learning_rate={self.learning_rate},",
            f"  weight_decay={self.weight_decay},",
            f"  num_epochs={self.num_epochs}",
            f")",
        ]
        return "\n".join(lines)


def getCnnOutputDimension1D(inDim: int, padding: int, filterSize: int, stride: int):
    """
    Compute the output dimension of a 1D convolution along a single axis.

    Parameters
    ----------
    inDim : int
        Size of the input along one dimension (e.g., width or height).
    padding : int
        Number of padding units applied to each side of the input.
    filterSize : int
        Size of the convolution kernel along this dimension.
    stride : int
        Step size with which the kernel moves across the input.

    Returns
    -------
    int
        The size of the output feature map along this dimension.

    Notes
    -----
    The formula used is:
        outDim = floor((inDim - filterSize + 2*padding) / stride) + 1

    This matches the standard convolution output size calculation
    used in deep learning frameworks such as PyTorch and TensorFlow.
    """
    return int(np.floor((inDim - filterSize + 2 * padding) / stride)) + 1


class CNNModel2D(nn.Module):
    def __init__(
        self,
        model_name="first_test",
        n_channels=6,
        samples_per_frame=2400,
        input_height=2,
        cnn_num_filter_list=[64, 128, 256, 256],  # same as output channels
        cnn_filter_size_list=[
            (2, 128),
            (2, 128),
            (2, 64),
            (2, 8),
        ],  # tuple (height, width)
        cnn_stride_list=[(1, 1), (1, 1), (1, 1), (1, 1)],  # tuple (height, width)
        cnn_padding_list=[(0, 0), (0, 0), (0, 0), (0, 0)],  # tuple (height, width)
        max_pool_filter_size_list=[
            (0, 0),
            (1, 2),
            (1, 3),
            (1, 4),
        ],  # use (0,0) to skip, tuple (height, width)
        max_pool_stride_size_list=[
            (1, 2),
            (1, 2),
            (1, 3),
            (1, 4),
        ],  # tuple (height, width)
        FC_hidden_dims=[512, 256, 128],
        output_dim=7,
        dropout=0.3,
    ):
        """
        Build a 2D CNN + FC regression model for audio-to-pose mapping.

        Here, the input is a 2D representation of the audio, for instance:
            * spectrogram (e.g., STFT): real and imaginary part
            * magnitude spectrogram and delay information

        Inputs:
            model_name: Model name - used in saving data and other parameters. Use this when you train a lot of these models.
            n_channels (int):
                Number of input channels (e.g., 2 for real+imaginary STFT, 6 for multi-mic magnitude).
            samples_per_frame (int):
                Width of the 2D input (e.g., number of time frames in spectrogram).
            input_height (int):
                Height of the 2D input (e.g., number of frequency bins in spectrogram).
                Default: 2
            cnn_num_filter_list (list[int]):
                Output channels (filters) for each Conv2d layer.
            cnn_filter_size_list (list[tuple[int, int]]):
                Kernel sizes for each Conv2d layer as (height, width) tuples.
                Example: [(2, 5), (2, 3)] means kernels of size 2×5 and 2×3.
            cnn_stride_list (list[tuple[int, int]]):
                Strides for each Conv2d layer as (height, width) tuples.
                Example: [(1, 1), (1, 2)] for different stride in each dimension.
            cnn_padding_list (list[tuple[int, int]]):
                Padding values for each Conv2d layer as (height, width) tuples.
                Example: [(0, 2), (0, 1)] for padding only width dimension.
            max_pool_filter_size_list (list[tuple[int, int]]):
                Kernel sizes for MaxPool2d layers as (height, width) tuples.
                Use (0, 0) to skip pooling at that stage.
            max_pool_stride_size_list (list[tuple[int, int]]):
                Strides for MaxPool2d layers as (height, width) tuples.
                Example: [(1, 2), (1, 3)] to pool only along width dimension.
            FC_hidden_dims (list[int]):
                Hidden layer sizes for the fully connected stack.
                Each FC layer is followed by BatchNorm1d, ReLU, and Dropout.
            output_dim (int):
                Dimension of the final regression output.
                For 6DOF pose: 3 position + 4 quaternion = 7.
            dropout (float):
                Dropout probability applied after CNN and FC activations
                (typical range: 0.1-0.5).

        Architecture:
            - CNN stack: Conv2d -> BatchNorm2d -> ReLU -> Dropout -> (optional MaxPool2d)
            - Flatten
            - FC stack: Linear -> BatchNorm1d -> ReLU -> Dropout (repeated per hidden dim)
            - Output layer: Linear -> regression output (no BN/Dropout)

        Input Shape:
            (batch_size, n_channels, input_height, samples_per_frame)
            where:
                - input_height: First spatial dimension (e.g., frequency bins)
                - samples_per_frame: Second spatial dimension (e.g., time frames)

        Notes:
            - All kernel sizes, strides, and padding must be specified as (height, width) tuples.
            - Tracks both spatial dimensions independently through the network.
            - BatchNorm parameters are tracked separately (gamma + beta per channel).
            - Dropout regularizes both CNN and FC layers.
            - Parameter counts are computed and stored in self.trainable_params.
        """
        self.n_cnn_filters = len(cnn_num_filter_list)
        assert self.n_cnn_filters == len(cnn_filter_size_list)
        assert self.n_cnn_filters == len(cnn_stride_list)
        assert self.n_cnn_filters == len(cnn_padding_list)
        assert self.n_cnn_filters == len(max_pool_filter_size_list)
        assert self.n_cnn_filters == len(max_pool_stride_size_list)

        super().__init__()

        self.n_channels = n_channels
        self.samples_per_frame = samples_per_frame
        self.input_height = input_height
        self.output_dim = output_dim

        self.model_name = model_name

        # For 2D convolutions, track both spatial dimensions separately
        # input_height: First spatial dimension (e.g., frequency bins in spectrogram)
        # input_width: Second spatial dimension (e.g., time frames), initialized from samples_per_frame
        current_height = input_height
        current_width = samples_per_frame
        prev_channels = n_channels

        layers = []
        self.trainable_params = []

        batch_norm_channels = 0
        # 2D CNN stack:
        #####################
        for cnn_layer_idx in range(self.n_cnn_filters):
            # Extract tuple parameters (all must be tuples)
            kernel_size = cnn_filter_size_list[cnn_layer_idx]
            stride = cnn_stride_list[cnn_layer_idx]
            padding = cnn_padding_list[cnn_layer_idx]
            pool_kernel = max_pool_filter_size_list[cnn_layer_idx]
            pool_stride = max_pool_stride_size_list[cnn_layer_idx]

            # 2D CNN on spectrogram-like input
            layers.append(
                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=cnn_num_filter_list[cnn_layer_idx],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            # BatchNorm after Conv2d
            layers.append(nn.BatchNorm2d(cnn_num_filter_list[cnn_layer_idx]))
            batch_norm_channels += cnn_num_filter_list[cnn_layer_idx] * 2
            # Activation
            layers.append(nn.ReLU())
            # Dropout (after BN + ReLU)
            layers.append(nn.Dropout(p=dropout))
            # For 2D conv: out_channels * in_channels * kernel_height * kernel_width + out_channels (bias)
            kernel_h, kernel_w = kernel_size
            conv_params = (
                cnn_num_filter_list[cnn_layer_idx] * prev_channels * kernel_h * kernel_w
            ) + cnn_num_filter_list[
                cnn_layer_idx
            ]  # biases (one per output channel)

            self.trainable_params.append({"CNN 2D": conv_params})
            prev_channels = cnn_num_filter_list[cnn_layer_idx]

            # Update both spatial dimensions separately
            pad_h, pad_w = padding
            stride_h, stride_w = stride
            current_height = getCnnOutputDimension1D(
                inDim=current_height,
                padding=pad_h,
                filterSize=kernel_h,
                stride=stride_h,
            )
            current_width = getCnnOutputDimension1D(
                inDim=current_width,
                padding=pad_w,
                filterSize=kernel_w,
                stride=stride_w,
            )

            # Check if pooling is enabled (kernel > 0 AND stride > 0)
            # Both dimensions must have valid kernel and stride to enable pooling
            pool_kernel_h, pool_kernel_w = pool_kernel
            pool_stride_h, pool_stride_w = pool_stride

            if (pool_kernel_h > 0 and pool_stride_h > 0) or (
                pool_kernel_w > 0 and pool_stride_w > 0
            ):
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_kernel,
                        stride=pool_stride,
                    )
                )
                # Update both dimensions after max pooling (padding = 0):
                # Only update dimension if that dimension's pooling is enabled
                if pool_kernel_h > 0 and pool_stride_h > 0:
                    current_height = getCnnOutputDimension1D(
                        inDim=current_height,
                        padding=0,
                        filterSize=pool_kernel_h,
                        stride=pool_stride_h,
                    )

                if pool_kernel_w > 0 and pool_stride_w > 0:
                    current_width = getCnnOutputDimension1D(
                        inDim=current_width,
                        padding=0,
                        filterSize=pool_kernel_w,
                        stride=pool_stride_w,
                    )
        # Flatten before connecting to FC layers:
        layers.append(nn.Flatten())
        # Compute total output dimensions for the FC first layer:
        # After 2D conv layers: (batch, channels, height, width) -> flatten to (batch, channels*height*width)
        prev_dim = prev_channels * current_height * current_width
        # FC stack:
        #######################
        for h_dim in FC_hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # Batchnorm:
            layers.append(nn.BatchNorm1d(h_dim))
            batch_norm_channels += h_dim * 2
            self.trainable_params.append(
                {"FC (weights + biases)": h_dim * prev_dim + h_dim}
            )
            # Activation:
            layers.append(nn.ReLU())
            # Dropout:
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # Add last layer:
        self.trainable_params.append(
            {"Output layer FC (weights + biases)": output_dim * prev_dim + output_dim}
        )
        # Batch norm trainable param:
        if batch_norm_channels > 0:
            self.trainable_params.append({"Batch norm params": batch_norm_channels})

        self.CnNetwork = nn.Sequential(*layers)

    def getModelName(self) -> str:
        return self.model_name

    def getTotalTrainableParams(self):
        if self.trainable_params is None:
            return 0
        sum_ = 0
        for paramdict in self.trainable_params:
            if len(paramdict.keys()) > 1:
                raise ValueError(
                    "I don't know how to calculate total number of parameters. I expected a single dict!"
                )
            for key_ in paramdict.keys():
                sum_ += paramdict[key_]
        return sum_

    def forward(self, x):
        pose = self.CnNetwork(x)
        # x: (batch_size, n_channels, input_height, samples_per_frame) 2D audio representation
        if self.output_dim == 3:
            return pose
        elif self.output_dim == 4:
            quaternion = F.normalize(pose, p=2, dim=1)
            return quaternion
        else:
            position = pose[:, :3]
            quaternion = pose[:, 3:]
            # Normalize quaternion part (last 4 elements):
            quaternion = F.normalize(quaternion, p=2, dim=1)
            return torch.cat([position, quaternion], dim=1)


def create_2d_cnn_model(
    model_params: CNNModel2DParams,
    config: src.baseline.config.Config,
    logger: Optional[logging.Logger] = None,
):
    """
    Create a 2D CNN model from a CNNModel2DParams dataclass.

    Parameters
    ----------
    model_params : CNNModel2DParams
        Dataclass containing all model hyperparameters for 2D CNN.
    config : src.baseline.config.Config
        Configuration object containing device and other settings.
    logger : logging.Logger, optional
        Logger instance for logging model creation details.

    Returns
    -------
    CNNModel2D
        The created model, moved to the appropriate device.

    Raises
    ------
    ValueError
        If model_params fails internal consistency checks.

    Notes
    -----
    This function is specifically for 2D CNN models that process spectrogram-like
    inputs (e.g., Fourier-transformed audio data). All kernel sizes, strides, and
    padding must be specified as (height, width) tuples in the model_params.

    Example
    -------
    >>> from src.CNN.cnn_2d_model import CNNModel2DParams, create_2d_cnn_model
    >>> from src.CNN import config
    >>>
    >>> params = CNNModel2DParams(
    ...     model_name="fourier_cnn",
    ...     n_channels=6,
    ...     samples_per_frame=2400,
    ...     input_height=2,
    ...     cnn_num_filter_list=[64, 128, 256],
    ...     cnn_filter_size_list=[(2, 128), (2, 64), (2, 32)],
    ...     cnn_stride_list=[(1, 1), (1, 1), (1, 1)],
    ...     cnn_padding_list=[(0, 0), (0, 0), (0, 0)],
    ...     max_pool_filter_size_list=[(0, 0), (1, 2), (1, 3)],
    ...     max_pool_stride_size_list=[(1, 2), (1, 2), (1, 3)],
    ...     FC_hidden_dims=[512, 256],
    ...     output_dim=7,
    ...     dropout=0.3
    ... )
    >>>
    >>> model = create_2d_cnn_model(params, config.Config)
    """
    # Validate parameters before creating the model
    model_params.checkInternalConsistency()

    # Extract parameters from dataclass
    params_dict = model_params.to_dict()

    model = CNNModel2D(
        model_name=params_dict["model_name"],
        n_channels=params_dict["n_channels"],
        samples_per_frame=params_dict["samples_per_frame"],
        input_height=params_dict["input_height"],
        cnn_num_filter_list=params_dict["cnn_num_filter_list"],
        cnn_filter_size_list=params_dict["cnn_filter_size_list"],
        cnn_stride_list=params_dict["cnn_stride_list"],
        cnn_padding_list=params_dict["cnn_padding_list"],
        max_pool_filter_size_list=params_dict["max_pool_filter_size_list"],
        max_pool_stride_size_list=params_dict["max_pool_stride_size_list"],
        FC_hidden_dims=params_dict["FC_hidden_dims"],
        output_dim=params_dict["output_dim"],
        dropout=params_dict["dropout"],
    )
    model = model.to(config.DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger is not None:
        msg_ = (
            f"2D CNN Model created with \n\t\t {num_params:,} trainable parameters\n"
            + f"\t\t Model located on '{ next(model.parameters()).device}' device.\n"
        )
        msg_ += "\t\t Layers: " + str(model.trainable_params)
        logger.debug(msg_)

    return model
