'''
Created on Nov 23, 2025

@author: Sebastian Prepelita based on basedline model by Prerana Rane
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import src.baseline.config


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
    return int(np.floor((inDim - filterSize + 2 * padding) / stride )) + 1

def getCnnOutputDimensions(
    inDims: list[int],
    paddings: list[int],
    filterSizes: list[int],
    strides: list[int]
) -> tuple[int, ...]:
    """
    Compute the output dimensions of a convolutional layer across multiple axes.
    
    Extension to getCnnOutputDimension1D() to multiple dimensions.

    Parameters
    ----------
    inDims : list[int]
        Input sizes along each dimension (e.g., [height, width] for 2D).
    paddings : list[int]
        Padding applied to each dimension (number of units added to both sides).
    filterSizes : list[int]
        Kernel sizes along each dimension.
    strides : list[int]
        Stride values along each dimension.

    Returns
    -------
    tuple[int, ...]
        Output sizes along each dimension.

    Notes
    -----
    The formula per dimension is:
        outDim = floor((inDim - filterSize + 2*padding) / stride) + 1

    Examples
    --------
    >>> getCnnOutputDimensions([32, 32], [1, 1], [3, 3], [1, 1])
    (32, 32)   # "same" convolution

    >>> getCnnOutputDimensions([64, 64], [0, 0], [5, 5], [2, 2])
    (30, 30)
    """
    if not (len(inDims) == len(paddings) == len(filterSizes) == len(strides)):
        raise ValueError("All parameter sequences must have the same length.")

    outDims = []
    for inDim, pad, fSize, stride in zip(inDims, paddings, filterSizes, strides):
        outDim = getCnnOutputDimension1D(inDim, pad, fSize, stride)
        outDims.append(outDim)
    return tuple(outDims)

class CNNModel1D(nn.Module):
    def __init__(self, 
                 n_channels = 6, 
                 samples_per_frame = 2400,
                 cnn_num_filter_list = [64, 128, 256, 256], #same as output channels
                 cnn_filter_size_list = [128, 128, 64, 8],
                 cnn_stride_list = [1, 1, 1, 1],
                 cnn_padding_list = [0, 0, 0, 0],
                 max_pool_filter_size_list = [0, 2, 3, 4], # use 0 to skip
                 max_pool_stride_size_list = [2, 2, 3, 4], # use 0 to skip
                 FC_hidden_dims = [512, 256, 128],
                 output_dim = 7, 
                 dropout = 0.3
               ):
        """
        The function will conduct 1D convolutions across the RAW audio input!
        
        Inputs:
            n_channels: Number of audio channels
            samples_per_frame: Number of audio samples per frame
            cnn filter config:
                cnn_num_filter_list: The number of filters (e.g., output channels
            FC_hidden_dims: List of hidden layer dimensions of the last FC layers.
            output_dim: Output dimension (7 for 6DOF: 3 position + 4 quaternion)
            dropout: Dropout probability
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
        self.output_dim = output_dim
        
        input_dim = samples_per_frame
        prev_channels = n_channels
        N_dims = 1

        layers = []
        self.trainable_params = []
        
        batch_norm_channels = 0
        # 1D CNN stack:
        #####################
        for cnn_layer_idx in range(self.n_cnn_filters):
            #1D CNN on raw audio
            layers.append(nn.Conv1d(in_channels=prev_channels, 
                                    out_channels=cnn_num_filter_list[cnn_layer_idx], 
                                    kernel_size=cnn_filter_size_list[cnn_layer_idx], 
                                    stride=cnn_stride_list[cnn_layer_idx], 
                                    padding=cnn_padding_list[cnn_layer_idx],)
                         )
            # BatchNorm after Conv1d
            layers.append(nn.BatchNorm1d(cnn_num_filter_list[cnn_layer_idx]))
            batch_norm_channels += cnn_num_filter_list[cnn_layer_idx]*2
            # Activation
            layers.append(nn.ReLU())
            # Dropout (after BN + ReLU)
            layers.append(nn.Dropout(p=dropout))
            # out_channels * in_channels * kernel_size + out_channels (bias)
            conv_params = (cnn_num_filter_list[cnn_layer_idx] * prev_channels * cnn_filter_size_list[cnn_layer_idx]) \
              + cnn_num_filter_list[cnn_layer_idx]  # biases

            self.trainable_params.append( {'CNN' : conv_params} )
            prev_channels = cnn_num_filter_list[cnn_layer_idx]
            input_dim = getCnnOutputDimension1D(inDim = input_dim, padding = cnn_padding_list[cnn_layer_idx], filterSize = cnn_filter_size_list[cnn_layer_idx], stride = cnn_stride_list[cnn_layer_idx])
            
            if max_pool_filter_size_list[cnn_layer_idx] > 0:
                layers.append(nn.MaxPool1d(kernel_size = max_pool_filter_size_list[cnn_layer_idx],
                                           stride = max_pool_stride_size_list[cnn_layer_idx])
                              )
                # Repeat output estimate for max pooling (padding = 0):
                input_dim = getCnnOutputDimension1D(inDim = input_dim, padding = 0, filterSize = max_pool_filter_size_list[cnn_layer_idx], 
                                                    stride = max_pool_stride_size_list[cnn_layer_idx])
        # Flatten before connecting to FC layers:
        layers.append(nn.Flatten())
        # Compute total output dimensions for the FC first layer:
        prev_dim = N_dims * prev_channels * input_dim # out_channels * N_samples        
        # FC stack:
        #######################
        for h_dim in FC_hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            self.trainable_params.append( {'FC (weights + biases)': h_dim*prev_dim+h_dim} )
            layers.append(nn.ReLU())
            # Maybe BatchNorm:
            #layers.append(nn.BatchNorm1d(h_dim))
            # batch_norm_channels+= h_dim*2
            # Maybe Dropout:
            # layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # Add last layer:
        self.trainable_params.append( {'Output layer FC (weights + biases)': output_dim*prev_dim+output_dim} )
        # Batch norm trainable param:
        if batch_norm_channels > 0:
            self.trainable_params.append( {'Batch norm params': batch_norm_channels} )

        self.CnNetwork = nn.Sequential(*layers)
        
    def getTotalTrainableParams(self):
        if self.trainable_params is None:
            return 0
        sum_ = 0
        for paramdict in self.trainable_params:
            if len(paramdict.keys()) > 1:
                raise ValueError("I don't know how to calculate total number of parameters. I expected a single dict!")
            for key_ in paramdict.keys():
                sum_ += paramdict[key_]
        return sum_

    def forward(self, x):
        #x: (batch, n_channels, samples_per_frame) raw audio
        # batch_size = x.shape[0]
        pose = self.CnNetwork(x)
        # Normalize quaternion part (last 4 values)
        position = pose[:, :3]
        quaternion = pose[:, 3:]
        quaternion = F.normalize(quaternion, p=2, dim=1)
        return torch.cat([position, quaternion], dim=1)


def create_model(config: src.baseline.config.Config):
    # Currently only manual parameters...
    model = CNNModel1D(
           n_channels = 6, 
           samples_per_frame = 2400,
           cnn_num_filter_list = [64, 128, 256, 256], #same as output channels
           cnn_filter_size_list = [128, 128, 64, 8],
           cnn_stride_list = [1, 1, 1, 1],
           cnn_padding_list = [0, 0, 0, 0],
           max_pool_filter_size_list = [0, 2, 4, 6], # use 0 to skip
           max_pool_stride_size_list = [2, 2, 4, 6], # use 0 to skip
           FC_hidden_dims = [512, 256, 128],
           output_dim = 7, 
           dropout = 0.3
        )
    model = model.to(config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model created with {num_params:,} trainable parameters")
    
    return model