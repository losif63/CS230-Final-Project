'''
Created on Dec 3, 2025

    Part of CS230 final project - module created to look more closely where the error on the test dataset comes from. 

@author: Sebastian prepelita
'''
from src.CNN import config
from src.CNN import dataset
from src.CNN import plotting
from src.CNN import history_io

import matplotlib.pyplot as plt

import sys
import time, datetime
import numpy as np
import scipy

from typing import Dict, Optional, Union
import logging
from src.CNN.cnn_model import CNNModelParams
from src.CNN.cnn_2d_model import CNNModel2DParams

def plot_test_results_error_analysis(
    avg_metrics: Dict[str, float],
    history_per_sample: Dict[str, np.ndarray],
    model_params: Optional[Union[CNNModelParams, CNNModel2DParams]] = None,
    save_path: str = 'test_results.png',
    N_frames_test: int = None,
    positional_vector: np.ndarray = None,
    orientation_change_vector: np.ndarray = None,
    logger: Optional[logging.Logger] = None
):
    """
    Plot test evaluation results with average metrics and per-sample history.
    
    Copy of function plot_test_results() from plotting.py, but adapted for error analysis.
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid: 2 rows x 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.13, height_ratios=[1, 2])

    # ========================================================================
    # Row 1, Left: Average Test Metrics (Text Display)
    # ========================================================================
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.axis('off')

    metrics_text = "Average Test Metrics\n" + "="*40 + "\n\n"

    if 'loss' in avg_metrics:
        metrics_text += f"Loss: {avg_metrics['loss']:.6f}\n\n"

    metrics_text += "Position Metrics:\n"
    if 'position_mse' in avg_metrics:
        metrics_text += f"  MSE: {avg_metrics['position_mse']:.6f} [m]\n"
    if 'position_mae' in avg_metrics:
        metrics_text += f"  MAE: {avg_metrics['position_mae']:.6f} [m]\n"

    metrics_text += "\nRotation Metrics:\n"
    if 'rotation_mse' in avg_metrics:
        metrics_text += f"  Quaternion MSE: {avg_metrics['rotation_mse']:.6f}\n"
    if 'rotation_mae' in avg_metrics:
        metrics_text += f"  Quaternion MAE: {avg_metrics['rotation_mae']:.6f}\n"
    if 'angular_error_deg' in avg_metrics:
        metrics_text += f"  Angular Error: {avg_metrics['angular_error_deg']:.2f}°\n"

    # Get number of samples
    num_samples = len(next(iter(history_per_sample.values())))
    metrics_text += f"\nTotal Samples: {num_samples}"

    ax_metrics.text(0.05, 1.1, metrics_text,
                   transform=ax_metrics.transAxes,
                   fontsize=11,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ========================================================================
    # Row 1, Right: Model Parameters (Text Display)
    # ========================================================================
    ax_params = fig.add_subplot(gs[0, 1])
    ax_params.axis('off')

    if model_params is not None:
        params_text = "Model Parameters\n" + "="*40 + "\n\n"
        params_text += f"Model Name: {model_params.model_name}\n"
        params_text += f"Input Channels: {model_params.n_channels}\n"
        params_text += f"Samples/Frame: {model_params.samples_per_frame}\n\n"

        params_text += "CNN Architecture:\n"
        params_text += f"  Filters: {model_params.cnn_num_filter_list}\n"
        params_text += f"  Kernel Sizes: {model_params.cnn_filter_size_list}\n"
        params_text += f"  Strides: {model_params.cnn_stride_list}\n"
        params_text += f"  Padding: {model_params.cnn_padding_list}\n"
        params_text += f"  MaxPool K: {model_params.max_pool_filter_size_list}\n"
        params_text += f"  MaxPool S: {model_params.max_pool_stride_size_list}\n\n"

        params_text += "FC Architecture:\n"
        params_text += f"  Hidden Dims: {model_params.FC_hidden_dims}\n"
        params_text += f"  Output Dim: {model_params.output_dim}\n"
        params_text += f"  Dropout: {model_params.dropout}"

        ax_params.text(0.05, 1.2, params_text,
                      transform=ax_params.transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    else:
        ax_params.text(0.5, 0.5, "No Model Parameters\nProvided",
                      transform=ax_params.transAxes,
                      fontsize=12,
                      ha='center',
                      va='center',
                      style='italic')

    # ========================================================================
    # Row 2, Left: Position MSE per Sample
    # ========================================================================
    ax_pos = fig.add_subplot(gs[1, 0])
    
    if 'position_mse' in history_per_sample:
        sample_indices = np.arange(len(history_per_sample['position_mse']))
        position_mse = history_per_sample['position_mse']
        num_samples = len(position_mse)

        ax_pos.plot(sample_indices, position_mse,
                   color='blue', linewidth=0.8, alpha=0.7, label = 'Test Pos Error')

        # Add average line
        avg_pos_mse = avg_metrics.get('position_mse', np.mean(position_mse))
        # ax_pos.axhline(y=avg_pos_mse, color='k', linestyle='--',
        #               linewidth=2, label=f'Average Euclidian Error: {avg_pos_mse:.6f} [m]')

        ax_pos.set_xlabel('Frame Number', fontsize=14)
        ax_pos.set_ylabel('Euclidian [m]', fontsize=14)
        ax_pos.set_title('Test set: Euclidian/Position "MSE" per Audio Frame', fontsize=12, fontweight='bold')
        ax_pos.set_xlim(0, num_samples)
        ax_pos.legend(loc='upper right')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.yaxis.set_label_coords(-0.05, 0.5)
    else:
        ax_pos.text(0.5, 0.5, "No Position MSE\nData Available",
                   transform=ax_pos.transAxes,
                   fontsize=12, ha='center', va='center', style='italic')

    # ========================================================================
    # Row 2, Right: Angular Error per Sample
    # ========================================================================
    #ax_ang = fig.add_subplot(gs[1, 1])
    # Row 2, Left: subdivide into two stacked plots
    sub_gs = gs[1, 1].subgridspec(2, 1, hspace=0.01)
    ax_ang   = fig.add_subplot(sub_gs[0])  # Position MSE
    ax_extra = fig.add_subplot(sub_gs[1])  # Extra plot (e.g. positional change)

    if 'angular_error_deg' in history_per_sample:
        sample_indices = np.arange(len(history_per_sample['angular_error_deg']))
        angular_error = history_per_sample['angular_error_deg']
        num_samples = len(angular_error)

        ax_ang.plot(sample_indices, angular_error,
                   color='red', linewidth=0.8, alpha=0.7, label = 'Test Rot Error')

        # Add average line
        avg_ang_error = avg_metrics.get('angular_error_deg', np.mean(angular_error))
        # ax_ang.axhline(y=avg_ang_error, color='k', linestyle='--',
        #               linewidth=2, label=f'Average Angular error: {avg_ang_error:.2f}°')

        #ax_ang.set_xlabel('Frame Number', fontsize=11)
        ax_ang.set_ylabel('Angular Error [°]', fontsize=14)
        ax_ang.set_title('Test set: Angular Error per Audio Frame', fontsize=12, fontweight='bold')
        ax_ang.set_xlim(0, num_samples)
        ax_ang.set_ylim(0, 60)
        ax_ang.legend(loc='upper right', fontsize=16)
        ax_ang.grid(True, alpha=0.3)
        ax_ang.set_xticklabels([])
    else:
        ax_ang.text(0.5, 0.5, "No Angular Error\nData Available",
                   transform=ax_ang.transAxes,
                   fontsize=12, ha='center', va='center', style='italic')

    # Add main title
    fig.suptitle('Test Results Analysis', fontsize=14, fontweight='bold', y=0.98)

    # Save the figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if logger:
        save_msg = f"      Test results plot saved to: {save_path}"
        logger.debug(save_msg)
        
    #ax_pos.plot(np.arange(N_frames_test), positional_change_vector, color='cyan', linewidth=1.8, alpha=0.7, linestyle = '--', label = "Positional change [m]")
    ax_pos.plot(np.arange(N_frames_test), positional_vector - np.average(positional_vector), color='green', alpha=1.0, linestyle = '-', label = r"Position - $\mu_{\mathrm{Position}}$ [m]")
    ax_pos.legend(loc='best', fontsize=16)
    
    ax_ang.plot(np.arange(N_frames_test), orientation_change_vector, label = 'Orientation change [°]', alpha = 0.6 )
    leg = ax_ang.legend(loc='best', fontsize=14)
    leg.set_alpha(0.3)

    ax_extra.plot(np.arange(N_frames_test), positional_vector - np.average(positional_vector), label = r"Position - $\mu_{\mathrm{Position}}$ [m]", color = 'g', alpha = 1.0)
    #ax_extra.plot(np.arange(N_frames_test), test_dataset.all_is_upside_down, label = "Is Upside Down", color = 'r', alpha = 1.0)
    
    ax_extra.set_xlabel('Frame Number', fontsize=14)
    ax_extra.set_ylabel('Position [m]', fontsize=14)
    ax_extra.yaxis.set_label_coords(-0.08, 0.5)
    ax_extra.set_xlim(0, N_frames_test)
    ax_extra.legend(loc='upper right', fontsize=16)
    ax_extra.grid(True, alpha=0.3)

    return ax_pos, ax_ang, ax_extra

def get_quaternion_angular_difference(pred_quat: np.ndarray, target_quat: np.ndarray) -> float:
    '''
    Function is a copy of metrics.compute_metrics() version on torch.Tensor, but this is on numpy.ndarray
    
    :param pred_quat: predicted quaternion (4D).
    :param target_quat: target quaternion (4D).
    '''
    assert len(pred_quat) == 4
    assert len(target_quat) == 4
    pred_quat_norm = pred_quat / np.linalg.norm(pred_quat, axis=-1, keepdims=True)
    target_quat_norm = target_quat / np.linalg.norm(target_quat, axis=-1, keepdims=True) 
    dot_product = np.sum(pred_quat_norm * target_quat_norm, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angular_error_rad = 2.0 * np.arccos(np.abs(dot_product))
    return np.rad2deg(angular_error_rad)

def get_positional_difference(pred_location_cart_m: np.ndarray, target_location_cart_m: np.ndarray) -> float:
    '''
    Function computes the distance between the two 3D Cartesian distances: pred_location_cart_m and target_location_cart_m.
    
    :param pred_location_cart_m: predicted 3D Cartesian location in meters.
    :param target_location_cart_m: target 3D Cartesian location in meters.
    '''
    assert len(pred_location_cart_m) == 3
    assert len(target_location_cart_m) == 3
    return np.linalg.norm(pred_location_cart_m-target_location_cart_m)

def do_error_analysis_CS230_report():
    print("  Retrieving history for model 03, 1D CNN 7d.")
    train_result_fn = r"C:\gDrive\Temp_papers\CS230_CNNs\testing_results_1D_100epochs_7D\03_sebastian_model_1__test_results.h5"
    # Test reading/writing:
    loaded_data, loaded_avg_metrics, loaded_metadata, loaded_model_params = (
        history_io.load_evaluation_results(filename=train_result_fn, logger=None)
    )
    print(f"  loaded history data: total length of {loaded_data['angular_error_deg'].shape}")
    
    
    print("Loading Test dataset...")
    test_dataset = dataset.AudioPoseDataset(
        data_root=str(config.Config.DATA_ROOT),
        session_ids=config.Config.TEST_SESSIONS,
        use_channels=config.Config.USE_CHANNELS,
        filter_silence=config.Config.FILTER_SILENCE,
        fs_audio=config.Config.FS_AUDIO,
        fs_head_tracking=config.Config.FS_HEAD_TRACKING,
        array_wearer_id=config.Config.ARRAY_WEARER_ID,
        cache_filename=config.Config.TEST_CACHE_FN,
        output_dim=7,
        apply_mag_and_gd=False,
        apply_spectograms_params=None,
        logger=None,
    )
    
    N_frames_test = len(test_dataset)
    assert N_frames_test == len(loaded_data['angular_error_deg'])
    print(f"Done... Loaded a total of {N_frames_test} frames.")
    
    start = time.perf_counter()
    positional_vector = np.zeros(N_frames_test)
    positional_change_vector = np.zeros(N_frames_test)
    orientation_change_vector = np.zeros(N_frames_test)
    for frame_idx, (audio_tensor, pose_tensor) in enumerate(test_dataset):
        pose_array = pose_tensor.numpy()
        if frame_idx == 0:
            prev_pose_array = pose_array
        angular_change_deg = get_quaternion_angular_difference(pred_quat = prev_pose_array[3:], target_quat = pose_array[3:])
        positional_change_deg = get_positional_difference(pred_location_cart_m = prev_pose_array[0:3], target_location_cart_m = pose_array[0:3])
        positional_change_vector[frame_idx] = positional_change_deg
        orientation_change_vector[frame_idx] = angular_change_deg
        positional_vector[frame_idx] = np.linalg.norm(pose_array[0:3])
        prev_pose_array = pose_array
        # print(f"#{frame_idx} --> pos error = {positional_change_deg} [m]; history = {loaded_data['position_mse'][frame_idx]}")
        # if frame_idx == 100:
        #     break
    print(f"   Building head change data took {datetime.timedelta(seconds=time.perf_counter()-start)}")
    
    # Get correlation coefficient:
    x_ = loaded_data['angular_error_deg']
    y_ = positional_vector - np.average(positional_vector)  
    r = np.corrcoef(x_, y_)[0,1]
    print(f"Pearson correlation coefficient rotation = {r}")
    x_ = loaded_data['position_mse']  
    r = np.corrcoef(x_, y_)[0,1]
    print(f"Pearson correlation coefficient pos = {r}")
    # Cross-correlation
    cross_corr = scipy.signal.correlate(x_, y_, mode='full')
    lags = np.arange(-len(x_)+1, len(x_))
    
    peak_lag = lags[np.argmax(np.abs(cross_corr))]
    peak_value = cross_corr[peak_lag]

    plt.figure(figsize=(10, 5))
    plt.title("Cross-correlation")
    plt.plot(lags, cross_corr, label='Cross-correlation')
    plt.axvline(0, color='k', linestyle='--', label='Zero lag')
    plt.axvline(peak_lag, color='r', linestyle='--', label=f'Peak lag = {peak_lag}')
    plt.scatter(peak_lag, peak_value, color='red')
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title('Cross-correlation between x and y')
    plt.legend()
    plt.grid(True)
    
    
    ###############################
    # Plotting:
    ################################
    print(loaded_model_params)
    loaded_model_params.model_name = "'M03 BASELINE 1D CNN'"
    plot_test_results_error_analysis(
        avg_metrics=loaded_avg_metrics,
        history_per_sample=loaded_data,
        model_params=loaded_model_params,
        save_path=None,
        N_frames_test = N_frames_test,
        positional_vector = positional_vector,
        orientation_change_vector = orientation_change_vector,
        logger=None,
    )
    
    plt.show()


if __name__ == '__main__':
    print("Error analysis CNN starting...")
    
    do_error_analysis_CS230_report()
    
    print("... Ending Error analysis CNN...")