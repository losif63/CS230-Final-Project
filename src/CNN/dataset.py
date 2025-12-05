"""
Part of training of CNN models, CS230 project, fall 2025.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import scipy
import torch
import h5py
import re
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import logging
import sys
import os

import time, datetime

import matplotlib.pyplot as plt

from src.CNN import data_utils
import src.baseline.config


def is_interactive_environment() -> bool:
    """
    Detect if we're running in an interactive environment (local terminal)
    or a non-interactive environment (SLURM, batch job, etc.).

    Returns
    -------
    bool
        True if interactive (show TQDM progress bars), False if non-interactive (disable TQDM)
    """
    if not sys.stderr.isatty():
        return False

    if any(
        var in os.environ
        for var in ["SLURM_JOB_ID", "SLURM_JOBID", "PBS_JOBID", "LSB_JOBID"]
    ):
        return False

    return True

def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding (CS230 official implementation).
    
    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d(integer) -- Encoding size
    
    Returns:
        angles -- (pos, d) numpy array 
    """
    # Get i from dimension span k
    i = k // 2
    # Calculate the angles using pos, i and d
    angles = pos / 10000**(2*i/d)
    return angles

def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings (CS230 official implementation).
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (d_model, position) A matrix with the positional encodings

    # Example usage: 
        N_max = 2400
        d = 1200
        pos_encoding = positional_encoding(positions = N_max, d = d)
        print (pos_encoding.shape)# (1200, 2400)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(18,12))
        plt.pcolormesh(pos_encoding, cmap='RdBu')
        plt.ylabel('d')
        plt.xlim((0, N_max))
        plt.xlabel('Position')
        plt.colorbar()
        plt.show()

    """
    # initialize a matrix angle_rads of all the angles
    pos = np.arange(positions)[:,None]
    k = np.arange(d)[None,:]
    angle_rads = get_angles(pos, k, d)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    #pos_encoding = angle_rads[np.newaxis, ...]
    
    return angle_rads.T

def _validate_spectogram_params(apply_spectograms_params: Optional[dict], fs_audio: int, fs_head_tracking: float) -> None:
    """
    Validate the spectogram parameters.
    
    Parameters
    ----------
    apply_spectograms_params : dict or None
        Dictionary with spectogram parameters
    fs_audio : int
        Audio sampling frequency
    fs_head_tracking : float
        Head tracking sampling frequency
        
    Raises
    ------
    ValueError
        If any of the parameters are invalid
    """
    if apply_spectograms_params is None:
        return
    
    if not isinstance(apply_spectograms_params, dict):
        raise ValueError(
            f"apply_spectograms_params must be a dictionary, but got {type(apply_spectograms_params)}"
        )
    
    # Check required keys (compute_on_gpu is optional with default False)
    required_keys = {"N_window", "hop_size", "apply_positional_encoding"}
    provided_keys = set(apply_spectograms_params.keys())
    
    if not required_keys.issubset(provided_keys):
        missing_keys = required_keys - provided_keys
        raise ValueError(
            f"apply_spectograms_params is missing required keys: {missing_keys}. "
            f"Required keys are: {required_keys}"
        )
    
    # Validate N_window
    N_window = apply_spectograms_params["N_window"]
    max_window_size = fs_audio / fs_head_tracking
    
    if not isinstance(N_window, int):
        raise ValueError(
            f"N_window must be an integer, but got {type(N_window)}"
        )
    
    if N_window <= 0:
        raise ValueError(
            f"N_window must be a positive integer, but got {N_window}"
        )
    
    if N_window >= max_window_size:
        raise ValueError(
            f"N_window must be smaller than fs_audio/fs_head_tracking = {max_window_size}, "
            f"but got N_window = {N_window}"
        )
    
    # Validate hop_size
    hop_size = apply_spectograms_params["hop_size"]
    max_hop_size = fs_audio / fs_head_tracking / 2
    
    if not isinstance(hop_size, int):
        raise ValueError(
            f"hop_size must be an integer, but got {type(hop_size)}"
        )
    
    if hop_size <= 0:
        raise ValueError(
            f"hop_size must be a positive integer, but got {hop_size}"
        )
    
    if hop_size >= max_hop_size:
        raise ValueError(
            f"hop_size must be smaller than fs_audio/fs_head_tracking/2 = {max_hop_size}, "
            f"but got hop_size = {hop_size}"
        )
    
    # Validate apply_positional_encoding
    apply_positional_encoding = apply_spectograms_params["apply_positional_encoding"]
    
    if not isinstance(apply_positional_encoding, bool):
        raise ValueError(
            f"apply_positional_encoding must be a boolean, but got {type(apply_positional_encoding)}"
        )
    
    # Validate compute_on_gpu (optional parameter with default False)
    if "compute_on_gpu" in apply_spectograms_params:
        compute_on_gpu = apply_spectograms_params["compute_on_gpu"]
        
        if not isinstance(compute_on_gpu, bool):
            raise ValueError(
                f"compute_on_gpu must be a boolean, but got {type(compute_on_gpu)}"
            )

def _compute_stft_valid_indices(n_samples: int, mfft: int, hop_size: int) -> tuple:
    """
    Compute the valid time indices for STFT truncation.
    
    When center=True is used in STFT, padding adds extra frames. This function
    computes which frames correspond to the actual audio duration [0, n_samples/fs].
    
    This matches the CPU version's _cached_valid_indices behavior.
    
    Parameters
    ----------
    n_samples : int
        Number of audio samples in the input signal
    mfft : int
        FFT size for STFT computation
    hop_size : int
        Hop size in samples
        
    Returns
    -------
    tuple
        (start_idx, end_idx) where:
        - start_idx: First valid time frame index
        - end_idx: Last valid time frame index (exclusive, for slicing)
    """
    # When center=True, PyTorch STFT pads by n_fft//2 on both sides
    # Total padded length: n_samples + 2*(n_fft//2) = n_samples + n_fft
    # Number of frames produced: ((n_samples + n_fft) - n_fft) // hop_size + 1 = n_samples // hop_size + 1
    
    # We want frames where time t is in [0, n_samples/fs]
    # Time for frame i: t[i] = (i * hop_size - n_fft//2) / fs
    # Valid frames: n_fft//(2*hop_size) <= i < (n_samples + n_fft//2) // hop_size
    
    start_idx = (mfft // 2 + hop_size - 1) // hop_size  # Ceiling division
    
    # Calculate actual number of frames produced by torch.stft
    actual_num_frames = n_samples // hop_size + 1
    
    # End index should not exceed actual number of frames
    # Theoretical end based on valid time range
    theoretical_end = (n_samples + mfft // 2) // hop_size
    end_idx = min(theoretical_end, actual_num_frames)
    
    return start_idx, end_idx


def _compute_spectogram_dimensions(samples_per_frame: int, hop_size: int, n_channels: int) -> tuple:
    """
    Compute the output dimensions for spectrograms - called only by the GPU implementation (safe to change 
    for CPU implementation).
    
    For GPU implementation without truncation, PyTorch STFT with center=True produces:
    time_bins = n_samples // hop_size + 1
    
    Parameters
    ----------
    samples_per_frame : int
        Number of audio samples per frame
    hop_size : int
        Hop size in samples
    n_channels : int
        Number of audio channels
        
    Returns
    -------
    tuple
        (mfft, freq_bins, time_bins) where:
        - mfft: FFT size for STFT computation
        - freq_bins: Number of frequency bins in output
        - time_bins: Number of time bins in output (no truncation)
    """
    mfft = _compute_mfft(samples_per_frame, hop_size)
    freq_bins = (mfft // 2) + 1  # One-sided FFT
    
    # For PyTorch STFT with center=True, output size is simply:
    time_bins = samples_per_frame // hop_size + 1
    
    return mfft, freq_bins, time_bins


def _create_spectogram_window(n_window: int, alpha: float = 0.34) -> np.ndarray:
    """
    Create the window function for spectrogram computation.
    
    Parameters
    ----------
    n_window : int
        Window size in samples
    alpha : float, default=0.34
        Tukey window parameter (0=rectangular, 1=Hann)
        
    Returns
    -------
    np.ndarray
        Window function of shape (n_window,)
    """
    return scipy.signal.windows.tukey(n_window, alpha=alpha)


def _create_spectogram_positional_encoding(freq_bins: int, time_bins: int) -> np.ndarray:
    """
    Create positional encoding for spectrograms.
    
    Parameters
    ----------
    freq_bins : int
        Number of frequency bins
    time_bins : int
        Number of time bins
        
    Returns
    -------
    np.ndarray
        Positional encoding of shape (1, freq_bins, time_bins) ready for broadcasting
    """
    pos_encoding = positional_encoding(positions=time_bins, d=freq_bins)
    return pos_encoding[np.newaxis, :, :]


def create_gpu_stft_params(
    n_window: int,
    hop_size: int,
    samples_per_frame: int,
    n_channels: int,
    apply_positional_encoding: bool,
    device: torch.device,
) -> dict:
    """
    Create GPU STFT parameters for use with apply_spectogram_batch_gpu().
    
    This function creates all necessary GPU tensors for on-the-fly spectrogram
    computation during training. All tensors are pre-allocated on the specified
    device to minimize overhead during training.
    
    Parameters
    ----------
    n_window : int
        Window size in samples (e.g., 240)
    hop_size : int
        Hop size in samples (e.g., 2)
    samples_per_frame : int
        Number of samples per audio frame (typically fs_audio / fs_head_tracking)
    n_channels : int
        Number of audio channels (used for dimension validation)
    apply_positional_encoding : bool
        Whether to include positional encoding
    device : torch.device
        Target device for tensors (e.g., torch.device('cuda:0'))
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'window': torch.Tensor on device, shape (n_window,)
        - 'hop_size': int
        - 'mfft': int
        - 'pos_encoding': torch.Tensor on device or None, shape (1, freq_bins, time_bins)
    
    See Also
    --------
    apply_spectogram_batch_gpu : GPU spectrogram computation function
    _compute_mfft : Helper to compute mfft parameter
    """
    # Compute dimensions
    mfft, freq_bins, time_bins = _compute_spectogram_dimensions(
        samples_per_frame, hop_size, n_channels
    )
    
    # Create window function (numpy) and convert to torch tensor on device
    window_np = _create_spectogram_window(n_window)
    window_tensor = torch.from_numpy(window_np).float().to(device)
    
    # Precompute window sum for normalization (optimization: avoid recomputing per batch)
    window_sum = float(window_tensor.sum().item())
    
    # Create positional encoding if needed
    if apply_positional_encoding:
        pos_encoding_np = _create_spectogram_positional_encoding(freq_bins, time_bins)
        pos_encoding_tensor = torch.from_numpy(pos_encoding_np).float().to(device)
    else:
        pos_encoding_tensor = None
    
    return {
        'window': window_tensor,
        'hop_size': hop_size,
        'mfft': mfft,
        'pos_encoding': pos_encoding_tensor,
        'window_sum': window_sum,
    }


def _compute_mfft(frame_size: int, hop_size: int) -> int:
    """
    Compute the mfft parameter for STFT based on frame size and hop size.
    
    This helper function ensures consistency in mfft calculation across all
    spectrogram-related functions.
    
    Parameters
    ----------
    frame_size : int
        Number of samples per frame (typically fs_audio / fs_head_tracking)
    hop_size : int
        Hop size in samples
        
    Returns
    -------
    int
        The mfft parameter for ShortTimeFFT
    """
    expected_width = frame_size // hop_size
    mfft = expected_width * 2
    return mfft

def compute_spectogram_freq_bins(apply_spectograms_params: dict, fs_audio: int, frame_size: int) -> int:
    """
    Compute the number of frequency bins in the spectrogram output.
    
    This is a general utility function that can be used to determine the input height 
    for 2D CNN models when using spectrogram features.
    
    Parameters
    ----------
    apply_spectograms_params : dict
        Dictionary with spectogram parameters containing:
        - "N_window": Window size in samples
        - "hop_size": Hop size in samples
        - "apply_positional_encoding": Boolean for positional encoding
    fs_audio : int
        Audio sampling frequency (Hz)
    frame_size : int
        Number of audio samples per frame (typically fs_audio / fs_head_tracking)
        
    Returns
    -------
    int
        Number of frequency bins in the spectrogram output
        
    Notes
    -----
    For a real-valued signal with one-sided FFT, the number of frequency bins is:
    freq_bins = (mfft // 2) + 1
    
    where mfft is computed using _compute_mfft()
    
    Example
    -------
    >>> params = {"N_window": 240, "hop_size": 2, "apply_positional_encoding": False}
    >>> freq_bins = compute_spectogram_freq_bins(params, fs_audio=48000, frame_size=2400)
    >>> print(freq_bins)  # Should output 1201
    """
    hop_size = apply_spectograms_params["hop_size"]
    
    # Use shared helper function to compute mfft
    mfft = _compute_mfft(frame_size, hop_size)
    
    # For one-sided FFT of real signal, freq_bins = (mfft // 2) + 1
    freq_bins = (mfft // 2) + 1
    
    return freq_bins

class AudioPoseDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        session_ids: List[int],
        use_channels: List[int] = [0, 1, 2, 3, 4, 5],
        filter_silence: bool = True,
        fs_audio: int = 48000,
        fs_head_tracking: float = 20.0,
        array_wearer_id: int = 2,
        cache_filename: str = None,
        output_dim: int = 7,
        apply_mag_and_gd: bool = False,
        apply_spectograms_params: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Inputs:
            data_root: Root directory of EasyCom dataset
            session_ids: List of session IDs to include (e.g., [1,2,3,...,10])
            use_channels: Which audio channels to use (0-5 for 6-channel array)
            filter_silence: If True, only include frames with active speech
            fs_audio: Audio sampling frequency
            fs_head_tracking: Head tracking sampling frequency
            array_wearer_id: Participant ID of glasses wearer
            cache_filename: String - if not None, the dataset is cached to/from an .hdf5 file.
            output_dim: Output dimension (3=position only, 4=rotation only, 7=full 6DOF)
            apply_mag_and_phase: If True, apply log-magnitude and group-delay to the audio data
            apply_spectograms_params: If not None, apply spectrograms to the audio data.
                The input should be a dictionary with the following keys:
                    "N_window": Window size in samples
                    "hop_size": Hop size in samples
                    "apply_positional_encoding": Boolean - if True, apply positional encoding to the spectrograms
            logger: Logger instance for debug-level logging
        """
        self.use_channels = use_channels
        self.filter_silence = filter_silence
        if output_dim not in [3, 4, 7]:
            raise ValueError(
                f"AudioPoseDataset() ERROR: Output dimension must be 3, 4, or 7, but got {output_dim}!"
            )
        # Check that only one of apply_mag_and_gd and apply_spectograms_params is set:
        if apply_mag_and_gd and apply_spectograms_params is not None:
            raise ValueError(
                "AudioPoseDataset() ERROR: Only one of apply_mag_and_gd and apply_spectograms_params can be set!"
            )
        self.output_dim = output_dim
        self.logger = logger
        self.loader = data_utils.EasyComDataLoader(
            data_root=data_root,
            fs_audio=fs_audio,
            fs_head_tracking=fs_head_tracking,
            array_wearer_id=array_wearer_id,
        )
        
        # Validate spectogram parameters if provided
        _validate_spectogram_params(
            apply_spectograms_params=apply_spectograms_params,
            fs_audio=fs_audio,
            fs_head_tracking=fs_head_tracking,
        )
        self.apply_spectograms_params = apply_spectograms_params
        
        # Initialize cached spectogram computation objects if needed
        if self.apply_spectograms_params is not None:
            self._initialize_spectogram_cache()
        
        self.session_ids = session_ids
        self.samples = []
        self.cache_filename = Path(cache_filename).stem
        self.apply_mag_and_gd = apply_mag_and_gd

        self._load_cached_dataset()

        dataset_built_msg = f"Built dataset with {len(self.all_audio_frames)} samples"
        if self.logger:
            self.logger.debug(dataset_built_msg)
        else:
            print(f" {dataset_built_msg}")

    def _load_cached_dataset(self):
        # Check to see if data is cached:
        if self.cache_filename is not None:
            fn_ = (
                Path(self.cache_filename).stem
                + "_S"
                + str(self.session_ids[0])
                + "_S"
                + str(self.session_ids[-1])
                + ".hdf5"
            )
            fn_ = self.loader.data_root / fn_
        else:
            fn_ = None
        if fn_ is not None and fn_.exists():
            print(f" Dataset cached file {fn_} found - loading it...")
            with h5py.File(fn_, "r") as f:
                root_groups = [
                    name for name in f.keys() if isinstance(f[name], h5py.Group)
                ]
                session_ids = []
                file_filter_silence = f["filter_silence"][...]
                if not file_filter_silence == self.filter_silence:
                    raise ValueError(
                        f"Cached hdf5 file has filtered silence to {file_filter_silence}, but the dataset needs to have filtered silence to {self.filter_silence}.\nb"
                        "Use different filename for cache or change filter_silence input."
                    )
                for group_string in root_groups:
                    match = re.search(r"session_(\d+)", group_string)
                    if match:
                        num = int(match.group(1))
                    else:
                        raise ValueError(
                            rf"Unexpected group type: expected session_\d+, but got {group_string}"
                        )
                    session_ids.append(num)
                if set(session_ids) != set(self.session_ids):
                    raise ValueError(
                        f"The cached database contains different sessions: {session_ids} . You requested sessions: {self.session_ids}"
                    )
                audio_arrays_sessions = []
                wearer_pose_arrays_sessions = []
                for group_string in root_groups:
                    audio_frames_session = f[f"{group_string}/audio_frames"][...]
                    wearer_pose_session = f[f"{group_string}/wearer_pose"][...]
                    if "is_upside_down" in f[f"{group_string}"].keys():
                        is_upside_down_array_sessions = f[f"{group_string}/is_upside_down"][...]
                    else:
                        is_upside_down_array_sessions = None
                    audio_arrays_sessions.append(audio_frames_session)
                    wearer_pose_arrays_sessions.append(wearer_pose_session)
        else:
            print(f" Cached file missing {fn_}! Building dataset manually!")
            audio_arrays_sessions, wearer_pose_arrays_sessions, is_upside_down_array_sessions, session_ids = (
                self._build_dataset()
            )

            # dump data:
            with h5py.File(fn_, "w") as f:
                f.create_dataset("filter_silence", data=self.filter_silence)
                for idx, session_id in enumerate(session_ids):
                    session_group = f.create_group(f"session_{session_id}")
                    session_group.create_dataset(
                        "audio_frames", data=audio_arrays_sessions[idx]
                    )
                    session_group.create_dataset(
                        "wearer_pose", data=wearer_pose_arrays_sessions[idx]
                    )
                    session_group.create_dataset(
                        "is_upside_down", data=is_upside_down_array_sessions[idx]
                    )

        self.all_audio_frames = np.concatenate(audio_arrays_sessions, axis=0)
        self.all_wearer_pose_6dof = np.concatenate(wearer_pose_arrays_sessions, axis=0)
        if is_upside_down_array_sessions is not None:
            self.all_is_upside_down = np.concatenate(is_upside_down_array_sessions, axis=0)
        assert self.all_audio_frames.shape[0] == self.all_wearer_pose_6dof.shape[0]

        print(f"   Loaded audio data: {self.all_audio_frames.shape}.")
        print(f"   Loaded wearer data: {self.all_wearer_pose_6dof.shape}.")
        
        # Pre-compute spectrograms if needed (for faster training)
        # WARNING: This can use MASSIVE amounts of memory for large datasets!
        # DISABLED by default - requires too much RAM (80TB+)
        # if self.apply_spectograms_params is not None:
        #     self._precompute_spectrograms()

    def _initialize_spectogram_cache(self):
        """
        Initialize and cache spectogram computation objects for efficiency.
        This is called once during __init__ if apply_spectograms_params is not None.
        
        When compute_on_gpu=True, this only stores basic parameters needed for
        GPU computation. The actual GPU tensors are created later using
        create_gpu_stft_params().
        
        When compute_on_gpu=False (default), this creates the full CPU-based
        scipy.signal.ShortTimeFFT objects and caches for efficient CPU computation.
        """
        # Guard check - should not be called if params are None
        if self.apply_spectograms_params is None:
            raise ValueError("_initialize_spectogram_cache called with None params")
        
        N_window = self.apply_spectograms_params["N_window"]
        hop_size = self.apply_spectograms_params["hop_size"]
        apply_positional_encoding = self.apply_spectograms_params["apply_positional_encoding"]
        compute_on_gpu = self.apply_spectograms_params.get("compute_on_gpu", False)
        
        # Calculate expected dimensions based on samples per frame
        samples_per_frame = int(self.loader.FS_AUDIO / self.loader.FS_HEAD_TRACKING)
        
        if compute_on_gpu:
            # GPU mode: Only store basic parameters - actual GPU tensors created later
            # by create_gpu_stft_params() which will be called before training
            self._cached_SFT = None
            self._cached_valid_indices = None
            self._cached_pos_encoding = None
            
            if self.logger:
                self.logger.debug(
                    f"      Initialized for GPU mode - CPU STFT objects skipped. "
                    f"Use create_gpu_stft_params() before training."
                )
        else:
            # CPU mode: Create full scipy STFT objects and caches
            # Create and cache the window function using helper
            window_ = _create_spectogram_window(N_window)
            
            # Use shared helper function to compute mfft
            mfft_ = _compute_mfft(samples_per_frame, hop_size)
            
            # Create and cache the ShortTimeFFT object
            self._cached_SFT = scipy.signal.ShortTimeFFT(
                window_, 
                hop=hop_size, 
                fs=self.loader.FS_AUDIO, 
                mfft=mfft_,
                scale_to='magnitude',  # |spectrum| not |spectrum|^2
                fft_mode='onesided',
                phase_shift=None
            )
            
            # Pre-compute and cache valid indices
            time_spectogram = self._cached_SFT.t(samples_per_frame)
            self._cached_valid_indices = np.where(
                (time_spectogram >= 0.0) & (time_spectogram <= samples_per_frame / self.loader.FS_AUDIO)
            )[0]
            
            # Pre-compute and cache positional encoding if needed
            if apply_positional_encoding:
                # We need to know the output shape to compute positional encoding
                # Create a dummy STFT to get the shape
                dummy_audio = np.zeros((len(self.use_channels), samples_per_frame))
                dummy_stft = self._cached_SFT.stft(dummy_audio, axis=-1)
                freq_bins = dummy_stft.shape[1]
                time_bins = len(self._cached_valid_indices)
                
                # Compute and cache positional encoding with newaxis already applied
                # Shape: (1, freq_bins, time_bins) ready for broadcasting
                pos_encoding = positional_encoding(positions=time_bins, d=freq_bins)
                self._cached_pos_encoding = pos_encoding[np.newaxis, :, :]
            else:
                self._cached_pos_encoding = None

    def _precompute_spectrograms(self):
        """
        Pre-compute all spectrograms and store them in memory.
        This trades memory for speed - spectrograms are computed once during initialization
        instead of on-the-fly during training.
        
        This is called at the end of _load_cached_dataset() if apply_spectograms_params is not None.
        """
        print("\n" + "="*80)
        print("PRE-COMPUTING SPECTROGRAMS FOR FASTER TRAINING")
        print("="*80)
        
        n_samples = len(self.all_audio_frames)
        
        # Compute the output shape by processing one sample
        dummy_spec = self._apply_spectogram(self.all_audio_frames[0, :, :])
        spec_shape = dummy_spec.shape  # (n_channels, freq_bins, time_bins)
        
        print(f"  Total samples: {n_samples}")
        print(f"  Spectrogram shape per sample: {spec_shape}")
        print(f"  Raw audio shape per sample: {self.all_audio_frames.shape[1:]}")
        
        # Calculate memory requirements
        spec_memory_mb = (n_samples * np.prod(spec_shape) * 4) / (1024**2)  # 4 bytes per float32
        audio_memory_mb = (n_samples * np.prod(self.all_audio_frames.shape[1:]) * 4) / (1024**2)
        
        print(f"  Memory for raw audio: {audio_memory_mb:.1f} MB")
        print(f"  Memory for spectrograms: {spec_memory_mb:.1f} MB")
        print(f"  Memory increase: {spec_memory_mb - audio_memory_mb:.1f} MB")
        print(f"\nComputing spectrograms...")
        
        # Pre-allocate array for all spectrograms
        self.all_spectrograms = np.zeros(
            (n_samples, *spec_shape), 
            dtype=np.float32
        )
        
        # Compute spectrograms with progress bar
        start_time = time.perf_counter()
        for idx in tqdm(
            range(n_samples), 
            desc="  Pre-computing spectrograms",
            disable=not is_interactive_environment()
        ):
            self.all_spectrograms[idx] = self._apply_spectogram(self.all_audio_frames[idx, :, :])
        
        elapsed = time.perf_counter() - start_time
        
        print(f"  ✓ Pre-computation complete in {datetime.timedelta(seconds=elapsed)}")
        print(f"  ✓ Speed: {n_samples/elapsed:.1f} samples/second")
        print(f"  ✓ Stored spectrograms shape: {self.all_spectrograms.shape}")
        print("="*80 + "\n")

    def _build_dataset(self):
        samples_per_frame = int(self.loader.FS_AUDIO / self.loader.FS_HEAD_TRACKING)

        audio_arrays_sessions = []
        wearer_pose_arrays_sessions = []
        is_upside_down_array_sessions = []
        session_ids = []
        for session_id in tqdm(
            self.session_ids,
            desc="Loading sessions",
            disable=not is_interactive_environment(),
        ):
            session_dir = self.loader.get_session_dir(session_id)

            if not session_dir.exists():
                print(f" Session {session_id} not found, skipping... !")
                continue

            wav_files = self.loader.get_wav_files(session_dir)

            audio_frames_session_list = []
            wearer_pose_6dof_session_list = []
            is_upside_down_session_list = []
            for wave_file_idx, wav_file in enumerate(wav_files):
                try:
                    audio, _ = self.loader.load_audio(wav_file)

                    if audio is None:
                        raise ValueError(f"Got no audio for file {wav_file}!")

                    # Remove DC global:
                    audio_DC = np.average(audio, axis=0)
                    audio -= audio_DC

                    n_samples, n_channels = audio.shape

                    if max(self.use_channels) >= n_channels:
                        raise ValueError(
                            f"Requesting too many channels: {self.use_channels}. Audio has {n_channels} channels only!"
                        )

                    poses_data = self.loader.load_tracked_poses(session_id, wav_file)
                    t_max = (1.0 * len(audio) - 1) / self.loader.FS_AUDIO
                    expected_N_frames_head_tracking = int(
                        round(t_max * self.loader.FS_HEAD_TRACKING)
                    )
                    if (
                        poses_data is None
                        or len(poses_data) != expected_N_frames_head_tracking
                    ):
                        raise ValueError(f"Poses loading error!!")

                    wearer_pose_6dof, is_upside_down = self.loader.extract_wearer_6dof(poses_data)
                    if wearer_pose_6dof is None:
                        raise ValueError("Wearer pose extraction returned None!")

                    n_frames = len(wearer_pose_6dof)

                    # Initialize defaults to avoid possibly-unbound variables
                    speech_lookup = {}
                    external_participant_ids = []

                    if self.filter_silence:
                        transcription_data = self.loader.load_speech_transcriptions(
                            session_id, wav_file
                        )
                        speech_lookup = self.loader.create_speech_lookup(
                            transcription_data, n_frames
                        )
                        participant_ids = self.loader.get_all_participant_ids(
                            poses_data
                        )
                        external_participant_ids = [
                            pid
                            for pid in participant_ids
                            if pid != self.loader.ARRAY_WEARER_ID
                        ]

                    for frame_idx in range(n_frames):
                        start_sample = frame_idx * samples_per_frame
                        end_sample = (frame_idx + 1) * samples_per_frame

                        if end_sample > n_samples:
                            raise ValueError("Out of bounds error!")

                        if self.filter_silence:
                            is_active = any(
                                speech_lookup[pid][frame_idx]
                                for pid in external_participant_ids
                            )
                            if not is_active:
                                continue

                        if np.linalg.norm(wearer_pose_6dof[frame_idx]) < 0.1:
                            raise ValueError("To small norm?!")

                        # Cache audio frame
                        audio_frame = audio[
                            start_sample:end_sample, self.use_channels
                        ].T
                        audio_frame = audio_frame.astype(np.float32)

                        # Remove DC (?):
                        # audio_frame_DC = np.average(audio_frame, axis = 1)
                        # audio_frame -= audio_frame_DC[:,None]

                        # We'll use self.all_audio_frames and self.all_wearer_pose_6dof as cached data instead
                        # self.samples.append({
                        #     'audio_frame': audio_frame,  # (n_channels, 2400)
                        #     'pose_6dof': wearer_pose_6dof[frame_idx].astype(np.float32)
                        # })
                        audio_frames_session_list.append(
                            audio_frame[None, :, :]
                        )  # n_frames x (n_channels, 2400)
                        wp_ = wearer_pose_6dof[frame_idx].astype(np.float32)
                        wearer_pose_6dof_session_list.append(
                            wp_[None, :]
                        )  # n_frames x (7)
                        iud_ = np.array([is_upside_down[frame_idx].astype(np.float32)])
                        is_upside_down_session_list.append( iud_[None, :] )

                except Exception as e:
                    raise ValueError(f"\n Error processing {wav_file.name}: {e}")

            audio_frames_session_array = np.concatenate(
                audio_frames_session_list, axis=0
            )
            wearer_pose_session_array = np.concatenate(
                wearer_pose_6dof_session_list, axis=0
            )
            is_upside_down_session_array = np.concatenate(
                is_upside_down_session_list, axis=0
            )
            

            # Build big list with all sessions:
            audio_arrays_sessions.append(audio_frames_session_array)
            wearer_pose_arrays_sessions.append(wearer_pose_session_array)
            is_upside_down_array_sessions.append(is_upside_down_session_array)
            session_ids.append(session_id)
        self.loader.print_stats()
        return audio_arrays_sessions, wearer_pose_arrays_sessions, is_upside_down_array_sessions, session_ids

    def _apply_fft_transform(self, audio: np.ndarray) -> np.ndarray:
        """
        Transform raw audio (6, 2400) to FFT magnitude in dB scale (6, 2, freq_bins).

        Normalizes magnitude to [-1, 1] range where:
        - -1.0 represents silence/low energy (-80 dB)
        - 1.0 represents high energy (+20 dB)

        Returns
        -------
        np.ndarray
            Shape (6, 2, freq_bins) where dim 1 contains [mag_db_normalized, mag_db_normalized]

        Notes
        -----
        FFT magnitude dB scale is different from waveform dB:
        - FFT magnitudes can exceed 0 dB (unlike normalized waveforms)
        - Typical range for this dataset: -80 dB (silence) to +20 dB (strong frequency component)
        - Total range: 100 dB
        """
        N_taps = audio.shape[1]
        N_channels = audio.shape[0]
        # Compute FFT (rfft for real signals - more efficient)
        fft_result = np.fft.rfft(audio, axis=-1)  # (6, freq_bins)

        # Magnitude in dB scale
        mag_db = 20 * np.log10(np.abs(fft_result) + 1e-10)

        # Clip to observed data range: -80 dB (silence) to +20 dB (strong components)
        # This is a 100 dB range, typical for our dataset:
        mag_db = np.clip(mag_db, -80.0, 25.0)

        # Normalize to [-1, 1] range
        # Maps: -80 dB -> -1.0, +25 dB -> 1.0
        mag_db_normalized = 2.0 * ((mag_db + 80.0) / 105.0) - 1.0

        # Get group delay:
        dT = 1.0 / self.loader.FS_AUDIO
        time_vector = np.arange(N_taps) * dT  # Shape: (2400,)

        # Broadcast time_vector to multiply with each channel
        # time_vector needs shape (1, 2400) to broadcast with audio (6, 2400)
        time_vector_broadcast = time_vector[np.newaxis, :]  # Shape: (1, 2400)

        # Now multiply: (1, 2400) * (6, 2400) -> (6, 2400)
        group_delay_seconds = np.real(
            np.fft.rfft(time_vector_broadcast * audio, axis=-1) / fft_result
        )
        gp_delay_channel_1 = np.real(
            np.fft.rfft(time_vector * audio[3, :]) / np.fft.rfft(audio[3, :])
        )
        assert np.all(group_delay_seconds[3, :] == gp_delay_channel_1)
        # Clip and normalize to [-1, 1] range:
        frame_size_seconds = N_taps * dT
        group_delay_seconds = np.clip(
            group_delay_seconds, -3.0 * frame_size_seconds, 3.0 * frame_size_seconds
        )
        group_delay_normalized = group_delay_seconds / (3.0 * frame_size_seconds)

        # Duplicate magnitude to maintain (6, 2, freq_bins) shape
        # Both channels contain the same dB magnitude data
        output = np.stack(
            [mag_db_normalized, group_delay_normalized], axis=1
        )  # (6, 2, freq_bins)

        # Option B: Magnitude + Phase
        # magnitude = np.abs(fft_result)
        # phase = np.angle(fft_result)
        # output = np.stack([magnitude, phase], axis=1)

        return output.astype(np.float32)

    def _apply_spectogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Transform raw audio (n_channels, n_samples) to spectrogram magnitude (n_channels, freq_bins, time_bins).
        
        Uses cached SFT object, valid_indices, and positional encoding for efficiency.

        These are intialized during __init__() call, inside _initialize_spectogram_cache() function.
        
        Parameters
        ----------
        audio : np.ndarray
            Input audio of shape (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            Shape (n_channels, freq_bins, time_bins) - 2D spectrogram for each channel
        """
        # Apply STFT to all channels at once using cached SFT object
        Sx_all = self._cached_SFT.stft(audio, axis=-1)  # Shape: (n_channels, freq_bins, time_bins)
        
        # Take absolute value and select valid time indices using cached valid_indices
        output = np.abs(Sx_all[:, :, self._cached_valid_indices])  # Shape: (n_channels, freq_bins, time_bins)
        # Clip to observed data range: -80 dB (silence) to +35 dB (strong components)
        # output = np.clip(20.0*np.log10(output+1e-10), -80.0, 35.0)
        # output = 2.0 * ((output + 80.0) / (80.0+35.0)) - 1.0
        
        # Apply cached positional encoding if it was pre-computed
        # The encoding is already stored with shape (1, freq_bins, time_bins) for broadcasting
        if self._cached_pos_encoding is not None:
            output = output * self._cached_pos_encoding
        
        return output.astype(np.float32)
        

    def __len__(self) -> int:
        assert self.all_audio_frames.shape[0] == self.all_wearer_pose_6dof.shape[0]
        return len(self.all_audio_frames)

    def __getitem__(self, idx: int):
        # Three options for spectrogram processing:
        # Option 1: Return raw audio for GPU spectrogram computation (fastest - use with apply_spectogram_batch_gpu)
        # Option 2: Compute spectrograms on CPU on-the-fly (slower but memory-efficient)
        # Option 3: Use pre-computed spectrograms (fastest, but uses massive memory)
        
        if self.apply_mag_and_gd:
            audio = self._apply_fft_transform(self.all_audio_frames[idx, :, :])
        elif self.apply_spectograms_params is not None:
            # Check if we should compute on GPU (return raw audio) or CPU (compute here)
            compute_on_gpu = self.apply_spectograms_params.get("compute_on_gpu", False)
            
            if compute_on_gpu:
                # OPTION 1: Return RAW audio - spectrograms will be computed on GPU in training loop
                audio = self.all_audio_frames[idx, :, :]
            else:
                # OPTION 2: Compute spectrograms on CPU on-the-fly (SLOWER)
                audio = self._apply_spectogram(self.all_audio_frames[idx, :, :]) * 100.0
                
                # OPTION 3: Use pre-computed spectrograms (DISABLED since it requires TOO MUCH DATA - uncomment to use)
                # audio = self.all_spectrograms[idx] * 100.0
        else:
            audio = self.all_audio_frames[idx, :, :]
        audio_tensor = torch.from_numpy(audio)
        full_pose = self.all_wearer_pose_6dof[idx, :]

        if self.output_dim == 3:
            pose_tensor = torch.from_numpy(full_pose[:3])
        elif self.output_dim == 4:
            pose_tensor = torch.from_numpy(full_pose[3:])
        else:
            pose_tensor = torch.from_numpy(full_pose)

        return audio_tensor, pose_tensor


def create_dataloaders(
    config: src.baseline.config.Config,
    output_dim: int = 7,
    apply_mag_and_gd: bool = False,
    apply_spectograms_params: Optional[dict] = None,
    is_parallel_training: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    Create train, validation, and test dataloaders.

    Parameters
    ----------
    config : Config
        Configuration object with dataset parameters
    apply_mag_and_gd: bool
        If True, apply log-magnitude and group-delay to the audio data
    apply_spectograms_params : dict or None
        If not None, apply spectrograms to the audio data.
        Dictionary with keys: "N_window", "hop_size", "apply_positional_encoding"
    output_dim : int
        Output dimension (3=position only, 4=rotation only, 7=full 6DOF). Note the data written in cached files in AudioPoseDataset() stores all 7 values!
    is_parallel_training : bool
        If True, uses NUM_WORKERS_PARALLEL (0 or 1 or smaller) to avoid process explosion.
        If False, uses NUM_WORKERS_SEQUENTIAL (2-4) for better performance.
    logger : logging.Logger, optional
        Logger instance for debug-level logging

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    if logger:
        logger.debug("   (a) Creating training dataset...")

    start = time.perf_counter()
    train_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TRAIN_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID,
        cache_filename=config.TRAIN_CACHE_FN,
        output_dim=output_dim,
        apply_mag_and_gd=apply_mag_and_gd,
        apply_spectograms_params=apply_spectograms_params,
        logger=logger,
    )
    end = time.perf_counter()

    train_time_msg = (
        f"       Training dataset loading took {datetime.timedelta(seconds=end-start)}"
    )
    if logger:
        logger.debug(train_time_msg)

    if logger:
        logger.debug("   (b) Creating validation dataset...")

    val_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.VAL_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID,
        cache_filename=config.VAL_CACHE_FN,
        output_dim=output_dim,
        apply_mag_and_gd=apply_mag_and_gd,
        apply_spectograms_params=apply_spectograms_params,
        logger=logger,
    )

    if logger:
        logger.debug("   (c) Creating test dataset...")

    test_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TEST_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID,
        cache_filename=config.TEST_CACHE_FN,
        output_dim=output_dim,
        apply_mag_and_gd=apply_mag_and_gd,
        apply_spectograms_params=apply_spectograms_params,
        logger=logger,
    )

    dataset_sizes_msg = (
        f"Dataset sizes:\n"
        f"\t\t\t  Train: {len(train_dataset)} samples\n"
        f"\t\t\t  Val: {len(val_dataset)} samples\n"
        f"\t\t\t  Test: {len(test_dataset)} samples"
    )

    if logger:
        logger.debug(dataset_sizes_msg)

    if logger:
        logger.debug("   (d) Creating dataloaders...")

    # Select appropriate num_workers based on training mode
    # CRITICAL: Use fewer workers in parallel mode to avoid process explosion
    num_workers = config.NUM_WORKERS_PARALLEL if is_parallel_training else config.NUM_WORKERS_SEQUENTIAL
    
    if logger:
        mode_str = "PARALLEL" if is_parallel_training else "SEQUENTIAL"
        logger.debug(f"      Training mode: {mode_str}, using num_workers={num_workers}")

    # Create dataloaders
    # Use persistent_workers to keep worker processes alive and reduce overhead
    use_persistent = num_workers > 0  # Only use persistent workers if multiprocessing is enabled
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if logger:
        logger.debug("      Dataloaders created successfully. Returning...")

    return train_loader, val_loader, test_loader


def apply_spectogram_batch_gpu(
    audio_batch: torch.Tensor,
    window: torch.Tensor,
    hop_size: int,
    mfft: int,
    pos_encoding_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GPU-accelerated batch spectrogram computation using PyTorch's torch.stft().
    
    This function computes spectrograms for an entire batch on GPU, providing
    massive speedup over CPU-based scipy computation (30ms → <1ms per sample).
    
    IMPORTANT: This function truncates the STFT output to match the CPU version's
    behavior (using _cached_valid_indices). Only time bins corresponding to the
    actual audio frame duration are kept.
    
    Parameters
    ----------
    audio_batch : torch.Tensor
        Input audio batch of shape (batch_size, n_channels, n_samples)
        Should already be on the target device (GPU)
    window : torch.Tensor
        Pre-allocated window tensor on GPU of shape (n_window,)
        Should be created once and reused across batches for efficiency
    hop_size : int
        Hop size in samples (e.g., 2)
    mfft : int
        FFT size computed using _compute_mfft(samples_per_frame, hop_size)
        Must match the CPU computation for identical outputs
    pos_encoding_tensor : torch.Tensor, optional
        Pre-computed positional encoding of shape (1, freq_bins, time_bins)
        If provided, will be multiplied with the spectrogram
        Note: time_bins should match the truncated output size (samples_per_frame // hop_size)
        
    Returns
    -------
    torch.Tensor
        Spectrogram magnitudes of shape (batch_size, n_channels, freq_bins, time_bins)
        where time_bins = samples_per_frame // hop_size (truncated to valid range)
        
    Notes
    -----
    This function is designed to be called from a custom collate function or
    as a preprocessing layer in the model. It operates on entire batches for
    maximum GPU efficiency.
    
    For best performance:
    - Pre-allocate window tensor once and reuse across all batches
    - Pre-allocate positional encoding once if using it
    - Keep audio_batch on GPU to avoid transfers
    """
    batch_size, n_channels, n_samples = audio_batch.shape
    
    # Reshape to (batch_size * n_channels, n_samples) for batch processing
    audio_flat = audio_batch.reshape(batch_size * n_channels, n_samples)
    
    # Compute STFT using PyTorch (GPU-accelerated - CUDA cuFFT library)
    # Use mfft parameter to match CPU ShortTimeFFT computation
    # center=True adds padding of n_fft//2 on both sides
    stft_result = torch.stft(
        audio_flat,
        n_fft=mfft,
        hop_length=hop_size,
        win_length=window.shape[0],
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Take magnitude (equivalent to np.abs - element-wise CUDA kernel)
    magnitude = torch.abs(stft_result)
    
    # Note: Window normalization is applied in train.py together with 100x scaling
    # for efficiency (one operation instead of two)
    
    # Reshape back to (batch_size, n_channels, freq_bins, time_bins)
    freq_bins = magnitude.shape[1]
    time_bins = magnitude.shape[2]
    magnitude = magnitude.reshape(batch_size, n_channels, freq_bins, time_bins)
    
    # Note: No truncation needed for GPU version - PyTorch STFT with center=True 
    # produces the correct number of time bins (n_samples // hop_size + 1)
    # The CPU version uses _cached_valid_indices for filtering, but for GPU we use the full output
    
    # Apply positional encoding if provided, Element-wise multiplication (CUDA kernel)
    if pos_encoding_tensor is not None:
        # Unsqueeze explanation: adds dimension of size 1 at position 0 (beginning, leftmost)
        # magnitude.shape after truncation: (batch_size, n_channels, freq_bins, time_bins)
        # pos_encoding_tensor.shape: (1, freq_bins, time_bins) - should match truncated size
        # pos_encoding_tensor.unsqueeze(0).shape: (1, 1, freq_bins, time_bins) - broadcastable
        magnitude = magnitude * pos_encoding_tensor.unsqueeze(0)
    
    return magnitude
