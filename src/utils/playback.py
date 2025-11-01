'''
Created on Oct 31, 2025

    Module with functions to playback stereo signals throughg the main audio output (no advanced querying done).
    
    Module is using portaudio. 

@author: Sebastian Prepelita
'''

import numpy as np

import pyaudio

def interleave_channels_for_pyaudio(data: np.ndarray) -> bytes:
    """
    Interleave multi-channel audio data for PyAudio playback.
    
    Parameters:
        data (np.ndarray): Shape (n_samples, n_channels), dtype must be int16, int32, or float32.
    
    Returns:
        bytes: Interleaved PCM byte stream.
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array of shape (n_samples, n_channels)")
    output_signal = np.zeros(2*len(data), dtype = data.dtype)
    output_signal[::2] = data[:,0]
    output_signal[1::2] = data[:,1]
    # Flatten in C order (row-major) to interleave channels
    return output_signal.tobytes()

def playback_sterero_signal(stereo_signal_to_play: np.ndarray, fs: int, chunk_size: int = 1024):
    # Initialize PyAudio
    print("Starting playback..")
    p = pyaudio.PyAudio()
    if stereo_signal_to_play.dtype == np.int32:
        format_ = pyaudio.paInt32
    elif stereo_signal_to_play.dtype == np.int64:
        stereo_signal_to_play = stereo_signal_to_play.astype(np.float32)/ np.iinfo(np.int64).max
        format_ = pyaudio.paFloat32
    elif stereo_signal_to_play.dtype == np.float32:
        format_ = pyaudio.paFloat32
    elif stereo_signal_to_play.dtype == np.float64:
        stereo_signal_to_play = stereo_signal_to_play.astype(np.float32)
        format_ = pyaudio.paFloat32
    else:
        raise ValueError(f"Unknown playback format for dtype {stereo_signal_to_play.dtype}")
    stream = p.open(format=format_,
                    channels=2,
                    rate=fs,
                    output=True,
                    frames_per_buffer = chunk_size,
                    input=False)
    for i in range(0, len(stereo_signal_to_play), chunk_size):
        chunk = stereo_signal_to_play[i:i+chunk_size]  # shape (chunk_size, 2)
        stream.write(interleave_channels_for_pyaudio(chunk))
    print("end playback..")
    stream.stop_stream()
    stream.close()
    p.terminate()