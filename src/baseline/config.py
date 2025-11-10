import torch
from pathlib import Path
from typing import List


class Config:    
    DATA_ROOT = Path("/Users/preranarane/EasyComDataset/Main") 
    CHECKPOINT_DIR = Path("./checkpoints")    
    FS_AUDIO = 48000
    FS_HEAD_TRACKING = 20.0
    DT_HEAD_TRACKING = 1.0 / FS_HEAD_TRACKING
    ARRAY_WEARER_ID = 2
    SAMPLES_PER_FRAME = int(FS_AUDIO / FS_HEAD_TRACKING)  # 2400
    
    # Session splits
    TRAIN_SESSIONS = list(range(1, 11))  # Sessions 1-10
    VAL_SESSIONS = [11]                   # Session 11
    TEST_SESSIONS = [12]                  # Session 12
    
    # Audio channels
    USE_CHANNELS = [0, 1, 2, 3, 4, 5]  # All 6 microphones
    FILTER_SILENCE = True
    
    # Model architecture
    N_CHANNELS = len(USE_CHANNELS)
    HIDDEN_DIMS = [256, 128, 64] #can be updated
    OUTPUT_DIM = 7  # [x, y, z, qx, qy, qz, qw]
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4 
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    SEED = 42
    
    @classmethod
    def set_data_root(cls, path: str):
        cls.DATA_ROOT = Path(path)
    
    @classmethod
    def get_paths(cls):
        return {
            'data_root': cls.DATA_ROOT,
            'mic_array_audio': cls.DATA_ROOT / "Glasses_Microphone_Array_Audio",
            'tracked_poses': cls.DATA_ROOT / "Tracked_Poses",
            'speech_transcriptions': cls.DATA_ROOT / "Speech_Transcriptions",
            'checkpoint_dir': cls.CHECKPOINT_DIR
        }
    
    @classmethod
    def print_config(cls):
        print("\n" + "="*60)
        print("Configuration")
        print("="*60)
        print(f"Data root: {cls.DATA_ROOT}")
        print(f"Device: {cls.DEVICE}")
        print(f"\nDataset:")
        print(f"  Train sessions: {cls.TRAIN_SESSIONS}")
        print(f"  Val sessions: {cls.VAL_SESSIONS}")
        print(f"  Test sessions: {cls.TEST_SESSIONS}")
        print(f"  Channels: {cls.USE_CHANNELS}")
        print(f"  Filter silence: {cls.FILTER_SILENCE}")
        print(f"\nModel:")
        print(f"  Input: {cls.N_CHANNELS} channels × {cls.SAMPLES_PER_FRAME} samples")
        print(f"  Hidden layers: {cls.HIDDEN_DIMS}")
        print(f"  Output: {cls.OUTPUT_DIM} (3 position + 4 quaternion)")
        print(f"  Dropout: {cls.DROPOUT}")
        print(f"\nTraining:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Weight decay: {cls.WEIGHT_DECAY}")
        print("="*60)
