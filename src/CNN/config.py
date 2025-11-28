import torch
from pathlib import Path


class Config:
    # DATA_ROOT = Path(r"D:\Temp_S230_Database\Main") # Local machine
    DATA_ROOT = Path(r"C:\easycom_dataset\Main")  # Meta desktop
    CHECKPOINT_DIR = Path("./checkpoints")
    TRAINING_RESUTLS_DIR = Path("./training_results")
    TESTING_RESUTLS_DIR = Path("./testing_results")
    MODEL_LIST_FILE = "model_configs.json"
    MODEL_2D_LIST_FILE = "2d_model_configs.json"
    FS_AUDIO = 48000
    FS_HEAD_TRACKING = 20.0
    DT_HEAD_TRACKING = 1.0 / FS_HEAD_TRACKING
    ARRAY_WEARER_ID = 2
    SAMPLES_PER_FRAME = int(FS_AUDIO / FS_HEAD_TRACKING)  # 2400

    # Session splits
    TRAIN_SESSIONS = list(range(1, 3))  # Sessions 1-10
    VAL_SESSIONS = [11]  # Session 11
    TEST_SESSIONS = [12]  # Session 12

    # Caching:
    TRAIN_CACHE_FN = "easycom_train_data_cache"
    VAL_CACHE_FN = "easycom_validation_data_cache"
    TEST_CACHE_FN = "easycom_test_data_cache"

    # Audio channels
    USE_CHANNELS = [0, 1, 2, 3, 4, 5]  # All 6 microphones
    FILTER_SILENCE = True

    # Model architecture
    N_CHANNELS = len(USE_CHANNELS)
    HIDDEN_DIMS = [256, 128, 64]  # can be updated
    OUTPUT_DIM = 7  # [x, y, z, qx, qy, qz, qw]
    DROPOUT = 0.3

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 0  # For windows, use 0 (does not work). For A100, not a big improvement but 2 is fine...

    # Parallel training parameters
    TRAININGS_PER_GPU = 2  # Number of parallel trainings per GPU (e.g., 2 = train 2 models per GPU simultaneously)

    # GPU assignment strategy for parallel training:
    # - "round_robin": Distributes models evenly across all GPUs (e.g., Model 0→GPU0, Model 1→GPU1, Model 2→GPU2, Model 3→GPU0, ...)
    #                  Best for balanced load distribution and when models have varying training times.
    #                  All GPUs stay busy throughout the entire training process.
    # - "sequential": Assigns models in blocks to GPUs (e.g., Models 0-1→GPU0, Models 2-3→GPU1, Models 4-5→GPU2, ...)
    #                 Best when you want GPUs to finish completely and become available for other work.
    #                 Easier to monitor progress by GPU and better for batch-based workflows.
    GPU_ASSIGNMENT_STRATEGY = "round_robin"  # Options: "round_robin" or "sequential"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    @classmethod
    def set_data_root(cls, path: str):
        cls.DATA_ROOT = Path(path)

    @classmethod
    def get_paths(cls):
        return {
            "data_root": cls.DATA_ROOT,
            "mic_array_audio": cls.DATA_ROOT / "Glasses_Microphone_Array_Audio",
            "tracked_poses": cls.DATA_ROOT / "Tracked_Poses",
            "speech_transcriptions": cls.DATA_ROOT / "Speech_Transcriptions",
            "checkpoint_dir": cls.CHECKPOINT_DIR,
        }

    @classmethod
    def print_config(cls):
        print("\n" + "=" * 60)
        print("Configuration (CNN)")
        print("=" * 60)
        print(f"Data root: {cls.DATA_ROOT}")
        print(f"Checkpoint dir: {cls.CHECKPOINT_DIR}")
        print(f"Training results dir: {cls.TRAINING_RESUTLS_DIR}")
        print(f"Testing results dir: {cls.TESTING_RESUTLS_DIR}")
        print(f"Model list file (1D): {cls.MODEL_LIST_FILE}")
        print(f"Model list file (2D): {cls.MODEL_2D_LIST_FILE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Seed: {cls.SEED}")
        print(f"\nAudio/Tracking:")
        print(f"  Audio sampling rate: {cls.FS_AUDIO} Hz")
        print(f"  Head tracking rate: {cls.FS_HEAD_TRACKING} Hz")
        print(f"  Head tracking dt: {cls.DT_HEAD_TRACKING} s")
        print(f"  Array wearer ID: {cls.ARRAY_WEARER_ID}")
        print(f"  Samples per frame: {cls.SAMPLES_PER_FRAME}")
        print(f"\nDataset:")
        print(f"  Train sessions: {cls.TRAIN_SESSIONS}")
        print(f"  Val sessions: {cls.VAL_SESSIONS}")
        print(f"  Test sessions: {cls.TEST_SESSIONS}")
        print(f"  Train cache file: {cls.TRAIN_CACHE_FN}")
        print(f"  Val cache file: {cls.VAL_CACHE_FN}")
        print(f"  Test cache file: {cls.TEST_CACHE_FN}")
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
        print(f"  Num workers: {cls.NUM_WORKERS}")
        print(f"\nParallel Training:")
        print(f"  Trainings per GPU: {cls.TRAININGS_PER_GPU}")
        print(f"  GPU assignment strategy: {cls.GPU_ASSIGNMENT_STRATEGY}")
        print("=" * 60 + "/CNN")
