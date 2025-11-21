from pathlib import Path
from typing import List
import numpy as np
import torch
import h5py
import re
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import time, datetime

from src.CNN import data_utils
import src.baseline.config

class AudioPoseDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 session_ids: List[int],
                 use_channels: List[int] = [0, 1, 2, 3, 4, 5],
                 filter_silence: bool = True,
                 fs_audio: int = 48000,
                 fs_head_tracking: float = 20.0,
                 array_wearer_id: int = 2,
                 cache_filename: str = None):
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
        """
        self.use_channels = use_channels
        self.filter_silence = filter_silence
        self.loader = data_utils.EasyComDataLoader(
            data_root=data_root,
            fs_audio=fs_audio,
            fs_head_tracking=fs_head_tracking,
            array_wearer_id=array_wearer_id
        )
        self.session_ids = session_ids
        self.samples = []
        self.cache_filename = Path(cache_filename).stem

        self._load_cached_dataset()
        print(f" Built dataset with {len(self.all_audio_frames)} samples")

    def _load_cached_dataset(self):
        # Check to see if data is cached:
        if self.cache_filename is not None:
            fn_ = Path(self.cache_filename).stem + "_S" + str(self.session_ids[0]) + "_S" + str(self.session_ids[-1]) + ".hdf5"
            fn_ = self.loader.data_root / fn_
        else:
            fn_ = None
        if fn_ is not None and fn_.exists():
            print(f" Dataset cached file {fn_} found - loading it...")
            with h5py.File(fn_, "r") as f:
                root_groups = [name for name in f.keys() if isinstance(f[name], h5py.Group)]
                session_ids = []
                file_filter_silence = f["filter_silence"][...]
                if not file_filter_silence == self.filter_silence:
                    raise ValueError(f"Cached hdf5 file has filtered silence to {file_filter_silence}, but the dataset needs to have filtered silence to {self.filter_silence}.\nb"
                                     "Use different filename for cache or change filter_silence input.")
                for group_string in root_groups:
                    match = re.search(r"session_(\d+)", group_string)
                    if match:
                        num = int(match.group(1))
                    else:
                        raise ValueError(f"Unexpected group type: expected session_\d+, but got {group_string}")
                    session_ids.append(num)
                if set(session_ids) != set(self.session_ids):
                    raise ValueError(f"The cached database contains different sessions: {session_ids} . You requested sessions: {self.session_ids}")
                audio_arrays_sessions = []
                wearer_pose_arrays_sessions = []
                for group_string in root_groups:
                    audio_frames_session = f[f"{group_string}/audio_frames"][...]
                    wearer_pose_session = f[f"{group_string}/wearer_pose"][...]
                    audio_arrays_sessions.append(audio_frames_session)
                    wearer_pose_arrays_sessions.append(wearer_pose_session)
        else:
            print(f" Cached file missing {fn_}! Building dataset manually!")
            audio_arrays_sessions, wearer_pose_arrays_sessions, session_ids = self._build_dataset()
            
            # dump data:
            with h5py.File(fn_, "w") as f:
                f.create_dataset("filter_silence", data = self.filter_silence)
                for idx, session_id in enumerate(session_ids):
                    session_group = f.create_group(f"session_{session_id}")
                    session_group.create_dataset("audio_frames", data=audio_arrays_sessions[idx])
                    session_group.create_dataset("wearer_pose", data=wearer_pose_arrays_sessions[idx])

        self.all_audio_frames = np.concatenate(audio_arrays_sessions, axis = 0)
        self.all_wearer_pose_6dof = np.concatenate(wearer_pose_arrays_sessions, axis = 0)
        assert(self.all_audio_frames.shape[0] == self.all_wearer_pose_6dof.shape[0])
        
        print(f"   Loaded audio data: {self.all_audio_frames.shape}.")
        print(f"   Loaded wearer data: {self.all_wearer_pose_6dof.shape}.")
    
    def _build_dataset(self):
        samples_per_frame = int(self.loader.FS_AUDIO / self.loader.FS_HEAD_TRACKING)
        
        audio_arrays_sessions = []
        wearer_pose_arrays_sessions = []
        session_ids = []
        for session_id in tqdm(self.session_ids, desc="Loading sessions"):
            session_dir = self.loader.get_session_dir(session_id)

            if not session_dir.exists():
                print(f" Session {session_id} not found, skipping... !")
                continue

            wav_files = self.loader.get_wav_files(session_dir)
            
            audio_frames_session_list = []
            wearer_pose_6dof_session_list = []
            for wave_file_idx, wav_file in enumerate(wav_files):
                try:
                    audio, _ = self.loader.load_audio(wav_file)
                    
                    if audio is None:
                        raise ValueError(f"Got no audio for file {wav_file}!")

                    n_samples, n_channels = audio.shape
                    
                    if max(self.use_channels) >= n_channels:
                        raise ValueError(f"Requesting too many channels: {self.use_channels}. Audio has {n_channels} channels only!")

                    poses_data = self.loader.load_tracked_poses(session_id, wav_file)
                    t_max = (1.0*len(audio) -1) / self.loader.FS_AUDIO
                    expected_N_frames_head_tracking = int(round(t_max * self.loader.FS_HEAD_TRACKING ))
                    if poses_data is None or len(poses_data) != expected_N_frames_head_tracking:
                        raise ValueError(f"Poses loading error!!")

                    wearer_pose_6dof = self.loader.extract_wearer_6dof(poses_data)
                    
                    n_frames = len(wearer_pose_6dof)

                    if self.filter_silence:
                        transcription_data = self.loader.load_speech_transcriptions(
                            session_id, wav_file
                        )
                        speech_lookup = self.loader.create_speech_lookup(
                            transcription_data, n_frames
                        )
                        participant_ids = self.loader.get_all_participant_ids(poses_data)
                        external_participant_ids = [pid for pid in participant_ids if pid != self.loader.ARRAY_WEARER_ID]

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
                        audio_frame = audio[start_sample:end_sample, self.use_channels].T
                        audio_frame = audio_frame.astype(np.float32)
                        # We'll use self.all_audio_frames and self.all_wearer_pose_6dof as cached data instead
                        # self.samples.append({
                        #     'audio_frame': audio_frame,  # (n_channels, 2400)
                        #     'pose_6dof': wearer_pose_6dof[frame_idx].astype(np.float32)
                        # })
                        audio_frames_session_list.append(audio_frame[None, :, :]) # n_frames x (n_channels, 2400)
                        wp_ = wearer_pose_6dof[frame_idx].astype(np.float32)
                        wearer_pose_6dof_session_list.append(wp_[None, :])  # n_frames x (7)
                
                except Exception as e:
                    raise ValueError(f"\n Error processing {wav_file.name}: {e}")
            
            audio_frames_session_array = np.concatenate(audio_frames_session_list, axis = 0)
            wearer_pose_session_array = np.concatenate(wearer_pose_6dof_session_list, axis = 0)
            
            # Build big list with all sessions:
            audio_arrays_sessions.append(audio_frames_session_array)
            wearer_pose_arrays_sessions.append(wearer_pose_session_array)
            session_ids.append(session_id)
        self.loader.print_stats()
        return audio_arrays_sessions, wearer_pose_arrays_sessions, session_ids

    def __len__(self) -> int:
        assert self.all_audio_frames.shape[0] == self.all_wearer_pose_6dof.shape[0]
        return len(self.all_audio_frames)

    def __getitem__(self, idx: int):
        #sample = self.samples[idx]
        #audio_tensor = torch.from_numpy(sample['audio_frame'])
        #pose_tensor = torch.from_numpy(sample['pose_6dof'])
        
        audio_tensor = torch.from_numpy(self.all_audio_frames[idx, :, :])
        pose_tensor = torch.from_numpy(self.all_wearer_pose_6dof[idx, :])

        return audio_tensor, pose_tensor


def create_dataloaders(config: src.baseline.config.Config):
    #Create train, validation, and test dataloaders
    
    from torch.utils.data import DataLoader
    start = time.perf_counter()
    train_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TRAIN_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID,
        cache_filename=config.TRAIN_CACHE_FN
    )
    end = time.perf_counter()
    print(f"   Training loading took {datetime.timedelta(seconds=end-start)}")

    val_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.VAL_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID,
        cache_filename=config.VAL_CACHE_FN
    )

    test_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TEST_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID,
        cache_filename=config.TEST_CACHE_FN
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader
