from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from data_utils import EasyComDataLoader


class AudioPoseDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 session_ids: List[int],
                 use_channels: List[int] = [0, 1, 2, 3, 4, 5],
                 filter_silence: bool = True,
                 fs_audio: int = 48000,
                 fs_head_tracking: float = 20.0,
                 array_wearer_id: int = 2):
        """
        Inputs:
            data_root: Root directory of EasyCom dataset
            session_ids: List of session IDs to include (e.g., [1,2,3,...,10])
            use_channels: Which audio channels to use (0-5 for 6-channel array)
            filter_silence: If True, only include frames with active speech
            fs_audio: Audio sampling frequency
            fs_head_tracking: Head tracking sampling frequency
            array_wearer_id: Participant ID of glasses wearer
        """
        self.use_channels = use_channels
        self.filter_silence = filter_silence
        self.loader = EasyComDataLoader(
            data_root=data_root,
            fs_audio=fs_audio,
            fs_head_tracking=fs_head_tracking,
            array_wearer_id=array_wearer_id
        )
        self.session_ids = session_ids
        self.samples = []

        self._build_dataset()
        print(f" Built dataset with {len(self.samples)} samples")
        self.loader.print_stats()

    def _build_dataset(self):
        samples_per_frame = int(self.loader.FS_AUDIO / self.loader.FS_HEAD_TRACKING)

        for session_id in tqdm(self.session_ids, desc="Loading sessions"):
            session_dir = self.loader.get_session_dir(session_id)

            if not session_dir.exists():
                print(f" Session {session_id} not found, skipping...")
                continue

            wav_files = self.loader.get_wav_files(session_dir)

            for wav_file in wav_files:
                try:
                    audio, fs = self.loader.load_audio(wav_file)
                    if audio is None:
                        continue

                    n_samples, n_channels = audio.shape
                    if max(self.use_channels) >= n_channels:
                        continue

                    poses_data = self.loader.load_tracked_poses(session_id, wav_file)
                    if poses_data is None:
                        continue

                    pose_6dof = self.loader.extract_wearer_6dof(poses_data)
                    if pose_6dof is None:
                        continue

                    n_frames = len(pose_6dof)

                    if self.filter_silence:
                        transcription_data = self.loader.load_speech_transcriptions(
                            session_id, wav_file
                        )
                        speech_lookup = self.loader.create_speech_lookup(
                            transcription_data, n_frames
                        )
                        participant_ids = self.loader.get_all_participant_ids(poses_data)

                    for frame_idx in range(n_frames):
                        start_sample = frame_idx * samples_per_frame
                        end_sample = (frame_idx + 1) * samples_per_frame

                        if end_sample > n_samples:
                            break

                        if self.filter_silence:
                            is_active = any(
                                speech_lookup[pid][frame_idx]
                                for pid in participant_ids
                                if pid != self.loader.ARRAY_WEARER_ID
                            )
                            if not is_active:
                                continue

                        if np.linalg.norm(pose_6dof[frame_idx]) < 0.1:
                            continue

                        # Cache audio frame
                        audio_frame = audio[start_sample:end_sample, self.use_channels].T
                        audio_frame = audio_frame.astype(np.float32)
                        self.samples.append({
                            'audio_frame': audio_frame,  # (n_channels, 2400)
                            'pose_6dof': pose_6dof[frame_idx].astype(np.float32)
                        })

                except Exception as e:
                    print(f"\n Error processing {wav_file.name}: {e}")
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        audio_tensor = torch.from_numpy(sample['audio_frame'])
        pose_tensor = torch.from_numpy(sample['pose_6dof'])

        return audio_tensor, pose_tensor


def create_dataloaders(config):
    #Create train, validation, and test dataloaders
    
    from torch.utils.data import DataLoader
    
    train_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TRAIN_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID
    )

    val_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.VAL_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID
    )

    test_dataset = AudioPoseDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TEST_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        fs_audio=config.FS_AUDIO,
        fs_head_tracking=config.FS_HEAD_TRACKING,
        array_wearer_id=config.ARRAY_WEARER_ID
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
