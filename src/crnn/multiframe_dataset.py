"""
Author: Prerana Rane
"""

from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_utils import EasyComDataLoader

class MultiFrameDataset(Dataset):

    def __init__(self, 
                 data_root: str, 
                 session_ids: List[int],
                 use_channels: List[int] = [0, 1, 2, 3, 4, 5],
                 filter_silence: bool = True,
                 n_consecutive_frames: int = 10):
        
        self.use_channels = use_channels
        self.filter_silence = filter_silence
        self.n_consecutive_frames = n_consecutive_frames
        self.loader = EasyComDataLoader(data_root)
        self.session_ids = session_ids
        self.samples = []

        self._build_dataset()
        self.loader.print_stats()

    def _build_dataset(self):
        for session_id in tqdm(self.session_ids, desc="Loading sessions"):
            session_dir = self.loader.get_session_dir(session_id)

            if not session_dir.exists():
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
                    samples_per_frame = int(self.loader.FS_AUDIO / self.loader.FS_HEAD_TRACKING)

                    if self.filter_silence:
                        transcription_data = self.loader.load_speech_transcriptions(session_id, wav_file)
                        speech_lookup = self.loader.create_speech_lookup(transcription_data, n_frames)
                        participant_ids = self.loader.get_all_participant_ids(poses_data)

                    recording_frames = []

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

                        recording_frames.append({
                            'audio_frame': audio_frame, 
                            'pose_6dof': pose_6dof[frame_idx].astype(np.float32)
                        })

                    # Create sequences of n_consecutive_frames using sliding window
                    for i in range(len(recording_frames) - self.n_consecutive_frames + 1):
                        sequence = recording_frames[i:i + self.n_consecutive_frames]

                        # n_consecutive_frames, n_channels, 2400
                        audio_sequence = np.stack([f['audio_frame'] for f in sequence])

                        # Target is last frame pose 
                        target_pose = sequence[-1]['pose_6dof']

                        self.samples.append({
                            'audio_sequence': audio_sequence,
                            'pose_6dof': target_pose
                        })

                except Exception as e:
                    print(f"\n Error processing {wav_file.name}: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_tensor = torch.from_numpy(sample['audio_sequence'])
        pose_tensor = torch.from_numpy(sample['pose_6dof'])

        return audio_tensor, pose_tensor
    
def create_dataloaders(config):
    #Create train, validation, and test dataloaders
    
    from torch.utils.data import DataLoader
    
    train_dataset = MultiFrameDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TRAIN_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        n_consecutive_frames=config.N_CONSECUTIVE_FRAMES
    )

    val_dataset = MultiFrameDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.VAL_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        n_consecutive_frames=config.N_CONSECUTIVE_FRAMES
    )

    test_dataset = MultiFrameDataset(
        data_root=str(config.DATA_ROOT),
        session_ids=config.TEST_SESSIONS,
        use_channels=config.USE_CHANNELS,
        filter_silence=config.FILTER_SILENCE,
        n_consecutive_frames=config.N_CONSECUTIVE_FRAMES
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
