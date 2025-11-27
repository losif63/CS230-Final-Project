"""
Part of training of CNN models, CS230 project, fall 2025.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
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
            logger: Logger instance for debug-level logging
        """
        self.use_channels = use_channels
        self.filter_silence = filter_silence
        if output_dim not in [3, 4, 7]:
            raise ValueError(
                f"AudioPoseDataset() ERROR: Output dimension must be 3, 4, or 7, but got {output_dim}!"
            )
        self.output_dim = output_dim
        self.logger = logger
        self.loader = data_utils.EasyComDataLoader(
            data_root=data_root,
            fs_audio=fs_audio,
            fs_head_tracking=fs_head_tracking,
            array_wearer_id=array_wearer_id,
        )
        self.session_ids = session_ids
        self.samples = []
        self.cache_filename = Path(cache_filename).stem

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
                    audio_arrays_sessions.append(audio_frames_session)
                    wearer_pose_arrays_sessions.append(wearer_pose_session)
        else:
            print(f" Cached file missing {fn_}! Building dataset manually!")
            audio_arrays_sessions, wearer_pose_arrays_sessions, session_ids = (
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

        self.all_audio_frames = np.concatenate(audio_arrays_sessions, axis=0)
        self.all_wearer_pose_6dof = np.concatenate(wearer_pose_arrays_sessions, axis=0)
        assert self.all_audio_frames.shape[0] == self.all_wearer_pose_6dof.shape[0]

        print(f"   Loaded audio data: {self.all_audio_frames.shape}.")
        print(f"   Loaded wearer data: {self.all_wearer_pose_6dof.shape}.")

    def _build_dataset(self):
        samples_per_frame = int(self.loader.FS_AUDIO / self.loader.FS_HEAD_TRACKING)

        audio_arrays_sessions = []
        wearer_pose_arrays_sessions = []
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

                    wearer_pose_6dof = self.loader.extract_wearer_6dof(poses_data)
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

                except Exception as e:
                    raise ValueError(f"\n Error processing {wav_file.name}: {e}")

            audio_frames_session_array = np.concatenate(
                audio_frames_session_list, axis=0
            )
            wearer_pose_session_array = np.concatenate(
                wearer_pose_6dof_session_list, axis=0
            )

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
        audio_tensor = torch.from_numpy(self.all_audio_frames[idx, :, :])
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
    logger: Optional[logging.Logger] = None,
):
    """
    Create train, validation, and test dataloaders.

    Parameters
    ----------
    config : Config
        Configuration object with dataset parameters
    output_dim : int
        Output dimension (3=position only, 4=rotation only, 7=full 6DOF). Note the data written in cached files in AudioPoseDataset() stores all 7 values!
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    if logger:
        logger.debug("      Dataloaders created successfully. Returning...")

    return train_loader, val_loader, test_loader
