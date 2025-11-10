import json
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from collections import defaultdict

import numpy as np
from scipy.io import wavfile

warnings.filterwarnings('ignore')


def convert_int_to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int32:
        return data.astype(np.float32) / np.iinfo(np.int32).max
    elif data.dtype == np.int16:
        return data.astype(np.float32) / np.iinfo(np.int16).max
    elif data.dtype == np.int64:
        return data.astype(np.float32) / np.iinfo(np.int64).max
    elif data.dtype == np.float32:
        return data
    elif data.dtype == np.float64:
        return data.astype(np.float32)
    else:
        raise ValueError(f"Unknown data type: {data.dtype}")


class EasyComDataLoader:
    def __init__(self, data_root: str, fs_audio: int = 48000, 
                 fs_head_tracking: float = 20.0, array_wearer_id: int = 2):        
        """
        Inputs:
            data_root: Root directory of EasyCom dataset
            fs_audio: Audio sampling frequency (Hz)
            fs_head_tracking: Head tracking sampling frequency (Hz)
            array_wearer_id: Participant ID of the glasses wearer
        """
        self.data_root = Path(data_root)
        self.mic_array_audio_path = self.data_root / "Glasses_Microphone_Array_Audio"
        self.tracked_poses_dir = self.data_root / "Tracked_Poses"
        self.speech_transcriptions_dir = self.data_root / "Speech_Transcriptions"

        self.FS_AUDIO = fs_audio
        self.FS_HEAD_TRACKING = fs_head_tracking
        self.DT_HEAD_TRACKING = 1.0 / self.FS_HEAD_TRACKING
        self.ARRAY_WEARER_ID = array_wearer_id

        self.stats = {'success': 0, 'failed': 0}

    def get_session_dir(self, session_id: int) -> Path:
        session_name = f"Session_{session_id}"
        return self.mic_array_audio_path / session_name

    def get_wav_files(self, session_dir: Path) -> List[Path]:
        return sorted(session_dir.glob("*.wav"))

    def load_audio(self, wav_file: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
        try:
            fs, data = wavfile.read(str(wav_file))
            data = convert_int_to_float(data)
            self.stats['success'] += 1
            return data, fs
        except Exception as e:
            self.stats['failed'] += 1
            print(f"\n Failed to load {wav_file.name}: {e}")
            return None, None

    def load_tracked_poses(self, session_id: int, wav_file: Path) -> Optional[List[dict]]:
        #Load pose data for audio file
        session_name = f"Session_{session_id}"
        pose_file = self.tracked_poses_dir / session_name / (wav_file.stem + ".json")
        if not pose_file.exists():
            return None
        with open(pose_file, 'r') as f:
            return json.load(f)

    def load_speech_transcriptions(self, session_id: int, wav_file: Path) -> Optional[List[dict]]:
        #Load speech transcription data
        session_name = f"Session_{session_id}"
        trans_file = self.speech_transcriptions_dir / session_name / (wav_file.stem + ".json")
        if not trans_file.exists():
            return None
        with open(trans_file, 'r') as f:
            return json.load(f)

    def extract_wearer_6dof(self, poses_data: List[dict]) -> Optional[np.ndarray]:
        # (position + rotation) for glasses wearer
        n_frames = len(poses_data)
        pose_6dof = np.zeros((n_frames, 7), dtype=np.float32)

        found_wearer = False
        for frame_idx, frame in enumerate(poses_data):
            for participant in frame["Participants"]:
                if participant["Participant_ID"] == self.ARRAY_WEARER_ID:
                    pose_6dof[frame_idx, 0] = participant["Position_X"]
                    pose_6dof[frame_idx, 1] = participant["Position_Y"]
                    pose_6dof[frame_idx, 2] = participant["Position_Z"]
                    pose_6dof[frame_idx, 3] = participant["Quaternion_X"]
                    pose_6dof[frame_idx, 4] = participant["Quaternion_Y"]
                    pose_6dof[frame_idx, 5] = participant["Quaternion_Z"]
                    pose_6dof[frame_idx, 6] = participant["Quaternion_W"] #[x, y, z, qx, qy, qz, qw]
                    found_wearer = True
                    break

        return pose_6dof if found_wearer else None

    def get_all_participant_ids(self, poses_data: List[dict]) -> List[int]:
        #Get unique participant IDs from poses data
        return sorted(list(set(
            part["Participant_ID"]
            for frame in poses_data
            for part in frame["Participants"]
        )))

    def create_speech_lookup(self, transcription_data: Optional[List[dict]], 
                           n_frames: int) -> Dict[int, List[bool]]:
        #lookup for when participants speak.

        lookup = defaultdict(lambda: [False] * n_frames)

        if transcription_data is None:
            return lookup

        for segment in transcription_data:
            pid = segment["Participant_ID"]
            start = segment["Start_Frame"] - 1
            end = segment["End_Frame"] - 1

            for frame_idx in range(start, end):
                if 0 <= frame_idx < n_frames:
                    lookup[pid][frame_idx] = True

        return lookup

    def print_stats(self):
        print(f"Audio loading: {self.stats['success']} success, {self.stats['failed']} failed")
