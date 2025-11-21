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
        if not self.data_root.exists():
            raise FileNotFoundError(f"{self.data_root} does not exist! Loading will fail. Please check data_root input: {data_root}.")
        
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"{self.data_root} is not a directory! Loading will fail. Please check data_root input: {data_root}.")

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
            assert fs == self.FS_AUDIO
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
        if not pose_file.exists() or not pose_file.is_file():
            raise ValueError(f"Corresponding pose file {pose_file} does not exist for session #{session_id} and wave file {wav_file}!")
        with open(pose_file, 'r') as f:
            return json.load(f)

    def load_speech_transcriptions(self, session_id: int, wav_file: Path) -> Optional[List[dict]]:
        #Load speech transcription data
        session_name = f"Session_{session_id}"
        trans_file = self.speech_transcriptions_dir / session_name / (wav_file.stem + ".json")
        if not trans_file.exists():
            return ValueError(f"Corresponding transcript file {trans_file} does not exist for session #{session_id} and wave file {wav_file}!")
        with open(trans_file, 'r') as f:
            return json.load(f)

    def extract_wearer_6dof(self, poses_data: List[dict]) -> Optional[np.ndarray]:
        # (position + rotation) for glasses wearer
        n_frames = len(poses_data)
        pose_6dof = np.zeros((n_frames, 7), dtype=np.float32)
        
        for frame_idx, frame in enumerate(poses_data):
            found_wearer = False
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
            assert(found_wearer)

        return pose_6dof

    def get_all_participant_ids(self, poses_data: List[dict]) -> List[int]:
        '''
        Function retrieves the unique participant IDs from a pose list read from a .json file. 
        
        :param poses_data: The list of orientation frame data as read from an orientation .json file. 
                    As read by, e.g., self.load_speech_transcriptions() 
        
        :returns: A list with the unique participants ID in the list of frames. 
        '''
        #Get unique participant IDs from poses data
        return sorted(list(set(
            part["Participant_ID"]
            for frame in poses_data
            for part in frame["Participants"]
        )))

    def create_speech_lookup(self, transcription_data: Optional[List[dict]], 
                           n_frames: int) -> Dict[int, List[bool]]:
        '''
        Function return a lookup dictionary that looks like:
            lookup[participant_id][frame_id] = True if participant participant_id talks/is active in frame_id 
                                             = False if participant participant_id does not talk/is not active in frame_id
        
            Note that the lookup works even if a participant_id is not in a wave-file - it will always return False!
        
        Use with, e.g., doesParticipantTalkInFrame(lookup, 2, 119).
        
        :param transcription_data: Dictionary of transcription data. See load_speech_transcriptions().
        :param n_frames: Number of frames in the current transcription data.
        
        :returns: a defaultdict as a lookup table of booleans where you can fastly query whether a participant ID talked in a frame.
                Use, e.g., doesParticipantTalkInFrame() helper function.
        '''
        #lookup for when participants speak.
        
        # Factory method: whenever a new Participant_ID is seen, it will create a fresh list of 
        #  length expected_N_frames_head_tracking+1 (so indices go from 0 to expected_N_frames_head_tracking) 
        #  filled with False:
        lookup = defaultdict(lambda: [False] * n_frames)

        if transcription_data is None:
            raise ValueError("create_speech_lookup() ERROR: Transcription data not intialized!")

        for segment in transcription_data:
            pid = segment["Participant_ID"]
            # Python index starts at 0, seems that .json index starts at 1:
            start = segment["Start_Frame"] - 1
            end = segment["End_Frame"] - 1
            assert(end<=n_frames and start>=0) # Negligible perf effects
            lookup[pid][start:end] = [True] * (end - start)

        return lookup

    def print_stats(self):
        print(f"  Audio loading: {self.stats['success']} success, {self.stats['failed']} failed")
