from pathlib import Path
import os
import sys
import json
from typing import Optional, List, Dict
from collections import defaultdict

import torch
import torchaudio

sys.path.append(str(Path(__file__).parent.parent))
from utils.utilsIO import get_head_tracking_fs


def load_speech_transcriptions(
    speech_dir: Path, session_id: int, wav_path: Path
) -> Optional[List[dict]]:
    """Load speech transcription annotations for a wav file."""
    session_name = f"Session_{session_id}"
    trans_file = speech_dir / session_name / f"{wav_path.stem}.json"
    if not trans_file.exists():
        return None
    try:
        with open(trans_file, "r", encoding="latin-1") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading speech transcriptions from {trans_file}: {e}")
        return None


def create_speech_lookup(
    transcription_data: Optional[List[dict]], n_frames: int
) -> Dict[int, List[bool]]:
    """Return lookup indicating when each participant speaks."""
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


def get_all_participant_ids(pose_data: List[dict]) -> List[int]:
    """Collect all participant IDs present in pose annotations."""
    participants = {
        participant["Participant_ID"]
        for frame in pose_data
        for participant in frame["Participants"]
    }
    return sorted(list(participants))


def preprocess_pair(
    wav_path: Path,
    json_path: Path,
    speech_dir: Path,
    session_id: int,
    participant_id: int = 2,
    pose_norm_threshold: float = 0.1,
):
    # Read audio
    audio_data, sr = torchaudio.load(wav_path)
    
    # Handle mono vs multi-channel audio
    if len(audio_data.shape) == 1:
        audio_data = audio_data.unsqueeze(0)
    
    _, n_samples = audio_data.shape
    dT_head_tracking = get_head_tracking_fs()
    samples_per_frame = int(sr / dT_head_tracking)
    n_audio_frames = n_samples // samples_per_frame
    
    # Read pose data
    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    pose_tensor = torch.zeros((len(pose_data), 7))
    for i, frame in enumerate(pose_data):
        assert i == frame['Frame_Number'] - 1
        for participant in frame['Participants']:
            if participant['Participant_ID'] != participant_id:
                continue
            else:
                pose_tensor[i, 0] = participant["Position_X"]
                pose_tensor[i, 1] = participant["Position_Y"]
                pose_tensor[i, 2] = participant["Position_Z"]
                pose_tensor[i, 3] = participant["Quaternion_X"]
                pose_tensor[i, 4] = participant["Quaternion_Y"]
                pose_tensor[i, 5] = participant["Quaternion_Z"]
                pose_tensor[i, 6] = participant["Quaternion_W"]
    
    n_pose_frames = pose_tensor.shape[0]
    max_frames = min(n_audio_frames, n_pose_frames)
    audio_frames: List[torch.Tensor] = []
    pose_frames: List[torch.Tensor] = []

    for frame_idx in range(max_frames):
        start_sample = frame_idx * samples_per_frame
        end_sample = start_sample + samples_per_frame

        pose_frame = pose_tensor[frame_idx]
        if torch.linalg.norm(pose_frame) < pose_norm_threshold:
            continue

        audio_frames.append(audio_data[:, start_sample:end_sample])
        pose_frames.append(pose_frame)

    if not audio_frames:
        return torch.empty((0, audio_data.shape[0], samples_per_frame)), torch.empty((0, 7))

    audio_tensor = torch.stack(audio_frames, dim=0)
    pose_tensor = torch.stack(pose_frames, dim=0)

    return audio_tensor, pose_tensor


def build_cache(audio_dir, pose_dir, speech_dir, session_ids, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = []
    
    for session in session_ids:
        session_dir = Path(f"Session_{session}")
        sub_dir = audio_dir / session_dir

        for wav_path in sorted(sub_dir.rglob("*.wav")):
            rel = wav_path.relative_to(sub_dir)
            stem = rel.with_suffix("")  # e.g. file_0001
            json_path = pose_dir / session_dir / stem.with_suffix(".json")

            if not json_path.exists():
                continue

            audio_t, pose_t = preprocess_pair(
                wav_path,
                json_path,
                speech_dir,
                session,
            )
            out_name = os.path.join(session_dir, str(stem) + ".pt")
            out_path = cache_dir / out_name
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            torch.save({"audio": audio_t, "pose": pose_t}, out_path)
            index.append(out_name)
            print(f"Saved audio: {audio_t.shape} | pose: {pose_t.shape} ", out_path)

    # optionally save index list
    torch.save(index, cache_dir / "index.pt")

if __name__ == '__main__':
    root_dir = Path('data/Main')
    audio_dir = root_dir / Path('Glasses_Microphone_Array_Audio')
    pose_dir = root_dir / Path('Tracked_Poses')
    speech_dir = root_dir / Path('Speech_Transcriptions')
    build_cache(audio_dir, pose_dir, speech_dir, list(range(1, 11)), root_dir / Path('Train'))
    build_cache(audio_dir, pose_dir, speech_dir, [11], root_dir / Path('Dev'))
    build_cache(audio_dir, pose_dir, speech_dir, [12], root_dir / Path('Test'))
