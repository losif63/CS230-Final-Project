from pathlib import Path
import torch, torchaudio, json
import sys, os
sys.path.append(str(Path(__file__).parent.parent))
from utils.utilsIO import get_head_tracking_fs


def preprocess_pair(wav_path, json_path, participant_id=2):
    # Read audio
    audio_data, sr = torchaudio.load(wav_path)
    
    # Handle mono vs multi-channel audio
    if len(audio_data.shape) == 1:
        audio_data.unsqueeze(0)
    
    N_channels, N_taps = audio_data.shape
    t_max = N_taps // sr
    dT_head_tracking = get_head_tracking_fs()

    N_frames = int(t_max * dT_head_tracking)
    N_samples_per_frame = int(sr / dT_head_tracking)
    
    # Reshape audio to (frames, channels, samples_per_frame)
    audio_tensor = torch.stack([
        audio_data[:, i * N_samples_per_frame:(i+1)*N_samples_per_frame] for i in range(N_frames)
    ], dim=0)
    
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
    
    # Ensure frames match
    min_frames = min(audio_tensor.shape[0], pose_tensor.shape[0])
    audio_tensor = audio_tensor[:min_frames, :]
    pose_tensor = pose_tensor[:min_frames, :]

    return audio_tensor, pose_tensor


def build_cache(audio_dir, pose_dir, session_ids, cache_dir):
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

            audio_t, pose_t = preprocess_pair(wav_path, json_path)
            out_name = os.path.join(session_dir, str(stem) + ".pt")
            out_path = cache_dir / out_name
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            torch.save({"audio": audio_t, "pose": pose_t}, out_path)
            index.append(out_name)
            print("Saved", out_path)

    # optionally save index list
    torch.save(index, cache_dir / "index.pt")

if __name__ == '__main__':
    root_dir = Path('data/Main')
    audio_dir = root_dir / Path('Glasses_Microphone_Array_Audio')
    pose_dir = root_dir / Path('Tracked_Poses')
    build_cache(audio_dir, pose_dir, list(range(1, 11)), root_dir / Path('Train'))
    build_cache(audio_dir, pose_dir, [11], root_dir / Path('Dev'))
    build_cache(audio_dir, pose_dir, [12], root_dir / Path('Test'))
