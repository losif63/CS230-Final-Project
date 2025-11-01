'''
Created on Oct 23, 2025

    Module with various util functions for reading/writing SPECIFICALLY for the Easycom dataset!
    
    References:
        [1] Easycom dataset: https://imperialcollegelondon.github.io/spear-challenge/data
        [2] Easycom dataset download: https://imperialcollegelondon.github.io/spear-challenge/downloads
        [3] Easycom dataset github: https://github.com/facebookresearch/EasyComDataset
        [4] Project github repository: https://docs.google.com/document/d/1m0JPmlP6f08ar7wwx1Zam8ooYojFWnwNwyy17qEouuE/edit?tab=t.0
        [5] Donley J., Tourbabin V., Lee J.-S., Broyles M., Jiang H., Shen J., Pantic M., Ithapu V.K., Mehra R., "EasyCom: An Augmented
        Reality Dataset to Support Algorithms for Easy Communication in Noisy Environments", arxiv, (2021)
        
    HeadPositions:
    
    From the perspective of the participant wearing the camera, the coordinate system is oriented as follows[3]:

            Positive Position_X points left
            Positive Position_Y points upwards
            Positive Position_Z points forwards
    
        The origin for the pose data is at the leg of the table on the ground to the immediate left, as viewed from the participant wearing the camera when looking forward.

    TODO:
        *) Output comparison functions:
            quaternions --> vector on unit sphere --> cosine of angle --> angle --> error

@author: Sebastian Prepelita
'''
from pathlib import Path

import re
import numpy as np

from scipy.io import wavfile

from collections import defaultdict

import csv
import json

import time
import datetime

import playback

from matplotlib import pyplot as plt

import sys
import warnings

# Regex pattern to extract D#, S#, and M##
wav_file_pattern = r"array_D(\d+)_S(\d+)_M(\d+)\.wav"
session_dir_pattern = r"Session_(\d+)"

EXTERNAL_TALKER_IDS = (1, 3, 4, 5, 6, 7)

def get_paths(path_to_dataset: str) -> tuple[Path]:
    '''
    Helper function that retireves the relevant paths based on the base path where the dataset from [4] is downloaded. 
    
    :param path_to_dataset: Root folder where you downloaded [4].
    '''
    mic_array_audio_path = Path(path_to_dataset) / "Glasses_Microphone_Array_Audio"
    # Session 1: 2880000 @ fs=48 kHz --> 60 seconds clips
    array_orientation_dir = Path(path_to_dataset) / "Tracked_Poses"
    speech_transcriptions_dir = Path(path_to_dataset) / "Speech_Transcriptions"
    return mic_array_audio_path, array_orientation_dir, speech_transcriptions_dir

def get_session_and_meas_ID_from_filename(filename: str):
    match = re.match(wav_file_pattern, filename)
    if match:
        database_num, session_num, measurement_num = map(int, match.groups())
    else:
        raise ValueError("Could not extract information from filename!")
    return database_num, session_num, measurement_num

def get_session_ID_from_dir(dirname: str):
    match = re.match(session_dir_pattern, dirname)
    if match:
        session_num = int(match.group(1))
    else:
        raise ValueError("Could not extract session ID from dirname!")
    return session_num

def parse_mm_ss_SSS(timestr: str) -> time:
    # return t.second}, t.minute, t.microsecond, tzone = t.tzinfo
    if timestr.endswith('.wav'):
        timestr = timestr[:-4]
    mm, ss, SSS = map(int, timestr.split('-'))
    print(f"mm = {mm}, ss = {ss}, SSS = {SSS}")
    return datetime.time(minute=mm, second=ss, microsecond=SSS * 1000)

def convert_int_to_float(data: np.ndarray):
    '''
    Function converts wave data from int32/int64 to float32.
    
    :param data: Data in presumably int32 or int64 as read form a a wav file.
    '''
    if data.dtype == np.int32:
        return data.astype(np.float32)/ np.iinfo(np.int32).max
    elif data.dtype == np.int64:
        return data.astype(np.float32)/ np.iinfo(np.int64).max
    elif data.dtype == np.float32:
        return data
    elif data.dtype == np.float64:
        if np.max(data>=1.0) or np.min(data<= -1.0):
            raise ValueError("Input data is not normalized between [-1,1] !!")
        return data.astype(np.float32)
    else:
        raise ValueError(f"convert_int_to_float(): Unknown data type for input {data.dtype}")

def read_csv_array_orientation_data(csv_file: Path):
    # Quaternions are sampled every 50 ms --> fs = 20 Hz
    if not csv_file.is_file():
        raise ValueError(f"read_array_orientation_data(): csv_file {csv_file} does not exist!")

    with open(csv_file, newline='') as f:
        line_count = sum(1 for _ in f) - 1
        # allocate arrays:
        time_seconds = np.zeros(line_count, dtype = np.float32)
        quaternions = np.zeros((line_count,4))
        f.seek(0)

        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for i, row in enumerate(reader):
            time_seconds[i] = float(row[1])
            quaternions[i] = [float(x) for x in row[2::]]
    fs_quaternions = 1./(time_seconds[1] - time_seconds[0])
    return time_seconds, quaternions, fs_quaternions

def read_csv_source_doa_data(doa_source_file: Path):
    # Doa data is sampled every 50 ms --> fs doa = 20 Hz
    # Coordinate system of azimuth/elevation?
    ################################################
    if not doa_source_file.is_file():
        raise ValueError(f"read_source_doa_data(): doa_source_file {doa_source_file} does not exist!")

    with open(doa_source_file, newline='') as f:
        line_count = sum(1 for _ in f) - 1
        print(f"line_count = {line_count}")
        # allocate arrays:
        time_seconds_doa = np.zeros(line_count, dtype = np.float32)
        doa_deg = np.zeros((line_count,2))
        f.seek(0)

        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for i, row in enumerate(reader):
            time_seconds_doa[i] = float(row[1])
            doa_deg[i] = [float(x) for x in row[2::]]
    fs_doa = 1./(time_seconds_doa[1] - time_seconds_doa[0])
    return time_seconds_doa, doa_deg, fs_doa

def get_head_tracking_fs():
    return 20.0

def get_corresponding_array_orientation_data(array_orientation_dir: Path, session_dir: Path, wav_file: Path, dT_head_tracking: float, t_max: float) -> list[dict]:
    '''
    Function gets the array orientation data (from a .json file) for a given wavefile and session. 
    
    :param array_orientation_dir: The root directory where all the orientation .json files are located.
    :param session_dir: The directory of the session where the wave file is located.
    :param wav_file: The wave file for which the orientation data is retrieved.
    :param dT_head_tracking: Sampling interval of the head tracking. Used for checks.
    :param t_max: Maximal time of the wave file. Used for checks.
    
    :returns: a list of dictionaries which contains all the frames of size t_max/dT_head_tracking
        Each frame contains orientations for each participant_ID. 
    '''
    # Find corresponding array orientation:
    array_orientation_file = array_orientation_dir / session_dir.name / (str(wav_file.stem) + ".json")
    if not array_orientation_file.is_file():
        raise ValueError(f"Corresponding orientation file {array_orientation_file} does not exist!")
    with open(array_orientation_file, 'r') as f:
        array_orientation_list = json.load(f)
    expected_N_frames_head_tracking = int(round(t_max/dT_head_tracking))
    if len(array_orientation_list) != expected_N_frames_head_tracking:
        raise ValueError(f"\nERROR: Expected {expected_N_frames_head_tracking} frames @20 Hz sampling rate, but read {len(array_orientation_list)} frames from .json file!")
    return array_orientation_list

def get_corresponding_lookup_when_participants_talk_lookup(speech_transcriptions_dir: Path, session_dir: Path, wav_file: Path, dT_head_tracking: float, t_max: float) -> defaultdict:
    '''
    Function return a lookup dictionary that looks like:
        lookup[participant_id][frame_id] = True if participant participant_id talks/is active in frame_id 
                                         = False if participant participant_id does not talk/is not active in frame_id
    
        Note that the lookup works even if a participant_id is not in a wave-file - it will always return False!
    
    Use with, e.g., doesParticipantTalkInFrame(lookup, 2, 119).
    
    :param speech_transcriptions_dir: The root directory where all the sppech transcription .json files are located.
    :param session_dir: The directory of the session where the wave file is located.
    :param wav_file: The wave file for which the orientation data is retrieved.
    :param dT_head_tracking: Sampling interval of the head tracking. Used for checks.
    :param t_max: Maximal time of the wave file. Used for checks.
    
    :returns: a defaultdict as a lookup table of booleans where you can fastly query whether a participant ID talked in a frame.
            Use, e.g., doesParticipantTalkInFrame() helper function.
    '''
    # Find corresponding transcription data:
    transcription_file = speech_transcriptions_dir / session_dir.name / (str(wav_file.stem) + ".json")
    if not transcription_file.is_file():
        raise ValueError(f"Corresponding trascription file {transcription_file} does not exist!")
    with open(transcription_file, 'r') as f:
        transcription_data = json.load(f)
    # We will create a lookup table (GPT-5) that will tell us if a participant talked in a specific frame:
    expected_N_frames_head_tracking = int(round(t_max/dT_head_tracking))
    # Factory method: whenever a new Participant_ID is seen, it will create a fresh list of 
    #  length expected_N_frames_head_tracking+1 (so indices go from 0 to expected_N_frames_head_tracking) 
    #  filled with False:
    does_participant_talk_in_frame_lookup = defaultdict(lambda: [False] * (expected_N_frames_head_tracking ))
    for row in transcription_data:
        pid = row["Participant_ID"]
        # Python index starts at 0, seems that .json index starts at 1:
        start, end = row["Start_Frame"]-1, row["End_Frame"]-1
        assert(end<=expected_N_frames_head_tracking and start>=0) # Neglihgible perf effects
        # Mark all frames in the interval as True
        does_participant_talk_in_frame_lookup[pid][start:end] = [True] * (end - start)
    return does_participant_talk_in_frame_lookup

def doesParticipantTalkInFrame(lookup: defaultdict, participantId: int, frameId: int) -> bool:
    '''
    Helper function that returns:
         = True if participant participant_id talks/is active in frame_id 
         = False if participant participant_id does not talk/is not active in frame_id
         
    :param lookup: A defaultdict, as built by get_corresponding_lookup_when_participants_talk_lookup().
    :param participantId: Participant ID.
    :param frameId: The Id of the frame, as sampled at get_head_tracking_fs().
    '''
    return lookup[participantId][frameId]

def isAnyActiveTalkerInFrame(lookup: defaultdict, frameId: int, includeSelfTalk: bool = False) -> bool:
    '''
    Helper function that returns:
    
        * True if there is any active talker in the frame
        * False if nobody is talking in the frame
    
    :param lookup: A defaultdict, as built by get_corresponding_lookup_when_participants_talk_lookup().
    :param frameId: The Id of the frame, as sampled at get_head_tracking_fs().
    :param includeSelfTalk: Boolean. If True, the wearer of the glasses is also considered as a talker.
    '''
    # Check external talkers first
    if any(lookup[talkerId][frameId] for talkerId in EXTERNAL_TALKER_IDS):
        return True
    # Optionally check self
    if includeSelfTalk and lookup[2][frameId]:
        return True
    return False

def isActiveTalkerInFrame(lookup: defaultdict, frameId: int, talkerIdsTuple: tuple[int]) -> bool:
    '''
    Same as isAnyActiveTalkerInFrame(), but targeted to specific talker IDs.
    
    Function that returns:
    
        * True if there is any active talker in the frame from IDs in talkerIdsTuple
        * False if nobody is talking in the frame
    
    :param lookup: A defaultdict, as built by get_corresponding_lookup_when_participants_talk_lookup().
    :param frameId: The Id of the frame, as sampled at get_head_tracking_fs().
    :param talkerIdsTuple: A tuple with talker IDs to check if talking. Comfing from, e.g., get_unique_part_IDs()
    '''
    # Check external talkers first
    if any(lookup[talkerId][frameId] for talkerId in talkerIdsTuple):
        return True
    return False

### # Vectorized versions: # ###
def areAnyActiveTalkersInFrames(
    lookup: defaultdict, 
    includeSelfTalk: bool = False
) -> np.ndarray:
    '''
    Vectorized version of isAnyActiveTalkerInFrame() that returns a boolean array, one entry per frame:

        * True if there is any active talker in that frame
        * False if nobody is talking in that frame

    :param lookup: A defaultdict, as built by get_corresponding_lookup_when_participants_talk_lookup().
                   Each lookup[talkerId] is assumed to be a 1D array-like of shape (n_frames,).
    :param includeSelfTalk: Boolean. If True, the wearer of the glasses (id=2) is also considered as a talker.
    :return: np.ndarray of shape (n_frames,), dtype=bool
    '''
    if EXTERNAL_TALKER_IDS:
        external_arrays = [np.asarray(lookup[talkerId], dtype=bool)
                           for talkerId in EXTERNAL_TALKER_IDS]
        external_active = np.any(np.vstack(external_arrays), axis=0)
    else:
        first_key = next(iter(lookup))
        n_frames = len(lookup[first_key])
        external_active = np.zeros(n_frames, dtype=bool)

    # Optionally include self
    if includeSelfTalk:
        self_active = np.asarray(lookup[2], dtype=bool)
        return np.logical_or(external_active, self_active)
    return external_active

def areActiveTalkersInFrames(
    lookup: defaultdict,
    talkerIdsTuple: tuple[int]
) -> np.ndarray:
    '''
    Vectorized version of isActiveTalkerInFrame().
    Returns a boolean array, one entry per frame:

        * True if any of the talkers in talkerIdsTuple is active in that frame
        * False if none of them are active

    :param lookup: defaultdict mapping talkerId -> 1D array of shape (n_frames,)
    :param talkerIdsTuple: Tuple of talker IDs to check
    :return: np.ndarray of shape (n_frames,), dtype=bool
    '''
    if not talkerIdsTuple:
        # No talkers specified -> return all-False vector sized from any entry
        first_key = next(iter(lookup))
        n_frames = len(lookup[first_key])
        return np.zeros(n_frames, dtype=bool)

    arrays = [np.asarray(lookup[t], dtype=bool) for t in talkerIdsTuple]
    return np.logical_or.reduce(arrays, axis=0) # Collapes across frames: axis = 0

def areTalkersMovingInFrames(six_DOF_data: defaultdict, talkerIdsTuple: tuple[int], thresholdMovementDegrees: float = 0.5) -> np.ndarray:
    '''
    Determine per-frame whether each talker moved more than a threshold: Function loops through the talker Ids and determines in each 
    frame whether the resepective talker moved:
    
        moved is considered in a frame if the total translation is such that more than about thresholdMovementDegrees is seen from the
        person wearing the microphone array.
        
        Note the function allows slow drifts in the data: if in each frame the movement is below threhsold, the function will return
        an array of False (meaning no movement is considered for that talker).
        
        For instance, ID=6 translates their head by 30 cm, which could mean a movement of about 1.8 degrees from ID=2 perspective. Then 
        this movement will not be considered movement and a False will be returned.
    
    :param lookup: defaultdict mapping talkerId -> 1D array of shape (n_frames,)
    :param talkerIdsTuple: Tuple of talker IDs to check
    :param thresholdMovementDegrees: A float with an approximate angle of how much the external talker's arc can be from one frame to another.
    :return: np.ndarray of shape (n_frames,), dtype=bool
    '''
    # We assume the worst case scenario for othjer talkers relative to ID=2: 1m[4]
    # We use chord length as proxy distance, where 1.0 = radius:
    thresholdMovementMeters = 1.0*np.sqrt(2.0-2.0*np.cos(np.deg2rad(thresholdMovementDegrees)))
    # Extract translation components (x,y,z) for selected talkers
    coords = six_DOF_data[np.arange(len(talkerIdsTuple)), :, 0:3]   # shape (n_talkers, n_frames, 3)
    
    # Compute per-frame radii
    radiiMeters = np.linalg.norm(coords, axis=2)

    # Compute forward differences, pad boundaries with 0
    movedMeters = np.diff(radiiMeters, axis=1, prepend=radiiMeters[:, :1])
    movedMeters[:, 0] = 0
    movedMeters[:, -1] = 0
    return np.abs(movedMeters) >= thresholdMovementMeters


def get_unique_part_IDs(array_orientation_list: list[dict], removeArrayTalker: bool = False) -> list[int]:
    '''
    Function retrieves the unique participant IDs from an orientation list read from a .json file. 
    
    :param array_orientation_list: The list of orientation frame data as read from an orientation .json file. 
                As read by, e.g., get_corresponding_array_orientation_data()
                
    :param removeArrayTalker: Boolean. If True, the ID of the array wearer is removed such that only the only the 
                    external talker IDs is returned. 
    
    :returns: A list with the unique participants ID in the list of frames. 
    '''
    #part_ID_set = set()
    #for frame_id in range(len(array_orientation_list)):
    #    for part in array_orientation_list[frame_id]["Participants"]:
    #        part_ID_set.add(part["Participant_ID"])
    part_ID_set = {
        part["Participant_ID"]
        for frame in array_orientation_list
        for part in frame["Participants"]
    }

    if removeArrayTalker:
        return removeArrayTalkerFromList(part_ID_set)
    return list(part_ID_set)

def removeArrayTalkerFromList(uniqueTalkerIdList: list[int]):
    uniqueTalkerIdList = set(uniqueTalkerIdList)
    uniqueTalkerIdList.discard(2)
    uniqueTalkerIdList = list(uniqueTalkerIdList)
    return uniqueTalkerIdList

def unpack_6DOF_data(array_orientation_list: list[dict], unique_part_ID_list: list[int]):
    '''
    Function unpacks a read array_orientation_list from a .json file.
    
    :param array_orientation_list: The list of orientation frame data as read from an orientation .json file. 
    :param unique_part_ID_list: A list of unique IDs from the array_orientation_list. Coming from get_unique_part_IDs().
    
    :returns:
        :ret six_DOF_data: Numpy array of size (N_participants, N_frames, 7) containing 6 DOF data:
                            [part_idx, frame_ID, x, y, z, q_x, q_y, q_z, q_w]
                            where part_idx is the same ID as in unique_part_ID_list. E.g., if unique_part_ID_list = {2,9,10}, 
                            part_idx = 1 will point towards part ID = 9 
        :ret is_upside_down: Numpy array with a boolean that stores isUpsideDown from the .json file
                        Shape of (N_participants, N_frames)
    '''
    six_DOF_data = np.zeros( (len(unique_part_ID_list), len(array_orientation_list), 7), dtype = np.float32)
    is_upside_down = np.zeros((len(unique_part_ID_list), len(array_orientation_list)), dtype = np.bool_ )
    for frame_idx in range(len(array_orientation_list)):
        for participant in array_orientation_list[frame_idx]["Participants"]:
            six_dof_part_data_idx = unique_part_ID_list.index(participant["Participant_ID"])
            six_DOF_data[six_dof_part_data_idx, frame_idx, 0] = participant["Position_X"]
            six_DOF_data[six_dof_part_data_idx, frame_idx, 1] = participant["Position_Y"]
            six_DOF_data[six_dof_part_data_idx, frame_idx, 2] = participant["Position_Z"]
            six_DOF_data[six_dof_part_data_idx, frame_idx, 3] = participant["Quaternion_X"]
            six_DOF_data[six_dof_part_data_idx, frame_idx, 4] = participant["Quaternion_Y"]
            six_DOF_data[six_dof_part_data_idx, frame_idx, 5] = participant["Quaternion_Z"]
            six_DOF_data[six_dof_part_data_idx, frame_idx, 6] = participant["Quaternion_W"]
            is_upside_down[six_dof_part_data_idx, frame_idx] =  participant["isUpSideDown"]
    return six_DOF_data, is_upside_down

def read_smaller_dataset_DEPRECATED(path_to_dataset = r"D:\Temp_S230_Database\core_train_dataset_1\Main\Train\Dataset_1"):
    ## DEPRECATED - no need ot use this!
    warnings.warn("Deprecated dataset! Don't use this one!")
    mic_array_audio_path = Path(path_to_dataset) / "Microphone_Array_Audio"
    # Session 1: 2880000 @ fs=48 kHz --> 60 seconds clips
    array_orientation_dir = Path(path_to_dataset) / "Array_Orientation"
    
    for session_dir in mic_array_audio_path.iterdir():
        if session_dir.is_dir():
            session_folder = mic_array_audio_path / session_dir
            for wav_file in session_folder.glob("*.wav"):
                print("Found WAV file:", wav_file.name)
                database_num, session_num, measurement_num  = get_session_and_meas_ID_from_filename(wav_file.name)
                print(f"\tDatabase: {database_num}, Session: {session_num}, Measurement: {measurement_num}")
                
                # Find corresponding array orientation:
                array_orientation_file = array_orientation_dir / session_dir.name / f"ori_D{database_num}_S{database_num}_M{measurement_num:02d}.csv"
                if not array_orientation_file.is_file():
                    raise ValueError(f"Corresponding orientation file {array_orientation_file} does not exist!")
                print(f"\t\tfound orientation_file = '{array_orientation_file.name}'")
                time_seconds_quaternions, quaternions, fs_quaternions = read_csv_array_orientation_data(array_orientation_file)
                dT_arraY = 1./fs_quaternions
                print(f"time_seconds = {time_seconds_quaternions}")
                print(f"dT_Quaternions = {np.round(dT_arraY*1e3,2)} [ms], fs quaternions = {fs_quaternions} Hz")
                
                fs, data = wavfile.read(wav_file)
                time_ = np.arange(len(data))/fs
                print(f"\t Read data @fs = {fs}, data = {type(data)} of shape {np.shape(data)} of type {type(data[0,0])}")
                
                max_time_audio = np.max(time_)
                max_time_quaternions = np.max(time_seconds_quaternions)
                print(f"max_time_audio = {np.round(max_time_audio,5)} [s]")
                print(f"max_time_quaternions = {np.round(max_time_quaternions,5)} [s]")
                
                ###################################################
                # playback_sterero_signal(data[0:4*fs, 4:6], fs)
                # Convert data to float32:
                data = convert_int_to_float(data)
                
                
                plt.figure(figsize=(15,10))
                plt.plot(time_[0:fs], data[0:fs, 4], label = "Left ear")
                plt.plot(time_[0:fs], data[0:fs, 5], linestyle = '--', label = "Right ear")
                plt.grid(True, which = 'both')
                plt.xlabel("Time [s]", fontsize=14.0)
                plt.ylabel("Amplitude", fontsize=14.0)
                plt.legend(fontsize=14.0)
                plt.tight_layout()
                plt.show()
                sys.exit()
            print("="*30)

if __name__ == '__main__':
    
    print("Start miniProjIO...")
    # read_smaller_dataset()
    
    #===========================================================================
    # Read main dataset example:
    #===========================================================================
    path_to_dataset = r"D:\Temp_S230_Database\Main"
    mic_array_audio_path, array_orientation_dir, speech_transcriptions_dir = get_paths(path_to_dataset)
    
    fs_head_tracking = get_head_tracking_fs()
    dT_head_tracking = 1.0/fs_head_tracking
    
    # Iterate over sessions:
    start = time.perf_counter()
    for session_dir in mic_array_audio_path.iterdir():
        if session_dir.is_dir():
            session_folder = mic_array_audio_path / session_dir
            ses_ID = get_session_ID_from_dir(session_dir.name)
            #print(f"Session {session_dir.name}, ID = {ses_ID}")
            # Iterate over wave-files in each session:
            for wav_file in session_folder.glob("*.wav"):
                print("  Found WAV file:", wav_file.name)
                t = parse_mm_ss_SSS(wav_file.name) 
                print(f"\t {t}, s={t.second}, m={t.minute}, us={t.microsecond}, tzone = {t.tzinfo}")
                
                fs, data = wavfile.read(wav_file)
                data = convert_int_to_float(data)
                N_taps, N_audio_channels = data.shape
                t_max = (1.0*len(data) -1) / fs
                N_frames = int(round(t_max/dT_head_tracking))
                N_samples_per_frame = int(round(len(data)/N_frames))
                assert(N_samples_per_frame == fs / fs_head_tracking)
                print(f"\t Read data @fs = {fs}, data = {type(data)} of shape {np.shape(data)} of type {type(data[0,0])}")
                #===============================================================
                # Extract relevant data for the wavefile:
                #===============================================================
                array_orientation_list = get_corresponding_array_orientation_data(array_orientation_dir, session_dir, wav_file, dT_head_tracking, t_max)
                lookup = get_corresponding_lookup_when_participants_talk_lookup(speech_transcriptions_dir, session_dir, wav_file, dT_head_tracking, t_max)
                
                part_unique_ID_list = get_unique_part_IDs(array_orientation_list)
                external_part_unique_ID_list = removeArrayTalkerFromList(part_unique_ID_list)
                
                # Six_DOF_data: [part_idx, frame_ID, x, y, z, q_x, q_y, q_z, q_w]
                six_DOF_data, is_upside_down = unpack_6DOF_data(array_orientation_list, part_unique_ID_list)
                
                ## Compute/extract useful data:
                ##################################
                # Glasses wearer position (first 3) and rotation (last 3):
                y_ground_truth = six_DOF_data[part_unique_ID_list.index(2), :]
                
                # Example of input:
                #  Here, the feature vector will be the raw audio data between [-1.0, 1.0]. 
                #    Note although the data is between -1,1, it has not been centered (mean is probably
                #  clsoe to 0.0 since we don't have DC, but variance is not 1.0)
                N_features = N_samples_per_frame
                # Swap axes to (channels, samples_per_frame, frames) and force a copy:
                # x will be x(channels, frame, :) samples in frame of channel:
                x = data.reshape(N_audio_channels, N_frames, N_features).copy()
                
                # Example 2 of input:
                # Now you can try taking the Fourier transform of each frame as a feature vector
                #  Note we can discard half of the Fourier coefficients since they don't bring any new
                #  information: the ones for negative freqs are conjugate symmetric of pos ones. Remember
                #  we double the floats from going from real to complex, so nothing it gained/lossed memory wise:
                X = np.fft.fft(x, axis=2) # or use rfft - a bit faster
                mask = np.fft.fftfreq(N_features)>=0
                X=X[:,:,mask]
                                
                # Vector of bools when any external talks:
                any_specific_talkers_vector = areActiveTalkersInFrames(lookup, external_part_unique_ID_list)
                # Vector of bools when the glasses wearer talks:
                self_talk_vector = areActiveTalkersInFrames(lookup, [2])
                
                # Vector of bools showing which external talker moved more:
                moved = areTalkersMovingInFrames(six_DOF_data, external_part_unique_ID_list)
                # Vector if any talker moved:
                any_moved = moved.any(axis=0) # (n_frames,)

                
                # If you want, you can loop through frames as well (not efficient):
                for frame_idx in range(N_frames):
                    x_ = x[:, frame_idx, :]
                    X_ = X[:, frame_idx, :]
                    
                    y_ = y_ground_truth[frame_idx]
                    
                    any_specific_talker = any_specific_talkers_vector[frame_idx]
                    # Same thing:
                    any_specific_talker = isAnyActiveTalkerInFrame(lookup, frame_idx)
                    talker_6_talks = doesParticipantTalkInFrame(lookup, 6, frame_idx)
                                
                
                # If you want to listen to 2 channels: here, the binaural channels 4 and 5:
                #playback.playback_sterero_signal(data[0:4*fs, 4:6], fs)
                
                # Feel free to remove this and loop all the way:
                print("Quitting loop!")
                sys.exit()
    end = time.perf_counter()
    print(f"Script took {datetime.timedelta(seconds=end-start)}")
    
    print(f"...End miniProjIO... Total time: {datetime.timedelta(seconds=time.perf_counter()-start)} [hh:mm:ss.dddd]")