import numpy as np
import os 
import re
import struct
from osrparse import Replay,Key
from coordinates_utils import normalize_coordinates

def process_replay(replay_path):

    
    loaded_replay_data = {"time_deltas":[],
                   "normalized_cursor_pos":[],
                   "key_presses":[]}


    replay = Replay.from_path(replay_path)

    for event in replay.replay_data:
        loaded_replay_data["time_deltas"].append(event.time_delta)
        normalized_x,normalized_y = normalize_coordinates(event.x,event.y)
        loaded_replay_data["normalized_cursor_pos"].append((normalized_x,normalized_y,))
        loaded_replay_data["key_presses"].append(one_hot_encode_key_pressed(event.keys))
 

    
    return loaded_replay_data




        


# def process_replay_file(replay_file_path, beatmap_data):
#     replay_data = np.load(replay_file_path)

#     # Get the beatmap ID from the replay file name
#     beatmap_id = int(os.path.basename(replay_file_path).split('.')[0].split('_')[1])

#     # Get the beatmap data
#     beatmap_info = beatmap_data[beatmap_id]

#     # Get the hitobject data
#     hitobjects = beatmap_info['hitobjects']
#     hitobject_times = [int(ho['time']) for ho in hitobjects]

#     # Get the replay data
#     replay_times = replay_data['time']
#     replay_keys = replay_data['keys']

#     # Initialize the state and action arrays
#     num_steps = len(hitobject_times)
#     state = np.zeros((num_steps, 4))
#     action = np.zeros((num_steps, 2))

#     # Process each time step
#     for i, ho_time in enumerate(hitobject_times):
#         # Find the closest replay time to the hitobject time
#         replay_idx = np.argmin(np.abs(replay_times - ho_time))

#         # Set the state information
#         state[i, 0] = replay_data['x'][replay_idx]
#         state[i, 1] = replay_data['y'][replay_idx]
#         state[i, 2] = replay_data['keys'][replay_idx]
#         state[i, 3] = ho_time

#         # Set the action information
#         if replay_keys[replay_idx] & 1:  # left mouse button
#             action[i, 0] = 1
#         if replay_keys[replay_idx] & 2:  # right mouse button
#             action[i, 1] = 1

#     return state, action


def one_hot_encode_key_pressed(key_pressed):
    is_k1_pressed = bool(key_pressed & Key.K1) or bool(key_pressed & Key.M1)
    is_k2_pressed = bool(key_pressed & Key.K2) or bool(key_pressed & Key.M2)

    return [int(is_k1_pressed),int(is_k2_pressed)]

