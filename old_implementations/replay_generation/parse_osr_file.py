from osrparse import Replay,Key
import numpy as np
from coordinates_utils import normalize,X_LOWER_BOUND,X_UPPER_BOUND,Y_LOWER_BOUND,Y_UPPER_BOUND

"""
Script permettant de parser un fichier .OSR, de normaliser toutes ses données (normalisation,one-hot encoding,padding,...) 
et de les séparer en chunks basés sur des intervals de temps.

Par exemple si l'interval de temps choisi est 1000 ms, le premier chunk contiendra les données du replay
ayant lieu entre 0 et 1000 ms, le deuxième entre 1000ms et 2000ms, etc..

L'interval de temps est ensuite aligné sur les données de la beatmap correspondant au replay.

Par exemple, si les premiers objets de la beatmap se situent dans l'interval de temps 2000/3000ms,
alors les données de replay qui se passent avant ce timing seront ignorés car inutiles.

De cette manière pour chaque chunk beatmap/replay on a les objets qui se situent dans ce timing
et les actions qui se situent au meme timing dans le replay.


Format des données pour le replay (target du LSTM):

1 action du replay contient les informations suivantes :

time (Normalized)
x (Normalized)
y (Normalized)
Keys (One hot encoded -> [0,0] for no key, [1,0] for K1 pressed, etc..)




"""



TIME_INTERVAL = 1000 # In milliseconds
LENGTH_THRESHOLD = 360000
BEGINNING_THRESHOLD = 2000
CHUNK_SIZE = 150

def one_hot_encode_key_pressed(key_pressed):
    is_k1_pressed = bool(key_pressed & Key.K1) or bool(key_pressed & Key.M1)
    is_k2_pressed = bool(key_pressed & Key.K2) or bool(key_pressed & Key.M2)

    return [int(is_k1_pressed),int(is_k2_pressed)]



def chunk_replay_data(data,replay_name, chunk_size=CHUNK_SIZE, time_interval=TIME_INTERVAL,normalized=False):

    start_thresh = BEGINNING_THRESHOLD
    if normalized:
        time_interval = normalize(time_interval,lower_bound=0,upper_bound=LENGTH_THRESHOLD)
        start_thresh = normalize(BEGINNING_THRESHOLD,lower_bound=0,upper_bound=LENGTH_THRESHOLD)
        

    data = sorted(data,key=lambda x : x[0])
    chunks = []
    current_chunk = []
    current_time = 0

    padding = [0,0,0,0,0]

    for replay in data:
        replay_time = replay[0]  

        if replay_time - current_time < time_interval and len(current_chunk) < chunk_size and replay_time > 0:
                    current_chunk.append(replay)


        if replay_time - current_time >= time_interval or len(current_chunk) == chunk_size:
            if len(current_chunk) == chunk_size:
                print(f"Chunk capacity was reached in {replay_name} at {current_time}, likely due to player tabbing out")

            else :
                while len(current_chunk) < chunk_size:
                    current_chunk.append(padding)  


                chunks.append(current_chunk)
            current_chunk = []

            while current_time + time_interval <= replay_time:
                current_time += time_interval

        

    if current_chunk:
        while len(current_chunk) < chunk_size:
            current_chunk.append(padding)  

        chunks.append(current_chunk)

    # if abs(chunks[0][0][0] - chunks[1][0][0]) > normalize(BEGINNING_THRESHOLD,lower_bound=0,upper_bound=LENGTH_THRESHOLD):
    #     chunks.pop(0) 

    if abs(chunks[0][0][0] - chunks[1][0][0]) > start_thresh:
        chunks.pop(0) 

    return chunks



def parse_osr_file_abs_time(osr_file_path):
    replay = Replay.from_path(osr_file_path)

    replay_events = []
    replay_events_chunk = []  # List to store a chunk of replay events
    current_time = 0
    for event in replay.replay_data:
        current_time += event.time_delta
        absolute_time = [normalize(current_time,lower_bound=0,upper_bound=LENGTH_THRESHOLD)]
        x = [normalize(event.x, X_LOWER_BOUND, X_UPPER_BOUND)]
        y = [normalize(event.y, Y_LOWER_BOUND, Y_UPPER_BOUND)]
        keys = one_hot_encode_key_pressed(event.keys)
        
        event_data = np.concatenate([absolute_time, x, y, keys])

        if 0 <= event_data[1] <= 1 and 0 <= event_data[2] <= 1:
            replay_events_chunk.append(event_data.tolist())  # Convert numpy array to list before appending

    return replay_events_chunk


def parse_osr_file_and_chunk(osr_file_path):
    parsed_datas = parse_osr_file_abs_time(osr_file_path)
    return chunk_replay_data(parsed_datas,osr_file_path,normalized=True)


def parse_osr_file_and_chunk_aligned(osr_file_path,beatmap_chunks):
    replay_chunks = parse_osr_file_and_chunk(osr_file_path)
    return align_chunks(replay_chunks,beatmap_chunks)



def align_chunks(replay_chunks, beatmap_chunks,time_interval=1000,normalized=True):

    if normalized : 
        time_interval = normalize(time_interval,lower_bound=0,upper_bound=LENGTH_THRESHOLD)

    aligned_replay_chunks = []
    chunk_index = 0  # index to keep track of the current beatmap chunk

    for replay_chunk in replay_chunks:
        replay_chunk_start_time = replay_chunk[0][0]  # start time of the first action in the replay chunk

        #print(chunk_index)
        # get the current beatmap chunk start time
        beatmap_chunk_start_time = beatmap_chunks[chunk_index][0][0]  # start time of the first hitobject in the chunk


        # if the start time of the replay chunk is within the start time of the beatmap chunk
        if abs(replay_chunk_start_time - beatmap_chunk_start_time) <= time_interval:
            aligned_replay_chunks.append(replay_chunk)
            chunk_index += 1  # move to the next beatmap chunk

        # if we have processed all the beatmap chunks, stop the process
        if chunk_index >= len(beatmap_chunks):
            break

    return aligned_replay_chunks

