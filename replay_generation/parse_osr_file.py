from osrparse import Replay,Key
import numpy as np
from coordinates_utils import normalize,X_LOWER_BOUND,X_UPPER_BOUND,Y_LOWER_BOUND,Y_UPPER_BOUND
from keras.utils import Sequence,pad_sequences


BATCH_LENGTH = 2048
TIME_INTERVAL = 500 # In milliseconds
LENGTH_THRESHOLD = 360000
BEGINNING_THRESHOLD = 2000

def one_hot_encode_key_pressed(key_pressed):
    is_k1_pressed = bool(key_pressed & Key.K1) or bool(key_pressed & Key.M1)
    is_k2_pressed = bool(key_pressed & Key.K2) or bool(key_pressed & Key.M2)

    return [int(is_k1_pressed),int(is_k2_pressed)]


def parse_osr_file(osr_file_path):
    replay = Replay.from_path(osr_file_path)

    replay_events = []
    replay_events_chunk = []  # List to store a chunk of replay events
    for event in replay.replay_data:

        time_delta = [event.time_delta]
        x = [normalize(event.x, X_LOWER_BOUND, X_UPPER_BOUND)]
        y = [normalize(event.y, Y_LOWER_BOUND, Y_UPPER_BOUND)]
        keys = one_hot_encode_key_pressed(event.keys)
        
        event_data = np.concatenate([time_delta, x, y, keys])

        replay_events_chunk.append(event_data.tolist())  # Convert numpy array to list before appending

        # If the chunk has reached the desired length, append it to the list of chunks
        if len(replay_events_chunk) == BATCH_LENGTH:
            replay_events.append(replay_events_chunk)
            replay_events_chunk = []

    # If there are any leftover replay events, append them as a smaller chunk
    if replay_events_chunk:
        padding_length = BATCH_LENGTH - len(replay_events_chunk)
        for _ in range(padding_length):
            # Append a list of zeros to match the dimension of event_data
            replay_events_chunk.append([0.] * len(event_data))

        replay_events.append(replay_events_chunk)

    return np.array(replay_events, dtype=np.float16)



def chunk_replay_data(data,replay_name, chunk_size=120, time_interval=1000,normalized=False):

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

        replay_events_chunk.append(event_data.tolist())  # Convert numpy array to list before appending

    return replay_events_chunk


def parse_osr_file_and_chunk(osr_file_path):
    parsed_datas = parse_osr_file_abs_time(osr_file_path)
    return chunk_replay_data(parsed_datas,osr_file_path,normalized=True)


def parse_osr_file_and_chunk_aligned(osr_file_path,beatmap_chunks):
    replay_chunks = parse_osr_file_and_chunk(osr_file_path)
    return align_chunks(replay_chunks,beatmap_chunks)
    



def chunk_replay_data_aligned(data, replay_name, beatmap_chunks, chunk_size=120, time_interval=1000, normalized=False):

    start_thresh = BEGINNING_THRESHOLD
    if normalized:
        time_interval = normalize(time_interval, lower_bound=0, upper_bound=LENGTH_THRESHOLD)
        start_thresh = normalize(BEGINNING_THRESHOLD, lower_bound=0, upper_bound=LENGTH_THRESHOLD)
        
    data = sorted(data, key=lambda x: x[0])

    padding = [0, 0, 0, 0, 0]
    replay_chunks = []
    current_chunk = []
    chunk_index = 0  # index to keep track of the current beatmap chunk

    for replay in data:
        replay_time = replay[0]  # assuming the time is the first element in replay

        # get the current beatmap chunk start time and end time
        print(chunk_index)
        beatmap_chunk_timings_no_padding = [e[0] for e in beatmap_chunks[chunk_index] if e[0] != 0]
        
        beatmap_chunk_start_time = beatmap_chunk_timings_no_padding[0]  # start time of the first hitobject in the chunk
        beatmap_chunk_end_time = beatmap_chunk_timings_no_padding[-1]  # end time of the last hitobject in the chunk

        # if the replay time is within the current beatmap chunk time interval
        if beatmap_chunk_start_time <= replay_time < beatmap_chunk_end_time:
            # if the current replay chunk is not full, add the replay
            if len(current_chunk) < chunk_size:
                current_chunk.append(replay)

        # if we moved to the next beatmap chunk time interval or the current replay chunk is full
        if replay_time >= beatmap_chunk_end_time or len(current_chunk) == chunk_size:
            # pad the current replay chunk if it's not full
            if len(current_chunk) == chunk_size:
                print(f"Chunk capacity was reached in {replay_name} at {replay_time}, likely due to player tabbing out")

            else :
                while len(current_chunk) < chunk_size:
                    current_chunk.append(padding)  


                replay_chunks.append(current_chunk)
            current_chunk = []
            # move to the next beatmap chunk
            chunk_index += 1

    # don't forget to append the last replay chunk if it's not empty and pad if necessary
    if current_chunk:
        while len(current_chunk) < chunk_size:
            current_chunk.append(padding)
        replay_chunks.append(current_chunk)

    # if the start time of the first replay chunk is too far from the start time of the first beatmap chunk, remove it
    if abs(replay_chunks[0][0][0] - beatmap_chunks[0][0][0]) > start_thresh:
        replay_chunks.pop(0)

    return replay_chunks



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

        # replay_chunk_no_padding = [e[0] for e in replay_chunk if e[0]!=0]
        # bm_chunk_no_padding = [e[0] for e in beatmap_chunks[chunk_index] if e[0]!=0]

        # if bm_chunk_no_padding and not replay_chunk_no_padding:
        #     print(bm_chunk_no_padding[0]*360000,bm_chunk_no_padding[-1]*360000)

        # if replay_chunk_no_padding and bm_chunk_no_padding:
        #     print(replay_chunk_no_padding[0]*360000,replay_chunk_no_padding[-1]*360000 ,bm_chunk_no_padding[0]*360000,bm_chunk_no_padding[-1]*360000,abs(replay_chunk_start_time - beatmap_chunk_start_time))


        # if the start time of the replay chunk is within the start time of the beatmap chunk
        if abs(replay_chunk_start_time - beatmap_chunk_start_time) <= time_interval:
            aligned_replay_chunks.append(replay_chunk)
            chunk_index += 1  # move to the next beatmap chunk

        # if we have processed all the beatmap chunks, stop the process
        if chunk_index >= len(beatmap_chunks):
            break

    return aligned_replay_chunks






if __name__ == "__main__":


    test_replay_path = "D:\\osu!rdr_dataset\\replays\\osr\\f937d9228eabf9a26d0ff1c1feeca3ce.osr"

    replay_data = Replay.from_path(test_replay_path)
    
    # for i in range(7227,len(replay_data.replay_data)):
    #     replay_data.replay_data[i].x = 0
    #     replay_data.replay_data[i].y = 0

    
    test_parse = parse_osr_file(test_replay_path)
    print(test_parse.shape)

    

    # print([(idx,event) for idx,event in enumerate(test_parse) if any([event["y"] < Y_LOWER_BOUND,
    #                                              event["y"] > Y_UPPER_BOUND,
    #                                              event["x"] < X_LOWER_BOUND,
    #                                              event["x"] > X_UPPER_BOUND,
    #                                              event["time_delta"] < 0])])



