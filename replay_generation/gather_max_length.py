from parse_osu_map_file import *
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import osrparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from parse_osr_file import *
from slider import Replay
import datetime
from time import perf_counter
from coordinates_utils import normalize

MAX_SLIDER_POINTS = 75
SLIDER_LENGTH_THRESHOLD = 3000
SLIDER_POINTS_THRESHOLD = 75
CHUNK_SIZE = 10
LENGTH_THRESHOLD = 360000 
BEGINNING_THRESHOLD = 2000 # Time used to consider if two chunks are too far apart


def count_hit_objets(beatmap):
    datas = parse_beatmap_file(beatmap)
    return len(datas)* CHUNK_SIZE


def count_player_action(replay):
    try:
        datas = osrparse.Replay.from_path(replay)
        return len(datas.replay_data)
    except:
        print(f"Error when parsing {replay}")
        return 0 


def add_object_count_column(bm_path,index):
    currentwd = os.getcwd()
    os.chdir(bm_path)
    with multiprocessing.Pool() as p:
        index["hitobject_count"] = p.map(count_hit_objets, tqdm([beatmapHash+".osu" for beatmapHash in index["beatmapHash"].tolist()]))
    os.chdir(currentwd)
    # print(len(index[index['beatmap_length'] <= LENGTH_THRESHOLD]))
    return index

def add_action_count(replay_path,index):
    currentwd = os.getcwd()
    os.chdir(replay_path)
    with multiprocessing.Pool() as p:
        index["action_count"] = p.map(count_player_action, tqdm([replayHash+".osr" for replayHash in index["replayHash"].tolist()]))
    os.chdir(currentwd)
    # print(len(index[index['beatmap_length'] <= LENGTH_THRESHOLD]))
    return index


def chunk_replay_data(data, chunk_size=200, time_interval=1000,normalized=False):

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
        replay_time = replay[0]  # assuming the time is the first element in replay

        # If the current replay is within the current time interval and the chunk is not full
        if replay_time - current_time < time_interval and len(current_chunk) < chunk_size:
                    current_chunk.append(replay)

        # If we moved to the next time interval or the current chunk is full

        if replay_time - current_time >= time_interval or len(current_chunk) == chunk_size:
            # Pad the current chunk if it's not full
            if len(current_chunk) == chunk_size:
                print("Chunk capacity was reached !")
                print(time_interval,replay_time,current_time)
            while len(current_chunk) < chunk_size:
                current_chunk.append(padding)  

            chunks.append(current_chunk)
            current_chunk = []

            # move to the next time interval that contains the current replay
            while current_time + time_interval <= replay_time:
                current_time += time_interval

        

    # don't forget to append the last chunk if it's not empty and pad if necessary
    if current_chunk:
        while len(current_chunk) < chunk_size:
            current_chunk.append(padding)  

        chunks.append(current_chunk)

    # if abs(chunks[0][0][0] - chunks[1][0][0]) > normalize(BEGINNING_THRESHOLD,lower_bound=0,upper_bound=LENGTH_THRESHOLD):
    #     chunks.pop(0) 

    if abs(chunks[0][0][0] - chunks[1][0][0]) > start_thresh:
        chunks.pop(0) 

    return chunks


def chunk_map_data(data,chunk_size=20, time_interval=1000):

    time_interval = normalize(time_interval,lower_bound=0,upper_bound=LENGTH_THRESHOLD)

    data = sorted(data, key=lambda x: x[0])

    chunks = []
    current_chunk = []
    current_time = data[0][0] # Initialize at first hitobject timing
    padding = [0] * 161

    for hitobject in data:
        if hitobject[0] <= current_time and len(current_chunk) < chunk_size:
            current_chunk.append(hitobject)
        else:
            # Add padding if necessary
            if len(current_chunk) > 0 and current_chunk[-1][2] == 1:  # If it's a slider
                new_chunk_start = [current_chunk[-1]]
            else:
                new_chunk_start = []
            
            # Add padding if necessary
            while len(current_chunk) < chunk_size:
                current_chunk.append(padding)
            chunks.append(current_chunk)

            current_chunk = new_chunk_start
            current_chunk.append(hitobject)
            current_time += time_interval

    # Add the last chunk
    while len(current_chunk) < chunk_size:
        current_chunk.append(padding)
    chunks.append(current_chunk)

    return chunks


if __name__ == '__main__':

    path = 'D:\\osu!rdr_dataset\\beatmaps'
    replay_folder = 'D:\\osu!rdr_dataset\\replays'
    
    #index = pd.read_csv("D:\\osu!rdr_dataset\\final_cleaned_index.csv")

    #print(index[index["beatmap-BPMMax"] == 320]["beatmapHash"])

    #print(count_player_action(f"{replay_folder}\\osr\\d7000ac77246ddd2c503ffe3d0bff61c.osr"))
    # index = index.head(20000)

    #print(index["beatmap_length"].sort_values(ascending=True))




    # # max_lengths  = index[index["beatmap_length"] == 359909]["replayHash"]
    # # print(max_lengths)

    # print("processing beatmaps")
    # index = add_object_count_column(path,index)
    # # index.to_csv("final_cleaned_index4685.csv",header=True,index=True,mode="w+")
    # print("processing replays")
    # index = add_action_count(replay_folder+"\\osr\\",index)

    # index["action_object_ratio"] = index["action_count"] / index['hitobject_count']

    # index["action_object_ratio"].to_csv("action_object_ratio.csv")

    # print(index["action_object_ratio"].mean())
    # print(index["action_object_ratio"].std())
    # print(index["action_object_ratio"].var())
    

    # index.to_csv("final_cleaned_index4685.csv",header=True,index=True,mode="w+")
    # print("osr parser")
    # deb = perf_counter()
    # for i in tqdm(range(1000),leave=True):


    test_map_parse = parse_beatmap_file_and_chunk(path+"\\d5afbe9f596be543c97b93783aa5833a.osu")

    #test_osrparser = parse_osr_file_and_chunk(replay_folder+"\\osr\\d8ee63832c0ed9245ee20f3720b910a7.osr")

    test_osrparser2 = parse_osr_file_and_chunk_aligned(replay_folder+"\\osr\\d8ee63832c0ed9245ee20f3720b910a7.osr",test_map_parse)

    
    #koto = chunk_replay_data(test_osrparser,normalized=True)
    # print(len([event for event in test_osrparser if 0.60540771484375 < event[0] < 0.60540771484375 + normalize(2000,lower_bound=0,upper_bound=LENGTH_THRESHOLD)]))

    # test_chunk = chunk_replay_data(test_osrparser,normalized=True)

    # print(test_chunk[0])
    # print(test_chunk[1])
    # print(test_chunk[2])
    print(len(test_map_parse))


    print(test_map_parse[98])
    # print(len(test_osrparser))

    # a = [e[0]*360000 for e in test_osrparser[30] if e[0] != 0]

    # print(a[0],a[-1])

    # #print(test_map_parse[0])

    # b = [e[0]*360000 for e in test_map_parse[28] if e[0] != 0]

    # print(b[0],b[-1])

    # print(abs(a[0] - b[0]))



    # print(len(test_map_parse))
    # print(len(test_osrparser))

    # print(test_osrparser[0][0][0])
    # print(test_map_parse[0][0][0])

    # print(test_osrparser[0][-1][0])
    # print(test_map_parse[0][-1][0])
    # print("__________________________________________")
    # print(test_osrparser[0][0])
    # print(test_map_parse[0][0])

    # print(test_osrparser[-1][0])
    # print(test_map_parse[-1][0])

    # print(test_map_chunk[45])





    # for idx,chunk in enumerate(test_map_chunk):
    #     if len(chunk) != 100:
    #         print("caca")
    #     for hitobject in chunk:
    #         if not isinstance(hitobject,list):
    #             print(idx,hitobject)
    #         if len(hitobject) != 161:
    #             print(len(hitobject))
    # test_beatmap_parser = parse_beatmap_file_flattened(path+"\\cfb4064ea4e1a3d99c8f7bc4d61edc14.osu")

    # #print(test_beatmap_parser[0])
    # domo = chunk_map_data(test_beatmap_parser,10,time_interval=500//LENGTH_THRESHOLD)
    # print(domo[0])



    #print(test_osrparser[:20])
    # test_chunk = chunk_data(test_osrparser,100,500)

    # print(test_chunk[0])


    # print(f"osr parser : {perf_counter() - deb }")

    # # print(len(test_osrparser))
    # print(test_osrparser[-1])
    # print([e.time_delta for e in test_osrparser])
    # print(list(np.cumsum([e.time_delta for e in test_osrparser])))
    # print("_"*80)

    # # test_npy_format = np.load(replay_folder+"\\npy\\3ad7fc0af16ff39a4d268048a4674f9a.npy")
    # # print(test_npy_format.shape)
    # # # print(test_npy_format[-1])
    # deb = perf_counter()
    # for i in tqdm(range(1000)):

    #     test_slider = Replay.from_path(replay_folder+"\\osr\\833b3bc7c8673c1576deff744e53c315.osr",retrieve_beatmap=False)
    #     milisecond_list = ([action.offset.days * 24 * 60 * 60 * 1000 + action.offset.seconds * 1000 + action.offset.microseconds // 1000 for action in test_slider.actions])

    # # print(chunk_data(test_slider.actions))

    # print(f"slider parser : {perf_counter() - deb }")
    # max_bm_objects = parse_beatmap_file(f"{path}\\b52a7e49045a76be5407390884bc9fa9.osu")
    # print(max_bm_objects)

    #print(count_hit_objets(f"{path}\\b52a7e49045a76be5407390884bc9fa9.osu"))
    #print("Max amount of objects : ",len(max_bm_objects["hit_objects"]))

    # for replay in max_lengths:
    #     replay_data = osrparse.Replay.from_path(f"{replay_folder}\\{replay}.osr")
    #     print(len(replay_data.replay_data))



