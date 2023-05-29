import numpy as np
import os
import pandas as pd
import multiprocessing
from parse_osu_map_file import parse_beatmap_file
from parse_osr_file import parse_osr_file
from tqdm import tqdm
import time





def parse_files(args):

    output_dir = f"D:\\osu!rdr_dataset\\preprocessed_datas"

    beatmap_file, replay_file, idx = args

    # Initialize timing dictionary for this process
    parse_times = {
        "parse_beatmap_file": 0.0,
        "parse_osr_file": 0.0
    }

    try:
        start_time = time.perf_counter()
        inputs = parse_beatmap_file(beatmap_file)  
        end_time = time.perf_counter()
        parse_times["parse_beatmap_file"] += end_time - start_time

        start_time = time.perf_counter()
        targets = parse_osr_file(replay_file)  
        end_time = time.perf_counter()
        parse_times["parse_osr_file"] += end_time - start_time

        data = {"inputs": inputs, "targets": targets}
        np.save(f'{output_dir}\\processed_data_{idx}.npy', data)

        # Return both the data and the timing information
        return data, parse_times

    except Exception as e:
        print(f"error while parsing {beatmap_file} or {replay_file} - {str(e)}")
        return None, parse_times



def preprocess_data(df, beatmap_folder, replay_folder):
    files = list(zip(df['beatmapHash'].apply(lambda x: os.path.join(beatmap_folder, f'{x}.osu')),
                df['replayHash'].apply(lambda x: os.path.join(replay_folder, f'{x}.osr')),
                df.index))[:2000]

    parse_times = {
        "parse_beatmap_file": 0.0,
        "parse_osr_file": 0.0
    }

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.map(parse_files, files), total=df.shape[0]))

    # Combine the timing data from all processes
    for _, times in results:
        for func_name, time in times.items():
            parse_times[func_name] += time

    return parse_times

# def parse_files(args):

#     output_dir = f"D:\\osu!rdr_dataset\\preprocessed_datas"

#     beatmap_file, replay_file, idx = args
#     try:
#         inputs = parse_beatmap_file(beatmap_file)  
#         targets = parse_osr_file(replay_file)  

#         data = {"inputs": inputs, "targets": targets}
#         np.save(f'{output_dir}\\processed_data_{idx}.npy', data)

#     except Exception as e:
#         print(f"error while parsing {beatmap_file} or {replay_file} - {str(e)}")



# def preprocess_data(df, beatmap_folder, replay_folder):
#     files = list(zip(df['beatmapHash'].apply(lambda x: os.path.join(beatmap_folder, f'{x}.osu')),
#                 df['replayHash'].apply(lambda x: os.path.join(replay_folder, f'{x}.osr')),
#                 df.index))[:1000]
#     with multiprocessing.Pool() as pool:
#         list(tqdm(pool.map(parse_files, files), total=df.shape[0]))



if __name__ == "__main__":
    DATASET_PATH = "D:\\osu!rdr_dataset"
    beatmap_folder = f"{DATASET_PATH}\\beatmaps\\"
    replay_folder = f"{DATASET_PATH}\\replays\\osr\\"
    df = pd.read_csv(f'{DATASET_PATH}\\final_cleaned_index.csv')

    output_dir = f"{DATASET_PATH}\\preprocessed_datas"
    parse_times = preprocess_data(df, beatmap_folder, replay_folder)


    for func_name, total_time in parse_times.items():
        print(f"{func_name} took a total of {total_time} seconds")


    