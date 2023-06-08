import os
import torch
import multiprocessing as mp
import pandas as pd 
from DatasetLoader import DataGenerator

def process_data(start, end, df, beatmap_folder, replay_folder, tensor_dir, batch_size):
    data_generator = DataGenerator(df.iloc[start:end], beatmap_folder, replay_folder, batch_size)
    for i, data in enumerate(data_generator):
        input_data, target_data = data
        if input_data is not None and target_data is not None:
            print(f"writing {start} to {end} chunk")
            torch.save(input_data, os.path.join(tensor_dir, f'input_tensor_{start+i}.pt'))
            torch.save(target_data, os.path.join(tensor_dir, f'target_tensor_{start+i}.pt'))

if __name__ == "__main__":
    DATASET_PATH = "D:\\osu!rdr_dataset"
    beatmap_folder = f"{DATASET_PATH}\\beatmaps"
    replay_folder = f"{DATASET_PATH}\\replays\\osr"
    df = pd.read_csv(f'{DATASET_PATH}\\beatmap_replay_index.csv')
    batch_size = 32

    tensor_dir = f'{DATASET_PATH}\\preprocessed_tensors_with_time_left'
    os.makedirs(tensor_dir, exist_ok=True)

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    chunk_size = len(df) // num_processes
    chunks = [(i*chunk_size, (i+1)*chunk_size) for i in range(num_processes)]
    if len(df) % num_processes != 0:
        chunks[-1] = (chunks[-1][0], len(df))

    print("Starting multiprocessing")
    for chunk in chunks:
        pool.apply_async(process_data, args=(*chunk, df, beatmap_folder, replay_folder, tensor_dir, batch_size))

    pool.close()
    pool.join()
